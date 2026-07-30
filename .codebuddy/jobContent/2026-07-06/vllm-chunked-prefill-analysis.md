# vLLM Chunked Prefill 实现分析

> 分析日期：2026-07-06  
> 代码基：vllm-project/vllm main 分支  

## 一、核心概念：什么是 Chunked Prefill？

### 问题背景

LLM 推理有两种基本操作：
- **Prefill**：处理完整 prompt，计算所有 token 的 KV cache（计算密集）
- **Decode**：逐个生成 token（内存带宽密集）

```
传统调度（无 chunked prefill）:

  请求 A (prompt=8192 tokens)
    │
    ├─→ Step 1: [████████ prefill 8192 ████████] ← GPU 满载，但 TTFT 高
    │
    ├─→ Step 2: [decode A]
    ├─→ Step 3: [decode A]
    │   ...（请求 A 独占 GPU）

  问题: 长 prompt 的 prefill 耗时很长（数百 ms），
        其他请求的 decode 被阻塞，tail latency 差
```

### Chunked Prefill 的解决方案

将长 prompt **分块**处理，每步只 prefill 一部分，跟 decode 混合执行：

```
Chunked Prefill (max_num_batched_tokens=2048):

  请求 A (prompt=8192 tokens) + 请求 B (decoding)

    │
    ├─→ Step 1: [prefill A: 0-2047] [decode B]   ← A 的 1/4 + B 的 decode
    ├─→ Step 2: [prefill A: 2048-4095] [decode B]
    ├─→ Step 3: [prefill A: 4096-6143] [decode B]
    ├─→ Step 4: [prefill A: 6144-8191] [decode B]
    ├─→ Step 5: [decode A] [decode B]             ← A prefill 完成，开始 decode
```

**好处**：
- **降低 TTFT**：长 prompt 不再独占 GPU
- **降低 tail latency**：decode 请求不会因大 prefill 被阻塞
- **提高 GPU 利用率**：prefill（compute-bound）+ decode（memory-bound）混合执行

---

## 二、配置

**文件**：`vllm/config/scheduler.py`

```python
@config
class SchedulerConfig:
    enable_chunked_prefill: bool = True       # 默认启用
    """If True, prefill requests can be chunked based
    on the remaining max_num_batched_tokens."""

    max_num_batched_tokens: int = 2048        # 每个 step 的 token 预算
    """Maximum number of tokens that can be processed in a single iteration."""

    long_prefill_token_threshold: int = 0     # 长请求阈值（0=禁用）
    """For chunked prefill, a request is considered long if the prompt is
    longer than this number of tokens. 0 disables the cap (default)."""

    max_num_partial_prefills: int = 1         # 同时进行 chunked prefill 的请求数
    max_long_partial_prefills: int = 1        # 同时进行的"长"chunked prefill 数
```

### 关键约束

```python
# __post_init__ (第 273-284 行)
def verify_max_model_len(self, max_model_len):
    if not self.enable_chunked_prefill:
        # 禁用 chunked prefill 时，max_num_batched_tokens 必须 >= max_model_len
        assert self.max_num_batched_tokens >= max_model_len
```

即：禁用 chunked prefill 时，必须一次性 prefill 整个 prompt，所以 `max_num_batched_tokens` 必须 ≥ `max_model_len`。

---

## 三、调度器实现核心

### 3.1 设计理念（第 396-407 行）

```python
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and num_tokens_with_spec.
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec. This is general enough to cover
# chunked prefills, prefix caching, speculative decoding,
# and the "jump decoding" optimization in the future.
```

**关键洞察**：vLLM v1 调度器**没有 prefill/decode 阶段的概念**。所有请求只有两个状态：
- `num_computed_tokens`：已计算的 token 数
- `num_tokens_with_spec`：总共需要计算的 token 数

调度器每步只做一件事：**给请求分配 token 预算，让 `num_computed_tokens` 追上 `num_tokens_with_spec`**。Chunked prefill 只是这个通用机制的自然结果。

### 3.2 Token 预算机制

```python
def schedule(self, throttle_prefills=False):
    token_budget = self.max_num_scheduled_tokens  # 初始预算 = max_num_batched_tokens
    
    # First, schedule RUNNING requests (包括正在 decode 和正在 chunked prefill 的)
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]
        
        # 计算这个请求还需要多少 token
        num_new_tokens = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        
        # 受限于 long_prefill_token_threshold
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        
        # 受限于剩余预算
        num_new_tokens = min(num_new_tokens, token_budget)
        
        # 执行调度...
        token_budget -= num_new_tokens   # 扣减预算
    
    # Then, schedule WAITING requests (新请求)
    while request_queue and token_budget > 0:
        request = request_queue.get_request()
        num_new_tokens = request.num_tokens - num_computed_tokens
        
        # chunked prefill 的核心逻辑
        if not self.scheduler_config.enable_chunked_prefill \
           and num_new_tokens > token_budget:
            break  # 禁用 chunked prefill 时，预算不够就停
        
        num_new_tokens = min(num_new_tokens, token_budget)
        # 只调度 num_new_tokens 个 token，剩余留到后续 step
```

### 3.3 Chunked Prefill 的调度流程

```
请求 A (prompt=8192), max_num_batched_tokens=2048

Step 1 (A 新到达):
  num_computed_tokens = 0
  num_new_tokens = min(8192, 2048) = 2048
  → 调度 token 0-2047
  → num_computed_tokens 变为 2048
  → A 加入 _inflight_prefills（还没 prefill 完）

Step 2 (A 在 RUNNING 队列):
  num_computed_tokens = 2048
  num_new_tokens = min(8192-2048, 2048) = 2048
  → 调度 token 2048-4095
  → num_computed_tokens 变为 4096

Step 3: → 调度 token 4096-6143
Step 4: → 调度 token 6144-8191
  → num_computed_tokens = 8192 = num_prompt_tokens
  → is_prefill_chunk = False
  → 从 _inflight_prefills 移除
  → 开始 decode
```

### 3.4 `is_prefill_chunk` 的判断

```python
# vllm/v1/request.py
@property
def is_prefill_chunk(self) -> bool:
    """Whether this request is being chunked-prefilled."""
    return self.num_computed_tokens < self.num_tokens
```

在 `update_from_output` 中（第 1178-1191 行）：

```python
request.num_computed_tokens += num_scheduled_token

# 更新 is_prefill_chunk 状态
request.is_prefill_chunk = request.num_computed_tokens < (
    request.num_tokens + request.num_output_placeholders
)

# 从 in-flight prefill 集合中移除已完成的
if not request.is_prefill_chunk:
    self._inflight_prefills.discard(request)
```

### 3.5 `_inflight_prefills` 追踪

```python
# scheduler.py 第 996-998 行
# Only track requests that will still be prefilling after this chunk.
if num_computed_tokens + num_new_tokens < request.num_tokens:
    self._inflight_prefills.add(request)
```

这个集合用于：
- 限制并发 chunked prefill 数量（`max_num_partial_prefills`）
- DP 场景下的 prefill 节流

---

## 四、Prefill 与 Decode 的混合调度

### 4.1 两阶段调度

```
每个 step:
  1. 先调度 RUNNING 队列（正在 decode + 正在 chunked prefill）
  2. 再调度 WAITING 队列（新请求）
  共享同一个 token_budget
```

**示例**：

```
max_num_batched_tokens = 2048
Running: 请求 B (decode, 1 token)
Waiting: 请求 A (prompt=4096)

Step N:
  1. 调度 B: num_new_tokens = 1, token_budget = 2048-1 = 2047
  2. 调度 A: num_new_tokens = min(4096, 2047) = 2047
     → A 的 token 0-2046 被调度
  
  本 step 实际执行: [prefill A: 2047 tokens] + [decode B: 1 token]
```

### 4.2 DP 场景下的 Prefill 节流

```python
# scheduler.py 第 434-438 行
defer_prefills = (
    throttle_prefills and not self.prefill_capacity_bound
) and any(not r.is_prefill_chunk for r in self.running)

# 第 467-471 行
if defer_prefills and request.is_prefill_chunk:
    # DP prefill balancing: defer this in-progress prefill chunk to a
    # cadence-aligned step; decodes still run to fill this step.
    req_index += 1
    continue
```

在 DP 模式下，prefill 可能被推迟到 cadence 对齐的 step 执行，decode 仍然填充本 step。

---

## 五、Attention 层如何处理 Chunked Prefill

### 5.1 问题

Chunked prefill 的每个 chunk 不是从序列开头开始的。比如 prompt 有 8192 token，第 3 个 chunk 处理 token 4096-6143，此时 token 0-4095 的 KV cache 已经在之前步骤算好了。

**Attention 必须区分**：
- **Context KV**（已计算的，从 KV cache 读取）
- **Query KV**（本 chunk 新计算的）

### 5.2 FlashAttention 后端的实现

```python
# vllm/v1/attention/backends/flash_attn.py

# Prefill chunk 的 attention 分两部分:
# 1. Context attention: query=新token, key/value=KV cache 中的旧 token
# 2. Query attention: query=新token, key/value=新token (causal)

context_attn_out, context_lse = flash_attn_varlen_func(
    q=query,                          # 新 chunk 的 query
    k=key_cache, v=value_cache,       # KV cache 中的旧 token
    seqused_k=attn_metadata.context_lens_tensor,  # 每个 request 的 context 长度
    ...
)

query_attn_out, query_lse = flash_attn_varlen_func(
    q=query, k=key, v=value,          # 新 chunk 内部 causal attention
    ...
)

# 合并两部分结果
merge_attn_states(output, context_attn_out, context_lse,
                  query_attn_out, query_lse)
```

### 5.3 Attention Metadata

```python
# 关键字段
query_start_loc: torch.Tensor       # 每个 request 的 query 起始位置
context_lens_tensor: torch.Tensor    # 每个 request 的 context 长度
# ↑ 用于区分 context 和 query 部分
```

---

## 六、KV Cache 分配

### 6.1 增量分配

每个 chunk 只需要分配它对应位置的 KV cache block：

```
prompt = 8192 tokens, block_size = 16

Step 1 (chunk 0-2047):
  分配 block 0-127 (128 个 block)
  KV cache 状态: [■■■...■■■■□□□...□□□□] (前 128 block 被填充)

Step 2 (chunk 2048-4095):
  分配 block 128-255
  KV cache 状态: [■■■...■■■■■■■■...■■■■] (前 256 block 被填充)

Step 3-4: 类似
```

### 6.2 `update_state_after_alloc` 在 chunked prefill 中的作用

```python
# scheduler.py
# 每次调度后，connector 更新状态
if self.connector is not None:
    self.connector.update_state_after_alloc(
        request,
        self.kv_cache_manager.get_blocks(request_id),
        num_external_computed_tokens,
    )
```

对于 PD 分离场景，chunked prefill 的每个 chunk 计算完后，KV cache 都会通过 connector 传输给 decode 实例。

---

## 七、并发 Chunked Prefill 控制

### 7.1 `max_num_partial_prefills`

```python
max_num_partial_prefills: int = 1
"""For chunked prefill, the maximum number of sequences that can be
partially prefilled concurrently."""
```

默认值 1 表示**同时只有一个请求在 chunked prefill**。如果设为 2，可以有两个长 prompt 交错 prefill：

```
max_num_partial_prefills=2:

Step 1: [prefill A: 0-1023] [prefill B: 0-1023]
Step 2: [prefill A: 1024-2047] [prefill B: 1024-2047]
...
```

### 7.2 `max_long_partial_prefills`

```python
max_long_partial_prefills: int = 1
"""For chunked prefill, the maximum number of prompts longer than
long_prefill_token_threshold that will be prefilled concurrently."""
```

对"长"prompt（超过 `long_prefill_token_threshold`）的并发数有额外限制。

### 7.3 阈值自动设置

```python
# __post_init__ (第 257-259 行)
if self.max_num_partial_prefills > 1:
    if self.long_prefill_token_threshold == 0:
        self.long_prefill_token_threshold = int(max_model_len * 0.04)
```

启用多并发 chunked prefill 时，自动设置长 prompt 阈值为 `max_model_len * 4%`。

---

## 八、Chunk 大小的决定因素

```
最终 chunk 大小 = min(
    max_num_batched_tokens - 已分配给其他请求的 token 数,  # 剩余预算
    long_prefill_token_threshold,                          # 长请求阈值
    max_model_len - num_computed_tokens,                   # 剩余 prompt 长度
    # Mamba 模型还有 block_size 对齐约束
)
```

### 各种场景的 chunk 大小

| 场景 | max_num_batched_tokens | chunk 大小 |
|------|----------------------|------------|
| 短 prompt (512 tokens), 无其他请求 | 2048 | 512（一次性完成） |
| 长 prompt (8192 tokens), 无其他请求 | 2048 | 2048（分 4 个 chunk） |
| 长 prompt (8192), 有 decode 请求 | 2048 | 2048 - decode 数 |
| 长 prompt (8192), long_prefill_threshold=1024 | 2048 | 1024 |

---

## 九、与其他功能的交互

### 9.1 Prefix Cache

```
请求 A: prompt = [prefix_1024] + [new_7168]

Step 1:
  1. 查本地 prefix cache → 命中 [prefix_1024]
     num_computed_tokens = 1024
  2. chunked prefill: num_new_tokens = min(7168, 2048) = 2048
     → 调度 token 1024-3071
```

Prefix cache 命中后，`num_computed_tokens` 直接跳到命中长度，chunked prefill 只需处理剩余部分。

### 9.2 异步调度

```
异步模式下，chunked prefill 的 placeholder 机制:

Step N:
  schedule → A 的 chunk 2048-4095 被调度
  _update_after_schedule: 
    A.num_output_placeholders += 0  ← prefill chunk 不产生 output
  
Step N+1 (不等 Step N 完成):
  schedule → A 的 chunk 4096-6143 被调度
  （A 的 is_prefill_chunk=True，不涉及 output placeholder）
```

Chunked prefill 的 chunk 不生成 output token，所以**异步调度的 placeholder 机制只作用于 decode 请求**。

### 9.3 PD 分离

```
P 实例 chunked prefill + KV 传输:

Step 1: P prefill chunk 0-2047 → save_kv_layer → 传输 KV
Step 2: P prefill chunk 2048-4095 → save_kv_layer → 传输 KV
...

D 实例:
  等待所有 chunk 的 KV 都传输完成
  → get_num_new_matched_tokens() 返回完整 prompt 长度
  → 一次性加载所有 KV
  → 开始 decode
```

---

## 十、关键文件索引

| 文件 | 内容 |
|------|------|
| `vllm/config/scheduler.py` | ★ `SchedulerConfig`：`enable_chunked_prefill`, `max_num_batched_tokens` |
| `vllm/v1/core/sched/scheduler.py` | ★ `schedule()` 方法（第 396-1013 行）：核心调度逻辑 |
| `vllm/v1/core/sched/scheduler.py` | `_inflight_prefills` 追踪（第 996-998 行） |
| `vllm/v1/core/sched/scheduler.py` | `update_from_output` 更新 `is_prefill_chunk`（第 1178-1191 行） |
| `vllm/v1/request.py` | `Request.is_prefill_chunk`, `num_computed_tokens`, `num_tokens` |
| `vllm/v1/attention/backends/flash_attn.py` | ★ context attention + query attention 分离计算 |
| `vllm/v1/attention/backends/flashinfer.py` | 同上，FlashInfer 后端实现 |
| `vllm/v1/core/kv_cache_manager.py` | `get_computed_blocks()`：prefix cache + chunked prefill 的 KV 分配 |

---

## 十一、一句话总结

> Chunked prefill 的核心是 **token 预算机制**：每个 step 有 `max_num_batched_tokens` 的预算，调度器将其分配给 prefill chunk 和 decode 请求。长 prompt 被自然切分为多个 chunk，每个 chunk 只处理一部分 token，跟 decode 混合执行。Attention 层通过分离 context attention（读 KV cache）和 query attention（causal）来正确处理部分 prefill 的情况。
