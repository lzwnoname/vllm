# vLLM 异步调度（Async Scheduling）源码分析

> 分析日期：2026-07-06  
> 代码基：vllm-project/vllm main 分支  

## 一、核心概念：异步调度解决什么问题？

### 同步调度的瓶颈

```
同步调度 (async_scheduling=False):
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │schedule()│ → │execute() │ → │update()  │ → 重复
  └──────────┘    └──────────┘    └──────────┘
   ↑ 这一步在等待 GPU 执行完成，CPU 空闲

  schedule() 调用 update_from_output() 吸收上一轮的采样输出后才能继续
  → schedule 和 execute 串行，GPU 执行期间 CPU 空等
```

### 异步调度的改进

```
异步调度 (async_scheduling=True):
  时间线 ──────────────────────────────────────────▶

  Step N:   [schedule()] [execute()────────────] [update_from_output()]
  Step N+1:             [schedule()] [execute()────────────] [update()]
                               ↑
                    关键：schedule() 不用等上一轮的
                    update_from_output() 完成！

  schedule() 用 placeholder 标记"在途"的 token，
  不等 GPU 执行完就开始排下一轮的 batch
```

**本质**：将调度（CPU 密集）和模型执行（GPU 密集）**流水线化**，掩盖调度延迟，减少 GPU 空闲气泡。

---

## 二、启用方式

### 2.1 配置

```bash
# CLI 参数
--async-scheduling      # 强制启用
--no-async-scheduling   # 强制禁用
```

**`SchedulerConfig.async_scheduling`**（`vllm/config/scheduler.py`）：

```python
async_scheduling: bool | None = None  # None = 自动推断
```

### 2.2 自动推断逻辑（`vllm/config/vllm.py`，第 992-1044 行）

```python
if scheduler_config.async_scheduling is None:
    # 默认启用，以下情况例外：
    if is_pooling_model:          → False   # pooling 模型不支持
    if spec_method not in compatible: → False  # 不兼容的投机解码
    if not executor_supports:     → False   # executor 不支持
    else:                         → True    # ★ 默认开启
```

### 2.3 调度器类选择（`vllm/config/scheduler.py`，第 180-201 行）

```python
def get_scheduler_cls(self):
    if self.async_scheduling:
        return AsyncScheduler     # vllm/v1/core/sched/async_scheduler.py
    return Scheduler              # vllm/v1/core/sched/scheduler.py
```

---

## 三、核心机制：Placeholder 模式

### 3.1 原理

异步调度的核心是**不等采样结果就预占位**：

```
Step N:
  schedule() → 发现请求 A 有 3 个 decode token 待生成
  → 在 _update_after_schedule() 中：
    request.num_output_placeholders += 3   ← "我预支了 3 个 token"
    request.spec_token_ids = [-1, -1, -1] ← placeholder
  → 发送 scheduler_output 给 GPU 执行
  → GPU 在运行 Step N 的 execute()

Step N+1（GPU 还在跑 Step N）:
  schedule() → 再次调度请求 A
  → 检查 num_output_placeholders > 0
  → 可以先调度预填充或排队，等 placeholder 被消费

Step N 的 update_from_output():
  → 收到 GPU 输出 [token_42, token_73, token_15]
  → num_output_placeholders -= 3          ← 兑现
  → 将真实 token 填入请求状态
```

### 3.2 `AsyncScheduler` 源码（`async_scheduler.py`，全部 76 行）

```python
class AsyncScheduler(Scheduler):
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """每次 schedule() 末尾调用，标记在途 token"""
        super()._update_after_schedule(scheduler_output)
        
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue  # prefill 不需要 placeholder
            
            # 增加 placeholder 计数
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # 预填 spec token placeholder
            request.spec_token_ids = self._spec_token_placeholders  # list of -1
            
            # PP 模式下标记 decode 资格步
            if self.use_v2_model_runner:
                request.next_decode_eligible_step = (
                    self.current_step + self.pp_size
                )

    def _update_request_with_output(self, request, new_token_ids):
        """收到 GPU 输出后，消费 placeholder"""
        if request.async_tokens_to_discard > 0:
            request.async_tokens_to_discard -= 1  # 丢弃过期的异步帧
            return [], False

        new_token_ids, stopped = super()._update_request_with_output(...)
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0
        
        # 缓存新生成的 KV blocks
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request,
                request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
```

---

## 四、在 Engine 主循环中的表现

### 4.1 主循环（`core.py`）

```python
def run_busy_loop(self):
    while self._handle_shutdown():
        self._process_input_queue()       # 1. 收新请求
        self._process_engine_step()       # 2. 执行一步

def _process_engine_step(self):
    outputs, model_executed = self.step_fn()  # schedule + execute + update
    
    for output in outputs.items():
        self.output_queue.put_nowait(output)  # 异步输出
    
    self.post_step(model_executed)
    
    # 关键：如果没有 model 执行但还有 scheduler 工作
    # （如 WAITING_FOR_REMOTE_KVS），让出 GIL 给后台传输线程
    if not model_executed and self.scheduler.has_requests():
        time.sleep(0.001)
    
    return model_executed
```

### 4.2 `step()` 方法

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    if not self.scheduler.has_requests():
        return {}, False
    
    # 1. 调度（异步模式：不等上一步结果就出 scheduler_output）
    scheduler_output = self.scheduler.schedule()
    
    # 2. 异步提交给 executor
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    
    # 3. 获取 grammar bitmask（与模型执行并行）
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    
    # 4. 等待 GPU 完成
    model_output = future.result()
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)
    
    # 5. 处理中断请求
    self._process_aborts_queue()
    
    # 6. 吸收模型输出（消费 placeholder）
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )
    
    return engine_core_outputs, ...
```

### 4.3 `post_step` 中的异步差异

```python
def post_step(self, model_executed):
    # 同步模式：post_step 中更新 draft token ids
    if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
        draft_token_ids = self.model_executor.take_draft_token_ids()
        self.scheduler.update_draft_token_ids(draft_token_ids)
    
    # 异步模式：draft token ids 在 worker 进程中直接更新
    # （因为 scheduler 侧无法预知）
```

---

## 五、请求状态机中的异步路径

### 5.1 `WAITING_FOR_REMOTE_KVS` 状态

这是 PD 分离（disaggregated prefill）特有的异步状态：

```python
# scheduler.py 中的调度流程（schedule() 方法）

if self.connector is not None:
    ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
        request, num_new_local_computed_tokens
    )

    if ext_tokens is None:
        # connector 还没准备好，请求推迟
        request_queue.pop_request()
        step_skipped_waiting.prepend_request(request)
        continue

    if load_kv_async:
        # ★ 标记为 WAITING_FOR_REMOTE_KVS
        # 请求在下一轮调度前不会加入 running queue
        # 等待 KV connector 完成远程加载
        ...
```

### 5.2 状态流转

```
新请求
  │
  ├─→ WAITING (排队中)
  │     │
  │     ├─→ 本地 prefix cache 命中 → PREEMPTED → RUNNING
  │     │
  │     └─→ 远程 KV cache 加载中 → WAITING_FOR_REMOTE_KVS
  │           │  (KV connector 异步加载完成)
  │           └─→ RUNNING
  │
  └─→ RUNNING
        │
        ├─→ async_scheduling=True:
        │   num_output_placeholders > 0 → 还在等待本轮输出
        │   但可以被再次调度（用 placeholder 预占）
        │
        └─→ FINISHED → 输出结果 → 释放资源
```

### 5.3 `async_tokens_to_discard` 机制

当 `reset_prefix_cache` 强制抢占请求时，在途的异步输出需要被丢弃：

```python
# 每次 _update_request_with_output 被调用时
if request.async_tokens_to_discard > 0:
    request.async_tokens_to_discard -= 1  # 丢弃一帧
    return [], False                       # 不更新请求状态
```

---

## 六、异步调度下的并发控制

### 6.1 Scheduler 内部的双缓冲设计

`scheduler.py` 中的 `_update_after_schedule` 和 `update_from_output` 构成"双缓冲"：

```python
# 内部请求字典，使用 update() 而不是直接替换
# 代码注释：
# "Use update() to preserve entries from the previous step
#  that have not yet been consumed by update_from_output
#  (async scheduling may call _update_after_schedule
#   before update_from_output)""

# 例如 running 队列：
self.running = {...}  # 用 .update() 增量更新，不是整个替换
```

### 6.2 Grammar bitmask 的并行计算

```python
# step() 方法中：
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
# ↑ grammar bitmask 计算和 GPU 执行并行
model_output = future.result()
```

---

## 七、与其他并行策略的交互

### 7.1 PP（Pipeline Parallel）

```python
# async_scheduler.py 第 46-49 行
if self.use_v2_model_runner:
    request.next_decode_eligible_step = self.current_step + self.pp_size
```

PP 模式下，一个请求在经过 `pp_size` 步后才能再次被调度为 decode，因为微批次在流水线中需要传播多个阶段。

### 7.2 DP（Data Parallel）

`DPEngineCoreProc.run_busy_loop()` 中，异步调度与 DP 协调结合：

```python
# DP 协调：各 rank 通过 all-reduce 同步
num_tokens_after_padding = coordinate_batch_across_dp(...)

# 如果无工作且无未完成请求，可以提前退出循环
if not has_unfinished_global:
    break
```

### 7.3 投机解码（Speculative Decoding）

异步调度对投机解码的影响：

```python
# 同步模式：draft token 在 post_step 由 scheduler 获取
self.scheduler.update_draft_token_ids(draft_token_ids)

# 异步模式：draft token 在 worker 进程内部更新
# scheduler 侧使用 placeholder [-1, -1, ...] 预占
```

---

## 八、不兼容项

| 场景 | 说明 |
|------|------|
| Pooling 模型 | 不支持异步调度（无 decode 阶段可流水线化） |
| 某些投机解码方法 | 需要同步获取 draft token 的方法不兼容 |
| 不支持的 executor | 某些 executor 不支持 `non_block=True` 的 `execute_model` |
| ROCm + DeepEP + DBO | 硬件限制，不可同时启用 |

---

## 九、关键文件索引

| 文件 | 内容 |
|------|------|
| `vllm/v1/core/sched/async_scheduler.py` | ★ `AsyncScheduler` 类（76行）：placeholder 管理 + 异步输出处理 |
| `vllm/v1/core/sched/scheduler.py` | 基类 `Scheduler`：`schedule()` + `update_from_output()` 主流程 |
| `vllm/v1/engine/core.py` | ★ `step()` + `run_busy_loop()` + `post_step()`：引擎主循环 |
| `vllm/v1/engine/core.py` | `_process_input_queue()` + `_process_engine_step()` |
| `vllm/config/scheduler.py` | `SchedulerConfig.async_scheduling` + `get_scheduler_cls()` |
| `vllm/config/vllm.py` | 自动推断逻辑（第 992-1044 行） |
| `vllm/v1/request.py` | `Request.num_output_placeholders`, `async_tokens_to_discard` |
| `vllm/v1/core/sched/output.py` | `SchedulerOutput`：`pending_structured_output_tokens` 等字段 |

---

## 十、一句话总结

> 异步调度通过 **Placeholder 模式**将 `schedule()` 和 `update_from_output()` 解耦：schedule 时不等待 GPU 输出，用虚拟 token（-1）预占位，实现调度与 GPU 执行的流水线化，填补 GPU 空闲气泡。
