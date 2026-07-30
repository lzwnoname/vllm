# vLLM 分布式并行策略源码分析

> 分析日期：2026-07-05  
> 代码基：vllm-project/vllm main 分支  

## 概览

vLLM 支持 **7 类并行策略**，它们在 `ParallelConfig`（`vllm/config/parallel.py`）中统一配置，运行时按固定的 rank 布局组织 GPU 进程组。

### 策略总览

| 并行策略 | 配置字段 | 缩写 | 适用模型 | 核心思想 |
|---------|---------|------|---------|---------|
| Tensor Parallel | `tensor_parallel_size` | TP | 所有模型 | 按矩阵维度切分权重 |
| Pipeline Parallel | `pipeline_parallel_size` | PP | 所有模型 | 按层切分模型 |
| Data Parallel | `data_parallel_size` | DP | MoE 模型 | 按 batch 切分 + expert shard |
| Expert Parallel | `enable_expert_parallel` | EP | MoE 模型 | 跨设备切分 experts |
| Prefill Context Parallel | `prefill_context_parallel_size` | PCP | 所有模型 | prefill 阶段 KV 分片 |
| Decode Context Parallel | `decode_context_parallel_size` | DCP | 所有模型 | decode 阶段 KV 分片 |
| Sequence Parallel | `use_sequence_parallel_moe` | SP | MoE+EP+TP+DP | MoE 输入 sequence 切分 |

---

## 一、配置入口：`ParallelConfig`

**文件**：`vllm/config/parallel.py`（第 117-1002 行）

```python
@config
class ParallelConfig:
    # === TP/PP/PCP ===
    tensor_parallel_size: int = 1        # TP 组大小
    pipeline_parallel_size: int = 1      # PP 组大小
    prefill_context_parallel_size: int = 1  # PCP 组大小

    # === DP ===
    data_parallel_size: int = 1          # DP 组数
    data_parallel_size_local: int = 1    # 本节点 DP 副本数
    data_parallel_rank: int = 0          # 当前 DP rank
    data_parallel_rank_local: int = None # 本地 DP rank（SPMD 模式）
    data_parallel_backend: str = "mp"    # DP 后端："mp" 或 "ray"
    data_parallel_external_lb: bool = False  # K8s 外部负载均衡
    data_parallel_hybrid_lb: bool = False    # 混合负载均衡

    # === EP ===
    enable_expert_parallel: bool = False # ★ 启用 EP（将 TP 转换为 EP）
    expert_placement_strategy: str = "linear"  # "linear" | "round_robin"
    all2all_backend: str = "allgather_reducescatter"  # 12 种 all2all 后端

    # === EPLB ===
    enable_eplb: bool = False            # 启用 Expert Parallel Load Balancing
    eplb_config: EPLBConfig              # EP 负载均衡参数
    enable_elastic_ep: bool = False      # 弹性 EP

    # === DCP ===
    decode_context_parallel_size: int = 1  # DCP 组大小（≤ TP）
    dcp_kv_cache_interleave_size: int = 1  # DCP KV cache 交错粒度
    dcp_comm_backend: str = "ag_rs"       # "ag_rs" | "a2a"
    cp_kv_cache_interleave_size: int = 1   # CP KV cache 交错粒度

    # === 其他 ===
    distributed_executor_backend: str     # "mp" | "ray" | "uni" | "external_launcher"
    disable_custom_all_reduce: bool = False
    ubatch_size: int = 0                  # ubatch（microbatch）大小
    enable_dbo: bool = False              # Dual Batch Overlap
    is_moe_model: bool | None = None      # 是否为 MoE 模型

    # 计算属性
    world_size: int                       # = TP × PP × PCP
    world_size_across_dp: int             # = TP × PP × PCP × DP
    data_parallel_index: int              # = data_parallel_rank
```

---

## 二、分布式初始化流程

**核心文件**：`vllm/distributed/parallel_state.py`

### 2.1 初始化入口

```
init_distributed_environment()     — 初始化全局 WORLD 组
  → initialize_model_parallel()    — 创建各并行子组
    → _TP, _DP, _DCP, _PCP, _PP, _EP, _EPLB
```

### 2.2 Rank 布局（5 维展开）

```python
# rank 编号按 (ExternalDP, DP, PP, PCP, TP) 顺序展开
# TP 在最内层 → 相邻 rank 属于同一 TP 组
all_ranks = torch.arange(world_size).reshape(
    -1,                # ExternalDP（batch 维度）
    data_parallel_size,
    pipeline_parallel_size,
    prefill_context_parallel_size,
    tensor_parallel_size,
)
```

### 2.3 各组创建逻辑

```python
# TP 组：TP 在最内层，直接 view+unbind
group_ranks = all_ranks.view(-1, tp_size).unbind(0)
_TP = init_group(group_ranks, "tp")

# DCP 组：复用 TP 组内的 GPU（不改变 world_size）
group_ranks = all_ranks.reshape(-1, dcp_size).unbind(0)
_DCP = init_group(group_ranks, "dcp")

# PCP 组：交换 TP/PCP 维度使 PCP 在最内层
group_ranks = all_ranks.transpose(3, 4).reshape(-1, pcp_size).unbind(0)
_PCP = init_group(group_ranks, "pcp")

# PP 组：将 PP 维度转置到最内层
group_ranks = all_ranks.transpose(2, 4).reshape(-1, pp_size).unbind(0)
_PP = init_group(group_ranks, "pp")

# DP 组：将 DP 维度转置到最内层
group_ranks = all_ranks.transpose(1, 4).reshape(-1, dp_size).unbind(0)
_DP = init_group(group_ranks, "dp")

# EP 组：跨越 DP × PCP × TP 的组（排除 PP）
group_ranks = all_ranks.transpose(1, 2).reshape(
    -1, dp_size * pcp_size * tp_size
).unbind(0)
_EP = init_group(group_ranks, "ep")

# EPLB 组：与 EP 相同 rank 组成，但使用独立 ProcessGroup
_EPLB = init_group(group_ranks, "eplb")
```

### 2.4 具体示例

**8 GPUs，TP=2，PP=4**：
- 4 个 TP 组：`[g0,g1]`, `[g2,g3]`, `[g4,g5]`, `[g6,g7]`
- 2 个 PP 组：`[g0,g2,g4,g6]`, `[g1,g3,g5,g7]`

---

## 三、Tensor Parallel (TP) — 张量并行

### 3.1 配置

```python
tensor_parallel_size: int = 1   # TP 组大小
```

### 3.2 核心实现

| 文件 | 说明 |
|------|------|
| `vllm/model_executor/layers/linear.py` | `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear` |
| `vllm/model_executor/layers/vocab_parallel_embedding.py` | `VocabParallelEmbedding`, `ParallelLMHead` 等 |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | TP 组的自定义 all-reduce 优化 |
| `vllm/distributed/device_communicators/pynccl_allocator.py` | PyNCCL 通信器 |
| `vllm/lora/layers/column_parallel_linear.py` | LoRA TP 层 |

### 3.3 影响的层

```
Embedding 层: VocabParallelEmbedding（按词表维度切分）
Attention 层: QKVParallelLinear（Q/K/V/O 投影切分）
MLP 层:     ColumnParallelLinear（列切分）+ RowParallelLinear（行切分）
LM Head:    ParallelLMHead（按词表维度切分）
```

### 3.4 TP 中的 all-reduce

- **ColumnParallelLinear**：输出需要 all-reduce（各 rank 持有部分 hidden dim）
- **RowParallelLinear**：输入已切分，输出各 rank 一致
- **Custom All-Reduce**：vLLM 实现了自己的 all-reduce kernel（`custom_all_reduce.py`），比 NCCL 更高效（单机场景）
- **disable_custom_all_reduce=True**：多节点时自动禁用，回退到 NCCL

---

## 四、Pipeline Parallel (PP) — 流水线并行

### 4.1 配置

```python
pipeline_parallel_size: int = 1   # PP 组大小
```

### 4.2 核心实现

| 文件 | 说明 |
|------|------|
| `vllm/v1/worker/gpu/pp_utils.py` | `PPHandler` 类：处理采样 token 的广播/接收 |
| `vllm/distributed/utils.py` | `get_pp_indices()`：计算每层归属的 PP rank |
| `vllm/model_executor/models/transformers/base.py` | 模型 PP 分片逻辑（`split_model_between_pipeline_stages`） |

### 4.3 PPHandler 工作原理

```
【每个 step slot 的前向流程】
Step T (last rank):
  → broadcast() 在 side stream 上将 sampled_tokens 广播给所有 PP rank

Step T + pp_size:
  → get_prev_sampled_outputs() 消费 T 时刻的采样输出
  → receive() 在 side stream 上接收来自 last rank 的采样 tokens
```

- **关键设计**：在 side stream 上运行 PP 通信，避免阻塞默认流
- **FIFO 队列**：预填充 `pp_size` 个 `None` 占位符，实现 T 步延迟匹配
- **Generation counter**：用于检测请求是否已被释放，避免消费过期数据

### 4.4 兼容性

- PP 与弹性 EP **不兼容**（`elastic_ep` 要求 `pipeline_parallel_size == 1`）
- PP 与 encoder-decoder 模型兼容，但与 KV connector 不兼容

---

## 五、Data Parallel (DP) — 数据并行

### 5.1 设计理念

vLLM 的 DP 与常见的"每个副本独立运行"不同，主要用于 **MoE 模型的 Expert Sharding**：

```
DP 的核心作用：
1. 将 DP rank 集合作为 expert 的额外分片维度
2. EP 的实际设备数 = DP × PCP × TP
3. 通过 all-reduce 协调各 DP rank 的 batch token 数量
```

对于 **非 MoE 模型**，`data_parallel_size` 强制为 1（非 MoE 模型应使用独立 vLLM 实例）。

### 5.2 配置

```python
data_parallel_size: int = 1
data_parallel_size_local: int = 1     # 本节点 DP 副本数
data_parallel_backend: str = "mp"     # "mp" | "ray"
data_parallel_external_lb: bool       # 外部 LB（K8s 一 Pod 一 rank）
data_parallel_hybrid_lb: bool         # 混合 LB
data_parallel_index: int              # = data_parallel_rank
```

### 5.3 核心实现

| 文件 | 说明 |
|------|------|
| `vllm/v1/worker/dp_utils.py` | `coordinate_batch_across_dp()`：DP 协调 batch 大小 |
| `vllm/v1/engine/core.py` | `DPEngineCoreProc`：DP Engine Core 进程 |
| `vllm/v1/worker/gpu/dp_utils.py` | GPU 层面 DP padding 和 CUDA graph 同步 |
| `vllm/forward_context.py` | `DPMetadata` 类：DP 元数据 |

### 5.4 DP 协调流程

```
run_busy_loop():
  1. coordinate_batch_across_dp()      — all-reduce 协调是否做 microbatch
  2. _has_global_unfinished_reqs()     — 每 32 步同步各 rank 是否完成
  3. _should_throttle_prefills()       — 按 cadence 节流 prefill
  4. 发布 request counts 给 coordinator
```

### 5.5 DP 三种负载均衡模式

| 模式 | 配置 | 说明 |
|------|------|------|
| **内部 LB** | 默认 | vLLM Client 管理所有 local+remote EngineCore |
| **外部 LB** | `data_parallel_external_lb=True` | K8s 一 Pod 一 rank，外部 LB 分发请求 |
| **混合 LB** | `data_parallel_hybrid_lb=True` | vLLM 管理本地 DP rank 内 LB，外部 LB 管理节点间 |

---

## 六、Expert Parallel (EP) — 专家并行

### 6.1 设计理念

EP 是一个**逻辑并行策略**而非物理分组——它没有独立的 GPU rank 组，而是**将 TP 转换为 EP**：

```
启用 EP 前：TP 在 MoE 层内切分 expert 权重
启用 EP 后：TP → 1（MoE 层内不切分），每个设备持有完整的某些 expert
```

### 6.2 TP → EP 转换

**核心类**：`FusedMoEParallelConfig.make()`（`vllm/model_executor/layers/fused_moe/config.py` 第 1110-1237 行）

```
FusedMoEParallelConfig.make(tp_size, pcp_size, dp_size, sp_size, parallel_config):

if not enable_expert_parallel:
    # 传统 TP 模式
    tp_size = flatten(dp_size * pcp_size * tp_size)  # 合并所有维度
    ep_size = 1

else:  # enable_expert_parallel = True
    # EP 模式：TP 转换为 EP
    ep_size = flatten(dp_size * pcp_size * tp_size)
    ep_rank = flatten_rank
    tp_size = 1   # MoE 层内不再切分
    tp_rank = 0
```

**示例对比**：

```
场景：TP=2, DP=2, EP=True
┌──────────┬─────────────────────────────────────────────┐
│ 模式     │ device 配置                                  │
├──────────┼─────────────────────────────────────────────┤
│ 非 EP    │ TP={4,0}, DP={2,0}, EP={1,0}  ← 4路TP切分  │
│ EP=True  │ TP={1,0}, DP={2,0}, EP={4,0}  ← 4路EP切分  │
└──────────┴─────────────────────────────────────────────┘
```

### 6.3 All2All 后端

`FusedMoEParallelConfig` 支持 12 种 all2all 通信后端：

```python
all2all_backend: All2AllBackend = "allgather_reducescatter"
# 选项:
# "allgather_reducescatter"    — 默认：AllGather + ReduceScatter
# "deepep_high_throughput"     — DeepEP 高吞吐量内核
# "deepep_low_latency"         — DeepEP 低延迟内核
# "deepep_v2"                  — DeepEP v2
# "mori_high_throughput"       — MoRI 高吞吐（ROCm） 
# "mori_low_latency"           — MoRI 低延迟（ROCm）
# "nixl_ep"                    — NIXL-EP
# "flashinfer_nvlink_two_sided"  — FlashInfer NVLink 双向
# "flashinfer_nvlink_one_sided"  — FlashInfer NVLink 单向
```

### 6.4 Expert Placement Strategy

```python
expert_placement_strategy: ExpertPlacementStrategy = "linear"
```

- **`linear`**：连续放置。4 experts, 2 ranks → rank0: `[0,1]`, rank1: `[2,3]`
- **`round_robin`**：轮询放置。4 experts, 2 ranks → rank0: `[0,2]`, rank1: `[1,3]`

Round-robin 对分组 expert 模型的负载均衡更友好（无冗余 expert 时）。

---

## 七、Sequence Parallel (SP) — 序列并行（MoE 专用）

### 7.1 触发条件

**`ParallelConfig.use_sequence_parallel_moe`**（第 641-656 行）：

```python
@property
def use_sequence_parallel_moe(self) -> bool:
    return (
        self.all2all_backend in (
            "allgather_reducescatter",
            "deepep_high_throughput", "deepep_low_latency",
            "mori_high_throughput", "mori_low_latency",
            "nixl_ep",
        )
        and self.enable_expert_parallel
        and self.tensor_parallel_size > 1
        and self.data_parallel_size > 1
    )
```

### 7.2 设计原因

当使用 EP + DeepEP All2All 时，TP 组的 all-reduce（在 o_proj 末尾）会导致 **输入被复制到每个 TP rank**。如果有 EP + TP 同时存在，这会导致：

- **无 SP**：MoE 输入在每个 TP rank 上完全复制 → 重复计算和通信
- **有 SP**：MoE 输入在 TP rank 间按 sequence 维度切分 → 只处理自己负责的 tokens

### 7.3 DPMetadata 中的 SP 支持

```python
# vllm/forward_context.py

class DPMetadata:
    num_tokens_across_dp_cpu: torch.Tensor   # 跨 DP rank 的 token 数
    local_sizes: list[int] | None = None      # SP 切分后的 local sizes

    def sp_local_sizes(self, sp_size):   # context manager
        # 将 num_tokens 按 sp_size 切分
        ...

    def cu_tokens_across_sp(self, sp_size):    # 跨 SP rank 累积 token 数
        ...
```

---

## 八、Context Parallel (PCP / DCP)

### 8.1 概念

Context Parallel 在 prefill 和 decode 阶段分别存在：

| 类型 | 配置 | 最大大小 | 是否改变 world_size |
|------|------|---------|-------------------|
| **PCP** (Prefill CP) | `prefill_context_parallel_size` | 无限制 | **是**（增加 world_size） |
| **DCP** (Decode CP) | `decode_context_parallel_size` | ≤ TP | **否**（复用 TP GPU） |

### 8.2 配置详情

```python
prefill_context_parallel_size: int = 1     # PCP 组大小
decode_context_parallel_size: int = 1      # DCP 组大小（必须整除 tp_size）
cp_kv_cache_interleave_size: int = 1       # KV cache 交错粒度
dcp_comm_backend: DCPCommBackend = "ag_rs" # DCP 通信后端
```

### 8.3 DCP 通信后端

| 后端 | 说明 | NCCL 调用次数 |
|------|------|--------------|
| `ag_rs` | AllGather + ReduceScatter | 3 次/层 |
| `a2a` | All-to-All + Triton kernel combine | 2 次/层（MLA 模型优化） |

### 8.4 KV Cache 交错存储

```
cp_kv_cache_interleave_size = 1  → token 级交错
  例：total_cp_rank = 0,1,2,3
      token 0 → rank 0, token 1 → rank 1, token 2 → rank 2, ...

cp_kv_cache_interleave_size = block_size → block 级交错
  例：先填满 (rank i, block j)，再填 (rank i+1, block j)
```

### 8.5 DCP 兼容性验证

`vllm/v1/worker/cp_utils.py` 中的 `check_attention_cp_compatibility()`：
- DCP 要求 attention 实现在 decode 期间返回 softmax LSE
- PCP 要求 attention 实现声明 `supports_pcp`
- MTP + `interleave_size > 1` 需要额外支持

---

## 九、EPLB — Expert Parallel Load Balancing

### 9.1 配置

```python
enable_eplb: bool = False
eplb_config: EPLBConfig = EPLBConfig(
    window_size: int = 1000           # expert 负载记录窗口
    step_interval: int = 3000         # expert 重排间隔
    num_redundant_experts: int = 0    # 冗余 expert 数量
    log_balancedness: bool = False    # 记录负载均衡性
    use_async: bool = True            # 异步 EPLB
    policy: EPLBPolicyOption = "default"
    communicator: EPLBCommunicatorBackend = None  # "nixl" | "pynccl" | "torch_gloo"
)
```

### 9.2 约束条件

- 要求 `enable_expert_parallel=True`
- 要求 `TP * DP > 1`
- 异步 EPLB 仅支持 `nixl` 或 `torch_gloo` 通信器（不与 NCCL 多流冲突）
- `elastic_ep=True` 时，优先选择 `nixl` 或 `pynccl`

### 9.3 EPLB 通信器选择（自动逻辑）

```python
if enable_eplb and communicator is None:
    if nixl_available:        → "nixl"    # 零拷贝 RDMA
    elif enable_elastic_ep:   → "pynccl"  # stateless groups
    else:                     → "torch_gloo"  # 静态 EP
```

---

## 十、弹性 EP (Elastic Expert Parallel)

### 10.1 配置

```python
enable_elastic_ep: bool = False
```

### 10.2 特点

- 允许运行时**动态增减 EP rank**（通过 stateless NCCL group）
- DP/EP/EPLB 组使用 `_init_stateless_group()` 创建（通过 TCPStore 协调端口）
- 要求 `eplb.enable=True`，`pp_size=1`
- 不与 `data_parallel_external_lb` / `data_parallel_hybrid_lb` 兼容

---

## 十一、Dual Batch Overlap (DBO)

### 11.1 配置

```python
enable_dbo: bool = False
ubatch_size: int = 0                  # ubatch 数量
dbo_decode_token_threshold: int = 32   # decode microbatch 阈值
dbo_prefill_token_threshold: int = 512 # prefill microbatch 阈值
```

### 11.2 机制

- DBO 将 batch 拆分为多个 microbatch，通过 overlay 计算和通信
- `ubatch_size=0` 且 `enable_dbo=True` → ubatch 数 = 2
- Token 数超过阈值时触发 microbatching

---

## 十二、并行策略组合关系

### 12.1 维度依赖

```
world_size = PP × TP × PCP
            └──────────┘ 组成一个 engine core

跨 engine core 的 DP 维度:
world_size_across_dp = PP × TP × PCP × DP

EP 跨设备维度（只在 MoE 层内生效）:
ep_size = DP × PCP × TP
```

### 12.2 典型部署场景

| 场景 | 配置 | 模型示例 |
|------|------|---------|
| 单 GPU | TP=1, PP=1 | 小模型 |
| 单节点多 GPU | TP=4 | Llama-70B |
| 多节点 | TP=4, PP=2 | Llama-405B |
| MoE 小规模 | TP=4, DP=1, EP=True | Mixtral-8x7B |
| MoE 中规模 | TP=1, DP=4, EP=True | Mixtral-8x7B 4 路 EP |
| MoE 大规模 | TP=2, DP=2, EP=True, EPLB=True | DeepSeek-V3 |
| 超大规模 | TP=4, DP=4, PP=2, EP=True, EPLB=True, SP | DeepSeek-V3 多节点 |
| P/D 分离 | TP=2 + kv_connector | 需控制 tail ITL 的场景 |

### 12.3 互斥/依赖关系

| 约束 | 说明 |
|------|------|
| `DCP ≤ TP` | DCP 复用 TP GPU，`tp_size` 必须被 `dcp_size` 整除 |
| `Elastic EP → EPLB=True, PP=1` | 弹性 EP 要求 EPLB 且不能有 PP |
| `EPLB → EP=True, TP×DP>1` | EPLB 要求 EP 已启用且有多设备 |
| `SP(use_sequence_parallel_moe) → EP + TP>1 + DP>1 + 特定 all2all` | |
| 非 MoE 的 DP | 强制 `data_parallel_size=1`，应使用独立实例 |

---

## 十三、关键文件索引

### 配置层

| 文件 | 内容 |
|------|------|
| `vllm/config/parallel.py` | ★ `ParallelConfig`, `EPLBConfig`：所有并行策略配置 |
| `vllm/config/model.py` | 模型配置（`is_moe_model` 判断） |
| `vllm/model_executor/layers/fused_moe/config.py` | ★ `FusedMoEParallelConfig.make()`：TP→EP 转换 + all2all 后端 |

### 分布式基础设施

| 文件 | 内容 |
|------|------|
| `vllm/distributed/parallel_state.py` | ★ `initialize_model_parallel()`：rank 布局 + 各组创建 |
| `vllm/distributed/utils.py` | `StatelessProcessGroup`、PP 层索引计算 |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | TP 自定义 all-reduce |
| `vllm/model_executor/layers/linear.py` | `ColumnParallelLinear`、`RowParallelLinear`、`QKVParallelLinear` |
| `vllm/model_executor/layers/vocab_parallel_embedding.py` | TP embedding 层 |

### 运行时调度

| 文件 | 内容 |
|------|------|
| `vllm/v1/worker/dp_utils.py` | `coordinate_batch_across_dp()`：DP 协调 |
| `vllm/v1/engine/core.py` | `DPEngineCoreProc`：DP Engine Core |
| `vllm/v1/worker/gpu/pp_utils.py` | `PPHandler`：PP 流水线协调 |
| `vllm/v1/worker/cp_utils.py` | CP 兼容性检查 |
| `vllm/forward_context.py` | ★ `DPMetadata`：DP/SP token 计数元数据 |

### 文档

| 文件 | 内容 |
|------|------|
| `docs/design/parallel_arch.md` | 并行架构设计文档 |
| `docs/features/disagg_prefill.md` | PD 分离文档 |

---

## 十四、阅读建议

### 入门路径

1. **`config/parallel.py`** → 了解所有配置项及其验证关系
2. **`distributed/parallel_state.py`** → 理解 rank 布局和 group 创建
3. **`layers/fused_moe/config.py::FusedMoEParallelConfig.make()`** → 理解 TP→EP 转换
4. **`v1/worker/dp_utils.py`** → 理解 DP 协调机制
5. **`v1/worker/gpu/pp_utils.py`** → 理解 PP 流水线

### 深入路径

- **TP 实现** → `model_executor/layers/linear.py` + `device_communicators/`
- **EP 运行时** → `model_executor/layers/fused_moe/` + 各种 all2all 后端
- **DP 完整流程** → `v1/engine/core.py::DPEngineCoreProc` + `v1/worker/dp_utils.py`
- **Elastic EP** → `parallel_state.py` 中的 `_init_stateless_group()`
- **All2All 加速** → deep_ep, flashinfer, mori 等第三方集成
