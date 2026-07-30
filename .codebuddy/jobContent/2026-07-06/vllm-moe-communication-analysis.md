# vLLM MoE 并行通信策略分析

> 分析日期：2026-07-06  
> 代码基：vllm-project/vllm main 分支  

## 一、总览：MoE 通信的核心问题

MoE 并行中的通信围绕一个核心问题展开：**token 如何在 expert 之间路由**。

```
输入: N 个 token, 每个 token 选 top-K 个 expert（如 top-2）

  Rank 0: [token_E, token_F]  → expert 0,2           expert 3,5
  Rank 1: [token_G, token_H]  → expert 4,7           expert 1,6
                                              ↓
                               执行 all-to-all 通信
                                              ↓
  Rank 0: expert 0: [token_E] expert 2: [token_F]  expert 3: [] expert 5: []
  Rank 1: expert 4: [token_G] expert 7: [token_G]  expert 1: [] expert 6: [token_H]
                                              ↓
                               各 rank 计算本地 expert
                                              ↓
                               反向 all-to-all (combine)
                                              ↓
  Rank 0: token_E 的 expert 0,2 结果聚合, token_F 的 expert 3,5 结果聚合
  Rank 1: 同理
```

## 二、All2All Backend 全景图

vLLM 中 MoE 通信由 `ParallelConfig.all2all_backend` 控制，共 12 种选项（`vllm/config/parallel.py` 第 40-53 行）：

```python
All2AllBackend = Literal[
    "allgather_reducescatter",           # ★ 默认：AllGather + ReduceScatter
    "deepep_high_throughput",            # DeepEP 高吞吐
    "deepep_low_latency",               # DeepEP 低延迟
    "deepep_v2",                         # DeepEP v2 ElasticBuffer
    "flashinfer_nvlink_two_sided",       # FlashInfer NVLink 双向
    "flashinfer_nvlink_one_sided",       # FlashInfer NVLink 单向
    "mori_high_throughput",              # MoRI ROCm 高吞吐
    "mori_low_latency",                 # MoRI ROCm 低延迟
    "nixl_ep",                           # NIXL EP
    "naive",    # 已废弃 → 回退到 allgather_reducescatter
    "pplx",     # 已废弃 → 回退到 allgather_reducescatter
]
```

### 各后端适用场景速查

| 后端 | 适用硬件 | 节点规模 | 特点 |
|------|---------|---------|------|
| `allgather_reducescatter` | 通用 | 单/多节点 | NCCL 原生通信，兼容性最好 |
| `deepep_high_throughput` | NVIDIA (NVLink+RDMA) | 多节点 | 高吞吐，优化大 batch prefill |
| `deepep_low_latency` | NVIDIA (NVLink+RDMA) | 多节点 | 低延迟，优化 decode |
| `deepep_v2` | NVIDIA | 多节点 | 统一 ElasticBuffer API |
| `flashinfer_*` | NVIDIA (NVLink) | 单节点 | NVLink 优化，零拷贝 P2P |
| `mori_*` | AMD (ROCm) | 多节点 | ROCm 平台专用 |
| `nixl_ep` | NVIDIA (UCX/RDMA) | 多节点 | RDMA 零拷贝传输 |

---

## 三、Dispatch 前的路由阶段

在通信之前，每个 MoE 层先执行 **gate/router 计算**，决定每个 token 去哪几个 expert。

### 3.1 路由策略（`fused_moe/config.py` 第 100-129 行）

```python
class RoutingMethodType(IntEnum):
    Default = 0          # Softmax → TopK
    Renormalize = 1      # TopK → Softmax
    DeepSeekV3 = 2       # Sigmoid → BiasAdd → Top2 in group → Top4 groups → Top8
    Llama4 = 3           # Top1 → Sigmoid
    RenormalizeNaive = 4 # Softmax → TopK → Renormalize
    TopK = 5             # TopK (no softmax)
    SigmoidRenorm = 6    # Sigmoid → TopK → Renormalize
    MiniMax2 = 7         # Sigmoid + Bias → TopK → ScaledSumNormalize
    Sigmoid = 8          # Sigmoid → TopK
    Unspecified = 9
    DeepseekV4 = 100     # sqrtsoftplus + Bias + Normalize
    Custom = 101
    Simulated = 102
```

### 3.2 路由表（Round-Robin）

当使用 DeepEP LL / NIXL EP 时，需要 round-robin 路由表（`FusedMoEParallelConfig` 第 1073-1074 行）：

```python
@property
def needs_round_robin_routing_tables(self):
    return self.use_deepep_ll_kernels or self.use_nixl_ep_kernels
```

Round-robin 将全局 expert ID 映射到物理 rank，确保跨 rank 负载均衡。

---

## 四、AllGather + ReduceScatter（默认策略）

### 4.1 核心实现

**文件**：`vllm/distributed/device_communicators/all2all.py`

```python
class AgRsAll2AllManager(All2AllManagerBase):
    def dispatch(self, hidden_states, topk_weights, topk_ids,
                 is_sequence_parallel=False, extra_tensors=None):
        """AllGatherV 沿 dim=0 收集所有 rank 的 token"""
        dp_metadata = get_forward_context().dp_metadata
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()

        gathered = dist_group.all_gatherv(
            [hidden_states, topk_weights, topk_ids],
            dim=0, sizes=sizes
        )
        # 各 rank 现在拥有所有 token → 过滤出自己负责的 expert

    def combine(self, hidden_states, is_sequence_parallel=False):
        """ReduceScatterV 沿 dim=0 聚合结果回各 rank"""
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()

        return dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)
```

### 4.2 通信图

```
Dispatcher (AllGatherV):
  Rank 0: [t0, t1]  ──┐
  Rank 1: [t2, t3]  ──┤  all_gatherv(dim=0)
  Rank 2: [t4, t5]  ──┤  ──────────────────► 每个 rank 得到 [t0..t7]
  Rank 3: [t6, t7]  ──┘
  → 每 rank 过滤出自己负责的 expert 的 token
  → 计算 expert forward

Combiner (ReduceScatterV):
  Rank 0: [out_E0, out_E1]  ──┐
  Rank 1: [out_E2, out_E3]  ──┤  reduce_scatterv(dim=0)
  Rank 2: [out_E4, out_E5]  ──┤  ───────────────────► 每 rank 拿回自己的 token
  Rank 3: [out_E6, out_E7]  ──┘
```

### 4.3 优缺点

| 优点 | 缺点 |
|------|------|
| NCCL 原生，兼容性最好 | 带宽利用率不如专用内核 |
| 实现简单，无需额外依赖 | 需要全量 AllGather = O(N) 通信量 |
| 支持 SP（Sequence Parallel） | 不支持 Async dispatch/combine |

---

## 五、DeepEP 通信策略

DeepEP 是专门为大模型 MoE 推理设计的通信库，vLLM 支持三种模式。

### 5.1 架构特点

DeepEP 的 `buffer` 对象管理内部的缓冲区池和通信：

```python
# 每个 MoE 层创建一个 buffer
self.buffer = DeepEPBuffer(
    num_token_buffers=...,      # 缓冲区池大小
    num_nvl_buffers=...,        # NVLink 缓冲区数
    num_rdma_buffers=...,       # RDMA 缓冲区数
    buffer_size=...,            # 每个 buffer 的 token 容量
)
```

### 5.2 DeepEP High-Throughput

**文件**：`vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py`

面向 **prefill** 场景，大规模 token dispatch：

```python
# Dispatch 步骤
def _do_dispatch(self, tokens, rank_topk_ids, rank_topk_weights, ...):
    # 1. 计算 dispatch layout（决定每个 rank 发多少 token）
    layout = self.buffer.get_dispatch_layout(
        topk_idx=rank_topk_ids,
        num_experts=num_experts,
        async_finish=False,
    )
    
    # 2. 执行 dispatch（NVLink + RDMA 混合通信）
    (token_data, expert_topk_ids, expert_topk_weights,
     expert_num_tokens_per_expert_list, handle, event
    ) = self.buffer.dispatch(
        x=token_data,
        num_tokens_per_rank=layout.num_tokens_per_rank,
        num_tokens_per_rdma_rank=layout.num_tokens_per_rdma_rank,
        is_token_in_rank=layout.is_token_in_rank,
        topk_idx=rank_topk_ids,
        topk_weights=rank_topk_weights,
        config=DispatchConfig(...),  # 控制 NVLink/RDMA 并行度
    )

# Combine 步骤
def _finalize(self, output, fused_expert_output, topk_weights, topk_ids, ...):
    combined_x, _, event = self.buffer.combine(
        x=fused_expert_output,
        handle=handle,
        config=CombineConfig(...),
    )
    output.copy_(combined_x)
```

**关键特性**：
- **NVLink + RDMA 混合**：节点内用 NVLink，跨节点用 RDMA
- **异步 dispatch/combine**：支持 DBO（Dual Batch Overlap）微批
- **量化支持**：dispatch 支持 FP8、block quant 和 per-token quant
- **MNNVL 支持**：支持 NVIDIA multi-node NVLink (MNNVL) fabric

### 5.3 DeepEP Low-Latency

**文件**：`vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py`

面向 **decode** 场景，token 数少但延迟敏感：

```python
def prepare_async(self, a1, topk_weights, topk_ids, ...):
    # 映射到物理 expert ID（round-robin）
    dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)
    
    (expert_x, expert_num_tokens, handle, _, hook
    ) = self.buffer.low_latency_dispatch(
        a1,
        dispatch_topk_ids,
        self.max_tokens_per_rank,
        num_experts,
        use_fp8=self.use_fp8_dispatch,
        use_ue8m0=self.use_ue8m0_dispatch,
        async_finish=False,
    )
    # 返回 handle 用于后续 combine

def finalize_async(self, expert_out, topk_weights, topk_ids, handle, hook, ...):
    dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)
    
    combined_x, _, event = self.buffer.low_latency_combine(
        expert_out,
        dispatch_topk_ids,
        topk_weights,
        handle=handle,
        async_finish=False,
    )
```

**与 HT 的关键区别**：
- 使用 `low_latency_dispatch` / `low_latency_combine` API
- 需要 round-robin 路由表（物理 expert 映射）
- 优化小 token 量的通信延迟
- Batched activation format（E × max_tokens × K）

### 5.4 DeepEP v2

**文件**：`vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py`

使用 DeepEP v2 的统一 ElasticBuffer API，简化 buffer 管理：

```python
class DeepEPV2ElasticBufferWrapper:
    def create(self, num_ranks, buffer_size, num_nvl_buffers, num_rdma_buffers):
        self.elb = deepep_v2.ElasticBuffer(...)
        # 为每个 rank 创建连接
        for rank in range(num_ranks):
            self.elb.connect(rank, buffer_size, num_nvl_buffers, num_rdma_buffers)
    
    def dispatch(self, tokens, topk_ids, ...):
        return self.elb.dispatch(tokens, topk_ids, ...)
```

---

## 六、FlashInfer NVLink All2All

仅适用于**单节点 NVLink** 场景，利用 NVLink 的 P2P 带宽做零拷贝传输。

### 6.1 双向（Two-Sided）

**文件**：`vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_two_sided.py`

```python
def flashinfer_alltoall_dispatch(all2all_manager, ...):
    # 1. 准备：记录各 rank 的 token 分配
    MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
        global_num_tokens_cpu, x, gs, topk_ids, topk_weights,
        top_k, num_experts, quant_config, ...
    )
    # 2. 执行 NVLink P2P all-to-all
    MnnvlMoe.mnnvl_moe_alltoallv(...)

# Combine 类似，使用 mnnvl_moe_alltoallv 反向
```

### 6.2 单向（One-Sided）

**文件**：`vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py`

使用 FlashInfer 的高吞吐 `one_sided` API，单向 NVLink 传输更高效：

```python
# 使用 MnnvlMoe.for_one_sided_alltoall() 初始化的 manager
all2all_manager = MnnvlMoe.for_one_sided_alltoall(...)

# Dispatch 和 combine 使用 one-sided API
```

### 6.3 通信模式

```
双向 (Two-Sided):
  Rank 0 ←──NVLink──→ Rank 1  双方向同时传输
  适合通用场景

单向 (One-Sided):
  Rank 0 ──NVLink──→ Rank 1  单方向 Push
  更高吞吐，适合大 batch prefill
```

---

## 七、NIXL EP

**文件**：`vllm/model_executor/layers/fused_moe/prepare_finalize/nixl_ep.py`

基于 NIXL 的零拷贝 RDMA 传输：

```python
class NixlEPMoEPrepareAndFinalize:
    def prepare(self, a1, topk_weights, topk_ids, ...):
        # Round-robin 路由映射
        dispatch_topk_ids = self._map_to_physical_ids(topk_ids)
        
        # NIXL dispatch - 零拷贝 RDMA 传输
        expert_x, expert_num_tokens, handle = self.nixl_manager.dispatch(
            a1, dispatch_topk_ids, ...
        )
    
    def finalize(self, expert_out, topk_weights, topk_ids, handle, ...):
        # NIXL combine
        combined_out = self.nixl_manager.combine(
            expert_out, dispatch_topk_ids, topk_weights, handle
        )
```

**特点**：
- 使用 UCX/RDMA 做跨节点零拷贝传输
- 需要 round-robin 路由表
- Batched activation format
- 异步 dispatch/combine

---

## 八、MoRI（ROCm 专用）

**文件**：`vllm/model_executor/layers/fused_moe/prepare_finalize/mori.py`

AMD ROCm 平台的 EP 通信，使用 MoRI 的 InterNode 内核：

```python
# mori.py 中的 dispatch
def _mori_dispatch(self, hidden_states, topk_ids, topk_weights, ...):
    # 使用 MoRI InterNode V1（高吞吐）或 InterNode V1LL（低延迟）
    return mori_inter_node_v1_dispatch(...)
```

**两种模式**：
- `mori_high_throughput`：InterNodeV1，优化大 batch
- `mori_low_latency`：InterNodeV1LL，优化小 token 量

依赖 ROCm aiter fused_moe，仅支持 AMD GPU 集群。

---

## 九、Sequence Parallel 对 MoE 通信的影响

### 9.1 问题

TP 组的 all_reduce（在 o_proj 末尾）会导致 MoE 输入被复制到每个 TP rank。如果同时有 EP + TP：

```
无 SP：
  TP rank 0: [完整 batch 的 token] → 全部送 EP
  TP rank 1: [完整 batch 的 token] → 全部送 EP
  → token 被重复计算，且通信量翻倍
```

### 9.2 解决方案

`use_sequence_parallel_moe`（`vllm/config/parallel.py` 第 641-656 行）：

```python
@property
def use_sequence_parallel_moe(self) -> bool:
    return (
        self.all2all_backend in (
            "allgather_reducescatter", "deepep_high_throughput",
            "deepep_low_latency", "mori_high_throughput",
            "mori_low_latency", "nixl_ep",
        )
        and self.enable_expert_parallel
        and self.tensor_parallel_size > 1
        and self.data_parallel_size > 1
    )
```

SP 启用时，MoE 输入按 sequence 维度在各 TP rank 间切分，每个 rank 只处理自己负责的 token slice。

### 9.3 对通信的影响

```python
# dispatch 时使用 EP group（而非 DP group）
dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()

# EP group = DP × PCP × TP 所有 rank 的并集
# SP 场景下需要跨越 TP 维度做 dispatch/combine
```

---

## 十、Batched DP MoE

### 10.1 触发条件（`config/parallel.py` 第 658-668 行）

```python
@property
def use_batched_dp_moe(self) -> bool:
    return (
        self.all2all_backend in ("deepep_low_latency", "nixl_ep")
        and self.enable_expert_parallel
        and self.data_parallel_size > 1
    )
```

### 10.2 Batched Activation Format

将 token 按 expert 分组，reshape 为 `[num_experts, max_tokens_per_expert, hidden]` 格式：

```
传统格式:
  [token_0_for_E0, token_1_for_E0, token_2_for_E1, ...]
  → 需要 permute 操作重排

Batched 格式:
  Expert 0: [token_0, token_1]  ← 连续存储
  Expert 1: [token_2]
  Expert 2: [token_3, token_4, token_5]
  → 直接 batched GEMM，无需 permute
```

### 10.3 BatchedPrepareAndFinalize

**文件**：`vllm/model_executor/layers/fused_moe/prepare_finalize/batched.py`

```python
class BatchedPrepareAndFinalize:
    def prepare(self, a1, topk_weights, topk_ids, ...):
        # dispatch 后按 expert 分组排列
        # 输出格式: [E][T_expert][hidden]
        ...
    
    def finalize(self, expert_out, ...):
        # 从 batched 格式还原为原始 token 顺序
        ...
```

---

## 十一、总结对比

### 通信量化

设总 token 数 = N，top-K = K，rank 数 = R，hidden_dim = H：

| 策略 | Dispatch 通信量 | Combine 通信量 | 总通信量 |
|------|----------------|----------------|---------|
| AllGather+RS | `N×H` (all_gather) | `N×K×H` (reduce_scatter) | `N×H×(1+K)` |
| DeepEP HT | `N×K×H` (a2a) | `N×K×H` (a2a) | `2×N×K×H` |
| DeepEP LL | `N×K×H` (a2a) | `N×K×H` (a2a) | `2×N×K×H` |
| FlashInfer NVLink | P2P 零拷贝 | P2P 零拷贝 | 理论 0（含 kernel launch） |
| NIXL EP | RDMA 零拷贝 | RDMA 零拷贝 | 理论 0 |
| MoRI | 专用 a2a | 专用 a2a | `2×N×K×H` |

### 选型决策

```
是否有 NVLink？
├─ 是 → 单节点？→ FlashInfer (零拷贝)
│              → 多节点？→ DeepEP (NVLink+RDMA混合)
└─ 否 → 多节点？
         ├─ NVIDIA → DeepEP (纯 RDMA)
         ├─ AMD → MoRI
         └─ 通用 → AllGather+ReduceScatter (NCCL)
```

### Seq Parallel 与 Batched DP 的组合

| 场景 | 通信模式 |
|------|---------|
| EP + TP + **无 SP** | DP group all2all（TP 内 token 重复） |
| EP + TP + **有 SP** | EP group all2all（TP 间 token 去重） |
| EP + DP + **DeepEP LL** | Batched DP MoE（E×T×hidden 格式） |
| EP + DP + **DeepEP HT** | 标准 dispatch（token-major 格式） |
