# vLLM PD 分离（Disaggregated Prefill/Decode）源码导读

> 分析日期：2026-07-05  
> 代码基：vllm-project/vllm main 分支  

## 一、功能概述

PD 分离（Prefill-Decode Disaggregation）将 LLM 推理的 **prefill** 和 **decode** 阶段拆分到不同 vLLM 实例运行：

- **Prefill 实例**（KV Producer）：处理输入 prompt，生成 KV cache，通过 Connector 传输给 decode 实例
- **Decode 实例**（KV Consumer）：接收 KV cache，跳过 prefill 阶段直接进行 token 生成

配套功能：
- **EPD (Disaggregated Encoder)**：将多模态模型的视觉编码器也拆分到独立实例，对应的 EC (Encoder Cache) 传输机制

```
┌─────────────────┐     KV Cache      ┌─────────────────┐
│  Prefill 实例    │ ────────────────▶ │   Decode 实例    │
│  (KV Producer)  │    Connector      │  (KV Consumer)   │
└─────────────────┘                   └─────────────────┘
```

---

## 二、整体目录结构

```
vllm/
├── config/
│   └── kv_transfer.py                 # ★ KVTransferConfig 配置类
│
├── distributed/
│   ├── kv_transfer/                   # ★ PD 分离核心：KV Cache 传输框架
│   │   ├── README.md                  # 架构说明文档
│   │   ├── kv_transfer_state.py       # 全局状态 + 初始化入口
│   │   └── kv_connector/
│   │       ├── base.py                # 基础类型 (旧版)
│   │       ├── factory.py             # ★ Connector 工厂：注册和创建
│   │       ├── utils.py               # KV cache 复制/输出聚合
│   │       └── v1/                    # ★★★ V1 Connector 实现（当前主力）
│   │           ├── base.py            # ★ 核心抽象基类 KVConnectorBase_V1
│   │           ├── metrics.py         # 指标统计
│   │           ├── multi_connector.py # 多 connector 组合
│   │           ├── example_connector.py         # 示例 connector（本地文件存储）
│   │           ├── example_hidden_states_connector.py
│   │           ├── offloading_connector.py       # KV Offload 到 CPU
│   │           ├── simple_cpu_offload_connector.py
│   │           ├── lmcache_connector.py          # LMCache 进程内集成
│   │           ├── lmcache_mp_connector.py       # LMCache 多进程集成
│   │           ├── flexkv_connector.py           # FlexKV 分布式 KV 存储
│   │           ├── decode_bench_connector.py     # Decode 基准测试 connector
│   │           ├── ssm_conv_transfer_utils.py    # SSM/KV 转换传输工具
│   │           ├── nixl/                  # NixlConnector（RDMA/P2P 传输）
│   │           ├── mooncake/              # MooncakeConnector
│   │           ├── moriio/                # MoRIIOConnector (ROCm only)
│   │           ├── hf3fs/                 # HF3FSKVConnector (3FS 文件系统)
│   │           ├── lmcache_integration/   # LMCache 集成适配
│   │           └── offloading/            # KV Offload 子模块
│   │
│   └── ec_transfer/                   # Encoder Cache 传输 (EPD)
│       ├── ec_transfer_state.py       # EC 传输全局状态
│       └── ec_connector/
│           ├── base.py                # EC Connector 基类
│           ├── factory.py             # EC Connector 工厂
│           └── example_connector.py
│
├── v1/
│   ├── engine/                        # 引擎层：PD 分离启动入口
│   ├── core/sched/
│   │   ├── scheduler.py               # ★ 调度器：PD 分离核心调度逻辑
│   │   └── output.py                  # SchedulerOutput（含 connector metadata）
│   └── ...
│
├── entrypoints/                       # CLI 入口，传递 kv_transfer_config
│
├── docs/features/
│   ├── disagg_prefill.md              # ★ PD 分离功能文档
│   ├── disagg_encoder.md              # EPD 编码器分离文档
│   ├── nixl_connector_usage.md
│   └── mooncake_connector_usage.md
│
└── examples/disaggregated/            # 示例脚本
    ├── example_connector/
    ├── disaggregated_encoder/
    ├── mooncake_connector/
    ├── flexkv_connector/
    └── lmcache/
```

---

## 三、三层抽象架构

官方文档将 KV 传输设计为三层抽象：

### 3.1 KV Pipe（可选层）
FIFO 管道，提供 `send_tensor` / `recv_tensor` API。适合简单的 P2P 传输场景。

### 3.2 KV Lookup Buffer（可选层）
键值查找缓冲区，提供 `insert` 和 `drop_select`（类似 SQL 语义）API。  
**存在原因**：prefill worker 可能按 A→B→C 顺序处理，但 decode worker 可能先处理 C，FIFO pipe 无法满足这种乱序需求。

### 3.3 KV Connector（核心层）
连接 KV Pipe/LookupBuffer 与 vLLM 引擎，提供：
- `send_kv_caches_and_hidden_states` — 发送 KV cache
- `recv_kv_caches_and_hidden_states` — 接收 KV cache

开发时可选择在哪一层实现：
- **完全自定义 Connector**：跳过 Pipe 和 Buffer，直接实现 Connector，最大灵活性
- **数据库式 Connector**：实现自己的 LookupBuffer
- **P2P Connector**：实现自己的 Pipe

---

## 四、核心源码深度剖析

### 4.1 配置入口：`KVTransferConfig`

**文件**：`vllm/config/kv_transfer.py`

```python
@config
class KVTransferConfig:
    kv_connector: str | None = None        # Connector 名称，如 "NixlConnector"
    engine_id: str | None = None            # 引擎唯一 ID（自动生成 UUID）
    kv_buffer_device: str = ...             # buffer 设备: 'cuda', 'cpu', 'xpu'
    kv_buffer_size: float = 1e9             # buffer 大小（bytes）
    kv_role: KVRole | None = None           # ★ 'kv_producer' | 'kv_consumer' | 'kv_both'
    kv_rank: int | None = None              # 0=prefill, 1=decode
    kv_parallel_size: int = 1               # 并行实例数
    kv_ip: str = "127.0.0.1"
    kv_port: int = 14579
    kv_connector_extra_config: dict         # 额外配置（透传给 connector）
    kv_connector_module_path: str | None    # 外部 connector 模块路径
    kv_load_failure_policy: Literal["recompute", "fail"] = "fail"
```

**关键属性/判断方法**：

| 属性 | 说明 |
|------|------|
| `is_kv_transfer_instance` | `kv_connector` 非空 且 `kv_role` 有效 |
| `is_kv_producer` | 角色为 `kv_producer` 或 `kv_both`（prefill 端） |
| `is_kv_consumer` | 角色为 `kv_consumer` 或 `kv_both`（decode 端） |

启动时通过命令行传入，例如：
```bash
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
```

---

### 4.2 全局状态管理：`kv_transfer_state.py`

**文件**：`vllm/distributed/kv_transfer/kv_transfer_state.py`

全局单例 `_KV_CONNECTOR_AGENT`，核心函数：

| 函数 | 作用 |
|------|------|
| `get_kv_transfer_group()` | 获取全局 connector 实例 |
| `has_kv_transfer_group()` | 检查是否已初始化 |
| `ensure_kv_transfer_initialized()` | ★ 初始化入口：`KVConnectorFactory.create_connector()` |
| `ensure_kv_transfer_shutdown()` | 关闭 connector |
| `_sync_engine_id_across_tp()` | 跨 TP rank 同步 engine_id |

初始化流程：
```
vllm_config.kv_transfer_config.is_kv_transfer_instance
  → _sync_engine_id_across_tp()   # 同步 engine_id
  → KVConnectorFactory.create_connector(role=WORKER)  # 创建 worker connector
```

---

### 4.3 Connector 工厂：`factory.py`

**文件**：`vllm/distributed/kv_transfer/kv_connector/factory.py`

**设计要点**：
- 使用 **懒加载注册器**（`_registry`），避免加载未使用的 connector 依赖
- 支持从**外部模块路径**（`kv_connector_module_path`）动态加载自定义 connector
- Scheduler connector 和 Worker connector **分开创建**，强制职责分离

**已注册的 Connector**：

| Connector 名称 | 说明 |
|---------------|------|
| `ExampleConnector` | 调试用，KV cache 写入本地 safetensors 文件 |
| `ExampleHiddenStatesConnector` | 隐藏状态传输示例 |
| `NixlConnector` | 基于 NIXL 的 RDMA/P2P 传输 |
| `NixlPullConnector` | NIXL Pull 模式 |
| `NixlPushConnector` | NIXL Push 模式 |
| `MooncakeConnector` | 基于 Mooncake 的分布式传输 |
| `MooncakeStoreConnector` | Mooncake Store（存储后端） |
| `MultiConnector` | 组合多个 connector |
| `MoRIIOConnector` | MoRI-IO（ROCm only） |
| `OffloadingConnector` | KV Cache 卸载到 CPU |
| `SimpleCPUOffloadConnector` | 简单 CPU 卸载 |
| `DecodeBenchConnector` | Decode 基准测试 |
| `FlexKVConnectorV1` | FlexKV 分布式 KV 存储 |
| `LMCacheConnectorV1` | LMCache 进程内 |
| `LMCacheMPConnector` | LMCache 多进程 |
| `HF3FSKVConnector` | HF3FS 文件系统 |

---

### 4.4 核心抽象：`KVConnectorBase_V1`

**文件**：`vllm/distributed/kv_transfer/kv_connector/v1/base.py`

这是整个 PD 分离的**核心契约接口**，定义了 Scheduler 侧和 Worker 侧两种角色的职责：

#### 角色定义

```python
class KVConnectorRole(enum.Enum):
    SCHEDULER = 0   # 运行在调度器进程
    WORKER = 1      # 运行在 worker 进程
```

#### Scheduler 侧方法（调度器进程调用）

| 方法 | 调用时机 | 作用 |
|------|---------|------|
| `get_num_new_matched_tokens(request, num_computed_tokens)` | 请求首次调度时 | 查询外部 KV cache 中可加载的 token 数。返回 `None` 则推迟调度 |
| `update_state_after_alloc(request, blocks, num_external_tokens)` | KV cache 块分配成功后 | 通知 connector 决定是否触发 KV 加载 |
| `build_connector_meta(scheduler_output)` | 每个调度步骤末尾 | 构建传递给 worker connector 的 metadata，同时清理内部状态 |
| `update_connector_output(connector_output)` | worker 输出返回后 | 处理 worker 侧的 connector 输出 |
| `request_finished(request, block_ids)` | 请求完成时 | 决定是否需要异步保存 KV cache |
| `take_events()` | 周期性 | 获取 KV cache 事件（用于统计） |
| `has_pending_push_work()` | 引擎主循环 | Push 模式下是否有待处理工作 |
| `on_new_request(request)` | 新请求到达 | 可选 hook，默认空操作 |

#### Worker 侧方法（worker 进程调用，在 forward context 和 attention 层内部执行）

| 方法 | 调用时机 | 作用 |
|------|---------|------|
| `register_kv_caches(kv_caches)` | 初始化 | 注册 KV cache tensor |
| `start_load_kv(forward_context)` | forward 前 | **开始异步加载** KV cache |
| `wait_for_layer_load(layer_name)` | attention 层内 | **等待特定层的 KV** 加载完成 |
| `save_kv_layer(layer_name, kv_layer, attn_metadata)` | attention 层内 | **保存当前层 KV cache** |
| `wait_for_save()` | forward 后 | **等待所有异步保存**完成 |
| `get_finished(finished_req_ids)` | 每个步骤 | 返回异步传输完成的请求 ID |
| `build_connector_worker_meta()` | 每个步骤 | 构建回传调度器的 metadata |
| `handle_preemptions(metadata)` | 抢占前 | 处理被抢占请求的 KV |
| `get_block_ids_with_load_errors()` | 加载后 | 返回加载失败的 block ID |
| `get_handshake_metadata()` | 连接建立时 | P/D 之间握手元数据 |

#### 元数据通信协议

```
Scheduler Connector ──KVConnectorMetadata──▶ Worker Connector
  (build_connector_meta)                      (bind_connector_metadata)

Worker Connector ──KVConnectorWorkerMetadata──▶ Scheduler Connector
  (build_connector_worker_meta)                 (update_connector_output)
```

---

### 4.5 调度器集成：`scheduler.py`

**文件**：`vllm/v1/core/sched/scheduler.py`

调度器在 `schedule()` 方法中的 PD 分离相关逻辑（按执行顺序）：

```
1. get_num_new_matched_tokens()      — 请求调度时查询外部 KV cache 命中情况
   ↓ 如果返回 None，请求推迟到下轮
   ↓ 返回 ext_tokens > 0，标记需要从外部加载

2. 分配 KV cache 块（本地 prefix cache + 外部 tokens）

3. update_state_after_alloc()        — 通知 connector 块已分配，决定加载计划
   ↓

4. build_connector_meta()            — 构建 metadata，通过 SchedulerOutput
   ↓                                   传递给 worker connector
5. 执行 forward（worker connector 执行 KV load/save）

6. update_connector_output()         — 处理 worker 返回结果
```

**SchedulerOutput 中的关键字段**：
```python
kv_connector_metadata: KVConnectorMetadata | None = None  # 给 worker 的 KV 指令
ec_connector_metadata: ECConnectorMetadata | None = None  # 给 worker 的 EC 指令
```

---

### 4.6 示例 Connector 实现：`ExampleConnector`

**文件**：`vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py`

这是一个最简实现，将 KV cache 以 safetensors 格式存储到本地磁盘，非常适合理解 connector 的工作流程：

```
【Prefill (store) 流程】
1. get_num_new_matched_tokens() → 检查本地是否有缓存的 KV
2. build_connector_meta() → 构建 ExampleConnectorMetadata（含 slot_mapping + is_store=True）
3. save_kv_layer() → 从 GPU paged buffer 提取 KV，写入 .safetensors 文件

【Decode (load) 流程】
1. get_num_new_matched_tokens() → 检查本地是否有缓存的 KV，返回匹配 token 数
2. update_state_after_alloc() → 记录需加载的请求到 _requests_need_load
3. build_connector_meta() → 构建 metadata（is_store=False）
4. start_load_kv() → 从 .safetensors 文件加载 KV，注入 GPU paged buffer
```

---

### 4.7 第三方 Connector 对比

| Connector | 传输方式 | 适用场景 | 核心特点 |
|-----------|---------|---------|---------|
| **NixlConnector** | RDMA/UCX/GDS | 生产级低延迟 | 支持 Push/Pull 模式，完全异步发送/接收，多后端 |
| **MooncakeConnector** | 分布式 KV 存储 | 大规模分布式 | 含 Store 后端，支持 Coordinator 协调 |
| **LMCacheConnectorV1** | 进程内集成 | KV 缓存复用 | 与 LMCache 深度集成 |
| **FlexKVConnectorV1** | 分布式 KV Store | 超大规模推理 | 多级缓存管理 |
| **MoRIIOConnector** | ROCm 专有 | AMD GPU 集群 | ROCm 平台优化 |
| **HF3FSKVConnector** | 3FS 文件系统 | 共享存储 | 基于 3FS 分布式文件系统 |
| **MultiConnector** | 组合多个 | 复杂链路 | 可串联多个 connector |

---

### 4.8 Encoder Disaggregation (EPD)

**目录**：`vllm/distributed/ec_transfer/`

多模态模型的视觉编码器分离，架构与 KV 传输平行：

| KV Transfer (PD分离) | EC Transfer (EPD) |
|---------------------|-------------------|
| `kv_transfer_state.py` | `ec_transfer_state.py` |
| `kv_connector/v1/base.py` | `ec_connector/base.py` |
| `kv_connector/factory.py` | `ec_connector/factory.py` |
| `KVConnectorBase_V1` | EC Connector 基类 |
| `kv_connector_metadata` | `ec_connector_metadata` |

示例脚本：`examples/disaggregated/disaggregated_encoder/`

---

## 五、启动与运行流程

### 5.1 启动两个实例

以 ExampleConnector 为例：

```bash
# Terminal 1 - Prefill 实例 (KV Producer)
vllm serve <model> \
  --kv-transfer-config '{"kv_connector":"ExampleConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"/tmp/kv_cache"}}'

# Terminal 2 - Decode 实例 (KV Consumer)
vllm serve <model> \
  --kv-transfer-config '{"kv_connector":"ExampleConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"shared_storage_path":"/tmp/kv_cache"}}'
```

### 5.2 完整请求生命周期

```
用户请求 → 路由到 Prefill 实例
  ↓
Prefill 实例:
  1. Scheduler 调度请求
  2. Forward pass 执行 prefill → 生成 KV cache
  3. Worker Connector.save_kv_layer() → 逐层保存/发送 KV
  4. request_finished() → 异步传输中
  5. get_finished() → 传输完成确认
  ↓ (KV Cache 通过 Connector 传输)
Decode 实例:
  1. Scheduler 收到请求
  2. get_num_new_matched_tokens() → 检测外部 KV cache 命中
  3. update_state_after_alloc() → 分配块，标记需加载
  4. build_connector_meta() → 传递加载指令给 worker
  5. Worker Connector.start_load_kv() → 开始加载
  6. Attention 层内 wait_for_layer_load() → 逐层等待加载完成
  7. Forward pass 执行 decode → 生成 token
```

---

## 六、阅读建议

### 入门路径（按顺序阅读）

1. **理解概念** → `docs/features/disagg_prefill.md`
2. **配置入口** → `vllm/config/kv_transfer.py`
3. **工厂与注册** → `vllm/distributed/kv_transfer/kv_connector/factory.py`
4. **核心接口** → `vllm/distributed/kv_transfer/kv_connector/v1/base.py`
5. **参考实现** → `vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py`
6. **调度集成** → `vllm/v1/core/sched/scheduler.py`（搜索 `connector` 定位相关代码段）

### 深入路径（按主题）

- **要写自己的 Connector** → 以 `example_connector.py` 为模板，实现 `KVConnectorBase_V1` 的所有抽象方法
- **要理解调度如何决策** → `scheduler.py` 中 `get_num_new_matched_tokens` 和 `update_state_after_alloc` 的调用上下文
- **要理解 RDMA/P2P 传输** → `vllm/distributed/kv_transfer/kv_connector/v1/nixl/`
- **要理解分布式 KV 存储** → `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/`
- **要理解编码器分离** → `vllm/distributed/ec_transfer/` + `docs/features/disagg_encoder.md`

---

## 七、关键设计决策与注意事项

1. **Scheduler 侧和 Worker 侧强制分离**：两类 connector 在不同进程创建，通过 `KVConnectorMetadata` 序列化通信
2. **异步传输支持**：`start_load_kv` / `wait_for_layer_load` 和 `save_kv_layer` / `wait_for_save` 配对设计支持 layer-by-layer 流水线
3. **懒加载注册**：factory 中每个 connector 一个 import loader，按需加载依赖
4. **外部模块支持**：`kv_connector_module_path` 允许动态加载用户自定义 connector
5. **HMA 兼容**：`SupportsHMA` 接口用于 Hybrid Memory Allocator 场景
6. **失败策略**：`kv_load_failure_policy` 支持 `recompute`（重新计算）和 `fail`（直接失败）
