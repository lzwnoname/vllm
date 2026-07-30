# vLLM V1 PD 分离 KV Cache 传输深度调研（NIXL 为主）

> 调研日期：2026-07-29；代码基线：本仓库当前 HEAD（nixl_connector 已拆分为 `nixl/` 包，含 pull/push 双模式）。

## 0. TL;DR

- 传输由 **KVConnector** 负责，分 Scheduler 侧（决策/元数据）与 Worker 侧（执行 RDMA）两个角色实例，定义在 `vllm/distributed/kv_transfer/kv_connector/v1/base.py`。
- 主力实现 **NixlConnector**（= NixlPullConnector）已重构为包：`vllm/distributed/kv_transfer/kv_connector/v1/nixl/`（connector.py / base_scheduler.py / pull_scheduler.py / base_worker.py / pull_worker.py / push_* / metadata.py / tp_mapping.py）。
- 主流模式是 **pull（READ）**：prefill 侧只在初始化时把 KV cache 显存注册给 NIXL，之后不做任何主动发送；decode 侧拿到 `kv_transfer_params`（含 P 侧 block_ids + side-channel 地址）后先做 zmq 握手拿 NIXL agent 元数据，再对 P 显存发起异步 RDMA READ，完成后通知 P 释放 blocks。
- NIXL **不做逐层传输**：`save_kv_layer`/`wait_for_layer_load` 是 no-op（connector.py:277-289）；逐层 save/load 接口是为 LMCache/MoRIIO 等 connector 设计的（`maybe_transfer_kv_layer` 装饰器，`vllm/model_executor/layers/attention/kv_transfer_utils.py`）。
- 调度侧关键状态：`RequestStatus.WAITING_FOR_REMOTE_KVS` + `finished_recving_kv_req_ids`；P 侧 blocks 延迟释放（lease，默认 30s，心跳续租）。

---

## 1. 整体架构

### 1.1 组件与目录

PD 分离时 prefill/decode 是两个独立 vLLM 进程，各自通过 `--kv-transfer-config` 指定 connector。所有实现位于 `vllm/distributed/kv_transfer/`：

```
kv_transfer/
├── kv_transfer_state.py        # get_kv_transfer_group() 全局句柄
├── kv_connector/
│   ├── factory.py              # KVConnectorFactory，注册 14 种 connector（L152-242）
│   ├── base.py                 # 旧版(V0)基类
│   └── v1/
│       ├── base.py             # KVConnectorBase_V1（核心抽象）
│       ├── nixl/               # NIXL（UCX/RDMA，跨机 P2P，生产主力）
│       │   ├── connector.py    # NixlBaseConnector / NixlPullConnector / NixlPushConnector 门面
│       │   ├── base_scheduler.py + pull_scheduler.py + push_scheduler.py
│       │   ├── base_worker.py  (2467 行) + pull_worker.py + push_worker.py
│       │   ├── metadata.py     # NixlAgentMetadata / NixlHandshakePayload / NixlConnectorMetadata
│       │   └── tp_mapping.py   # 异构 TP 映射
│       ├── mooncake/           # Mooncake（kvcache-ai，RDMA 零拷贝 + DRAM/SSD 池）
│       ├── moriio/             # MoRI-IO（仅 ROCm）
│       ├── lmcache_connector.py / lmcache_mp_connector.py / lmcache_integration/
│       ├── offloading_connector.py / offloading/   # GPU→CPU/磁盘 卸载
│       ├── simple_cpu_offload_connector.py / flexkv_connector.py / hf3fs/
│       ├── multi_connector.py  # 组合多个 connector
│       └── example_connector.py / example_hidden_states_connector.py / decode_bench_connector.py
```

每个进程里 connector 有两个实例（factory.py:67-75 注释明确要求分离）：
- **Scheduler connector**（`KVConnectorRole.SCHEDULER`）：与调度器同进程，决定哪些请求要收/发 KV，产出 `KVConnectorMetadata` 随 `SchedulerOutput` 下发。
- **Worker connector**（`KVConnectorRole.WORKER`）：每个 TP worker 一个，持有 NIXL agent，真正发 RDMA。

### 1.2 抽象接口（v1/base.py）

`KVConnectorBase_V1`（L171）分两组方法：

**Worker 侧**（base.py:207-437）：

| 方法 | 行号 | 语义 |
|---|---|---|
| `register_kv_caches(kv_caches)` | 251 | 初始化时把每层 KV tensor 注册给 connector（NIXL 注册显存） |
| `register_cross_layers_kv_cache` | 261 | cross-layer（所有层共享一块大 tensor）布局的注册 |
| `start_load_kv(forward_context)` | 292 | forward 前启动异步 load（D 侧发起 READ 的入口） |
| `wait_for_layer_load(layer_name)` | 310 | 逐层 load 流水线用，attention 层入口等第 i 层 |
| `save_kv_layer(layer_name, kv_layer, attn_metadata)` | 325 | 逐层 save，attention 层出口发起第 i 层保存 |
| `wait_for_save()` | 346 | forward context 退出时等所有异步 save 完成，防止 paged buffer 被覆写 |
| `get_finished(finished_req_ids)` | 357 | 返回 (finished_sending, finished_recving) 两个 req_id 集合 |
| `get_block_ids_with_load_errors()` | 375 | 传输失败的 block 集合（配合 recompute 策略） |
| `get_handshake_metadata()` | 417 | Worker 产出握手元数据（经 executor 汇总回 Scheduler 侧） |
| `set_host_xfer_buffer_ops` | 278 | 注册 h2d/d2h copy op（host buffer 场景） |

**Scheduler 侧**（base.py:439-708）：

| 方法 | 行号 | 语义 |
|---|---|---|
| `get_num_new_matched_tokens(request, num_computed_tokens)` | 453 | 返回 (可从外部加载的 token 数, 是否异步 load) |
| `update_state_after_alloc(request, blocks, num_external_tokens)` | 488 | block 分配后更新 connector 状态 |
| `build_connector_meta(scheduler_output)` | 514 | 生成本 step 的 `KVConnectorMetadata` 下发 worker |
| `on_new_request` / `update_connector_output` | 529/537 | 新请求钩子 / 消化 worker 回传的 finished 集合 |
| `request_finished(request, block_ids)` | 547 | 请求结束时调用；返回 (是否延迟释放 blocks, kv_transfer_params) |
| `set_xfer_handshake_metadata_pp_aware` | 664 | Scheduler 侧接收全部 worker 的握手元数据（按 (pp,tp) rank 索引） |
| `has_pending_push_work` | 577 | push 模式下让 engine loop 保持运转 |

握手元数据的流通路径：Worker `get_handshake_metadata()`（gpu_worker.py:680-699，按 (pp_rank,tp_rank) 打包）→ executor 汇总 → EngineCore `kv_connector.set_xfer_handshake_metadata_pp_aware(content)`（v1/engine/core.py:177-193）→ Scheduler 侧起 zmq listener 对外服务。

---

## 2. NIXL connector 详解

> 门面：`nixl/connector.py`。`NixlConnector = NixlPullConnector`（L387，向后兼容别名）。pull=READ（D 拉），push=WRITE（P 推）。以下以 pull 为主。

### 2.1 Prefill 侧：KV 注册与"保存"

**注册（一次性，初始化时）**：`GPUModelRunner.initialize_kv_cache()` 末尾（gpu_model_runner.py:7610-7619）调用 `kv_transfer_group.register_kv_caches(kv_caches)` → `NixlBaseConnectorWorker.register_kv_caches`（base_worker.py:1024）：

- 每层 KV cache（或 HMA 下被多 group 共享的 tensor）按 `TransferTopology.get_transfer_cache_regions` 拆 region（非 MLA 每层 K/V 各一个 region，MLA 每层 1 个），拼 `caches_data = [(base_addr, size_bytes, device_id, "")]`（L1195-1197）；
- `self.nixl_wrapper.get_reg_descs(...)` + `register_memory(descs, backends=["UCX"])`（L1223-1225）把整片 GPU KV 池注册给 NIXL/RDMA 网卡；
- 计算兼容性哈希 `compute_nixl_compatibility_hash`（metadata.py:81，含 vllm 版本/模型/dtype/KV heads/层数/attn backend/cache dtype/HMA）；
- 生成 `NixlAgentMetadata`（engine_id、agent_metadata、kv_caches_base_addr、num_blocks、block_lens、kv_cache_layout、block_size…）包进 `NixlHandshakePayload` 存为 `self.xfer_handshake_metadata`（base_worker.py:1000-1022），供 D 侧握手拉取。

**关键：P 侧 forward 时不做任何逐层保存**。`connector.py:277-289`：

```python
def wait_for_layer_load(self, layer_name):  # L277
    """NixlConnector does not do layerwise saving."""
    pass

def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):  # L281
    """NixlConnector does not save explicitly."""
    pass
```

KV 本来就写在已注册的显存里，D 侧直接远程 READ，无需 P 配合拷贝。`wait_for_save`（connector.py:291-295）只在 `use_host_buffer`（kv_buffer_device=cpu，NIXL 不支持直注的加速器）时把 KV 拷到 host buffer：`self.connector_worker.save_kv_to_host(...)`（base_worker.py:1848）。

**请求结束 → 延迟释放 + 导出 params**：P 侧请求 finish（max_tokens=1 的 prefill 请求立即结束）时 `Scheduler._free_request` → `_connector_finished`（scheduler.py:2468-2503）→ `NixlPullConnectorScheduler.request_finished`（pull_scheduler.py:181-280）：

```python
# pull_scheduler.py:239-280（核心）
delay_free_blocks = any(len(group) > 0 for group in block_ids)
if delay_free_blocks:
    request_kv_blocks_ttl = self._kv_lease_duration          # 默认 30s
    self._reqs_need_send[request.request_id] = (
        time.perf_counter() + request_kv_blocks_ttl)          # lease 到期时间
    block_ids = self.get_sw_clipped_blocks(block_ids)         # SWA 裁剪
    remote_num_tokens = request.num_computed_tokens
return delay_free_blocks, dict(
    do_remote_prefill=is_p_node, do_remote_decode=is_d_node,
    remote_block_ids=block_ids,              # P 侧物理 block 号！
    remote_engine_id=self.engine_id,
    remote_request_id=request.request_id,
    remote_host=self.side_channel_host,      # zmq 握手地址
    remote_port=self.side_channel_port,
    tp_size=..., remote_num_tokens=remote_num_tokens,
    remote_blocks_expiry_time=blocks_expiry_time)
```

- 返回 `True` → scheduler 不立即释放这些 blocks，等 `get_finished` 回报 finished_sending 才释放（scheduler.py:2626-2629 `self._free_blocks(...)`）。
- `kv_transfer_params` 随 `EngineCoreOutput`（scheduler.py:1836）→ 响应里 `kv_transfer_params` 字段返回给 proxy。

### 2.2 Decode 侧：发现、握手、READ

**调度入口**：D 侧请求带着 `kv_transfer_params{do_remote_prefill:True, remote_block_ids, remote_engine_id, remote_request_id, remote_host, remote_port, tp_size}` 到达（经 OpenAI 协议 `kv_transfer_params` 字段 → `sampling_params.extra_args` → `Request.kv_transfer_params`，v1/request.py:102-118）。

`NixlPullConnectorScheduler.get_num_new_matched_tokens`（pull_scheduler.py:34-110）：`do_remote_prefill` 时返回 `(prompt_token_count - num_computed_tokens, True)` —— 全部 prompt token 都算作"外部已算好"，且 **异步加载**（`load_kv_async=True`）。

`update_state_after_alloc`（pull_scheduler.py:112-179）：把请求连同本地分配的（未 hash 的）block_ids 放入 `self._reqs_need_recv`，并清掉 `do_remote_prefill` 标志保证只传一次。

调度器据此把请求置为 `WAITING_FOR_REMOTE_KVS`（scheduler.py:972-1002），block 已分配但 `delay_cache_blocks=True` 暂不写 prefix cache。

**Worker 发起 READ**：每个 step，model runner 经 `_get_kv_connector_output`（kv_connector_model_runner_mixin.py:78-112）`bind_connector_metadata` → `start_load_kv` → `NixlPullConnectorWorker.start_load_kv`（pull_worker.py:44-103）：

1. 本地 logical block_ids → kernel（物理）block_ids（`_logical_to_kernel_block_ids`）；
2. 若该 remote engine 未握手 → `_background_nixl_handshake`（base_worker.py:898）：单线程 ThreadPoolExecutor 里跑 `_nixl_handshake`（base_worker.py:565+）—— 向 `remote_host:remote_port` 的 zmq REQ socket 发 `(GET_META_MSG, pp_rank, tp_rank)`，拿回 `NixlHandshakePayload`，校验 compatibility hash 后解出 `NixlAgentMetadata`，`add_remote_agent`（base_worker.py:1474）注册远端 agent 并按 TP 比例预建 xfer descriptor 句柄（异构 TP 时按 kv_head 维度切分）；
3. 握手完成 → `_read_blocks_for_req` → `_read_blocks`（pull_worker.py:215-353）：

```python
# pull_worker.py:311-342（核心）
remote_block_descs_ids = self._compute_desc_ids(block_ids=remote_block_ids, ...)
local_block_descs_ids  = self._compute_desc_ids(block_ids=local_block_ids, ...)
handle = self.nixl_wrapper.make_prepped_xfer(
    "READ", local_xfer_side_handle, local_block_descs_ids,
    remote_xfer_side_handle, remote_block_descs_ids,
    notif_msg=notif_id)                # notif_id = f"{remote_request_id}:{world_size}"
self.nixl_wrapper.transfer(handle)     # 异步 RDMA READ，立即返回
self._recving_transfers[request_id].append(handle)
```

**Block 布局对齐**：传的是 **block 级** 数据（descriptor = region × block_id）。P、D 各自用自己的 block_id 空间：`remote_block_ids` 由 P 通过 kv_transfer_params 显式带给 D，D 建立 (remote_block_id, local_block_id) 的一一映射，RDMA READ 直接把 P 侧第 i 个 block 写进 D 侧第 i 个本地 block。block_size 不同也支持（`block_size_ratio` + `get_mapped_blocks` 映射裁剪，pull_worker.py:237-261）；逻辑 block 与 kernel block 不一致时用 `physical_blocks_per_logical_kv_block` 比例换算；异构 TP 时按 kv_head 切分或 MLA 整读（`add_remote_agent` docstring base_worker.py:1491-1521 有完整 ASCII 图）；HND/NHD layout 不一致可 `enable_permute_local_kv` 收后重排。

**完成检测**：`get_finished()`（base_worker.py:1948-2046）= `_get_new_notifs()`（P 侧收到 D 发来的 notif，pull_worker.py:355-408，异构 TP 要等齐 N 个 consumer）+ `_pop_done_transfers`（D 侧轮询 handle 状态 `check_xfer_state == "DONE"`，base_worker.py:2092-2121）。结果经 `KVConnectorOutput.finished_sending/finished_recving` 回 scheduler。

**P 侧 blocks 释放**：D READ 完成后 NIXL 自动把 `notif_id` 推给 P worker → P 在 `_get_new_notifs` 里计数收齐 → 该 req 进入 `done_sending` → scheduler 释放 blocks。若 D 迟迟不读，lease（默认 `kv_lease_duration=30s`）到期在 `get_finished` 里超期释放（base_worker.py:2027-2044）；D 排队期间周期性发 `HB:req1,req2` 心跳续租（`_send_heartbeats` base_worker.py:2157；P 侧 `_handle_heartbeat` L2071 延长 lease）。

### 2.3 传输发起方与握手通道

- **数据面**：pull 模式由 **D 发起 RDMA READ**（P 完全被动）；push 模式由 **P 发起 WRITE**（专用 `nixl-push-writer` 后台线程，push_worker.py:5-25,196-241；D 先通过 NIXL notif `PUSH_REG:` 前缀把自己 agent 注册推给 P）。
- **控制面（握手/元数据）**：zmq。每个 vLLM 实例的 scheduler 进程起 ROUTER listener（base_scheduler.py:291-332 `_nixl_handshake_listener`，端口 `VLLM_NIXL_SIDE_CHANNEL_PORT`(默认5600)+dp_rank），worker 侧用 REQ 去 `remote_host:remote_port` 拉对方全部 (pp,tp) rank 的 `NixlAgentMetadata`。握手响应里附带 perf_counter 时间戳用于跨机时钟对齐（lease  expiry 换算）。
- **完成通知**：NIXL 自带 notif 机制（`send_notif`/`get_new_notifs`），消息即 `req_id:tp_size`，心跳为 `HB:` 前缀。

---

## 3. 调度器侧的 PD 感知（vllm/v1/core/sched/scheduler.py）

核心状态：`self.finished_recving_kv_req_ids: set[str]`（L202）、`RequestStatus.WAITING_FOR_REMOTE_KVS`。

调度循环（schedule()）对新请求（L711-1002）：

1. `get_computed_blocks` 查本地 prefix cache（L750-758）；
2. `connector.get_num_new_matched_tokens(request, num_new_local_computed_tokens)`（L762-766）拿到 `(ext_tokens, load_kv_async)`；
3. `allocate_slots(..., num_external_computed_tokens=ext, delay_cache_blocks=load_kv_async)`（L928-940）为外部 token 分配本地 blocks；
4. `connector.update_state_after_alloc(request, blocks, ext)`（L956-960）→ NIXL 记入 `_reqs_need_recv`；
5. 若 `load_kv_async`：请求置 `WAITING_FOR_REMOTE_KVS`，塞回 skipped 队列，本轮不参与 forward（L972-1002）；`num_computed_tokens` 先乐观置位，失败时由 `_update_requests_with_invalid_blocks` 回退重算（`kv_load_failure_policy=recompute`）或直接 fail。

请求"解封"路径：worker 每步 `get_finished` 回报 → `update_from_output` → `_update_from_kv_xfer_finished`（scheduler.py:2602-2629）：

```python
# scheduler.py:2617-2629
for req_id in kv_connector_output.finished_recving or ():
    req = self.requests[req_id]
    if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
        self.finished_recving_kv_req_ids.add(req_id)   # 下个 step 可被调度
    else:
        self._free_blocks(self.requests[req_id])       # 已结束的请求直接释放
for req_id in kv_connector_output.finished_sending or ():
    self._free_blocks(self.requests[req_id])           # P 侧：远端已读完，释放
```

下一轮 schedule 时 `_try_promote_blocked_waiting_request`（L2569-2584）发现 req 在 `finished_recving_kv_req_ids` → `_update_waiting_for_remote_kv`（真正 cache_blocks 入 prefix cache；全命中时回退 1 个 token 以采样）→ 状态回到 WAITING 参与本轮调度。P 侧 `finished_sending` 一到即释放延迟的 blocks。

`build_connector_meta`（base_scheduler.py:402-437）把 `_reqs_need_recv`（新收）、`_reqs_need_send`（待发+到期时间）、`_reqs_in_batch`、心跳包一并打成 `NixlConnectorMetadata` 随 SchedulerOutput 下发 worker。

---

## 4. 请求路由与 kv_transfer_params 时序

`kv_transfer_params` 的完整旅程（以 NIXL + toy proxy 为例，`tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`）：

1. **proxy → P**：请求里塞 `kv_transfer_params={"do_remote_decode": True, "do_remote_prefill": False, remote_*: None}`（toy_proxy_server.py:162-168）——表示"你只做 prefill，产出 KV 给别人"。
2. **P 引擎内**：`Request.kv_transfer_params` 被 NixlScheduler 识别（`is_p_node = params.get("do_remote_decode")`）；请求以 max_tokens=1 跑完 prefill 即 finish；`request_finished` 返回延迟释放 + 填充好的 params（真实 remote_block_ids/engine_id/host/port，见 §2.1）。
3. **P → proxy**：响应 JSON 带 `kv_transfer_params`（toy_proxy_server.py:235-237 提取）。
4. **proxy → D**：原请求 + P 的 params 发给 decode 实例。D 侧 `do_remote_prefill=True` 走 §2.2/§3 流程：异步 READ → WAITING_FOR_REMOTE_KVS → 收齐 → 正式 decode。
5. token↔block 映射：不靠 token 序号对齐，而是 P 直接导出**自己的物理 block_id 列表**（经 SWA 裁剪、按 group 组织），D 按序一一映射到本地新分配 block。`remote_num_tokens` 记录 token 数供调度记账。

OpenAI 服务层细节：`kv_transfer_params` 是请求/响应协议的一等字段（vllm/entrypoints/openai/{completion,chat_completion}/protocol.py），请求侧经 `extra_args["kv_transfer_params"]` 传入引擎；响应侧由 `final_res.kv_transfer_params` 透出。D 侧若请求在进入引擎前被拒，`_with_kv_transfer_rejection_cleanup`（entrypoints/generate/base/serving.py:218-250）会通知 P 释放 pinned blocks。

---

## 5. 部署文档与示例

- `docs/features/disagg_prefill.md`：动机（分开调 TTFT/ITL、控尾延迟；明确"不提升吞吐"）、9 种 connector 清单与启动命令、架构图（`docs/assets/features/disagg_prefill/{abstraction.jpg,overview.jpg,high_level_design.png,workflow.png}`，其中 workflow.png 即逐层 save/load 示意）。
- `docs/features/nixl_connector_usage.md`：NIXL 安装、UCX/LIBFABRIC backend 配置、1P1D 同机/多机完整命令、`VLLM_NIXL_SIDE_CHANNEL_HOST/PORT`、`kv_lease_duration`、`bidirectional_kv_xfer` 多轮 D→P 回拉（含 mermaid 时序图）、`kv_load_failure_policy`、异构 layout、Prometheus 指标。
- 示例：`examples/disaggregated/disaggregated_serving/`（`disagg_proxy_demo.py` XpYd 轮询 proxy、`disagg_proxy_multiturn.py` 有状态多轮 proxy、`disagg_proxy_pushconnector_demo.py`）、`examples/disaggregated/example_connector/`（最小教学实现）、`examples/disaggregated/lmcache/`、`mooncake_connector/`、`flexkv_connector/`；集成测试 `tests/v1/kv_connector/nixl_integration/`（`toy_proxy_server.py`、`run_accuracy_test.sh`）。
- 典型部署：P 实例 `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'`，D 实例 `kv_role":"kv_consumer"`，前置 proxy 按 round-robin 分流并搬运 kv_transfer_params。`kv_both` 对 NixlConnector 已 deprecated（connector.py:122-128）。

---

## 6. Connector 全景对比（factory.py:152-242 注册表）

| Connector | 底层 | 适用场景 |
|---|---|---|
| **NixlConnector / NixlPullConnector** | NIXL(UCX/LIBFABRIC/GDS)，RDMA READ | 生产主力；同机 NVLink/跨机 IB；支持异构 TP/PP、MLA、Mamba、HMA、双向回拉 |
| **NixlPushConnector** | NIXL WRITE | P 主动推（P 侧 PP>1、D 侧无轮询压力等场景；D 不支持 PP>1） |
| **MooncakeConnector / MooncakeStoreConnector** | mooncake-transfer-engine（GPUDirect RDMA 零拷贝）+ DRAM/SSD 缓存池 | 跨机 PD + KV 池化/慢对象存储环境 |
| **LMCacheConnectorV1 / LMCacheMPConnector** | LMCache（底层可走 NIXL）；MP 模式独立 lmcache server | 需要跨请求 KV 复用/多级缓存；逐层流水 |
| **MoRIIOConnector** | MoRI-IO | 仅 ROCm |
| **OffloadingConnector / SimpleCPUOffloadConnector / FlexKVConnectorV1 / HF3FSKVConnector** | host 内存/文件系统/分布式 KV store | 非 PD 直连，而是 KV 卸载扩容（prefix cache 外溢） |
| **MultiConnector** | 组合 | 同时启用多个（如 Nixl + Offloading） |
| **ExampleConnector / ExampleHiddenStatesConnector / DecodeBenchConnector** | 共享存储 / naive | 教学、hidden states 传输、decode 压测 |

---

## 7. 延迟优化与 layerwise

**NIXL 的 overlap 策略不是逐层，而是"整请求异步 + 跨 step 流水"**：

- P 侧零拷贝：KV 写在已注册显存即"已发送"，P 的 TTFT 不被传输阻塞；blocks 靠 lease  pinning。
- D 侧 READ 在 `start_load_kv`（forward 前、metadata bind 后）发起，与**其它请求的 decode 计算 overlap**——请求在 WAITING_FOR_REMOTE_KVS 期间不占 batch，若干 step 后 `finished_recving` 到达再进入调度；握手也在后台线程（`_handshake_initiation_executor`）。
- 代价是首 token 要等整段 KV 到齐，而不是逐层就绪即算。

**逐层（layerwise）接口是框架级能力**（base.py:310-355；`maybe_transfer_kv_layer` 装饰器在 attention forward 入口 `wait_for_layer_load`、出口 `save_kv_layer`，kv_transfer_utils.py:38-59），LMCache/MoRIIO 用它做到"传完第 i 层就算第 i 层"。因无法被 CUDA graph 捕获，这类 connector 会强制 PIECEWISE cudagraph（base.py:608-628 `requires_piecewise_for_cudagraph`）。NIXL 显式不用它（connector.py:278/288 注释）。NIXL 侧的替代优化：HND layout（`get_required_kvcache_layout` connector.py:140-157）、`enable_cross_layers_blocks`（单 region 全层连续，descriptor 数骤降，`prefer_cross_layer_blocks` connector.py:82-110）、异构 TP 下按 kv_head 切片并行 READ。

**kv_role 的影响**（config/kv_transfer.py:41-43, 108-118）：`kv_producer`（P）/ `kv_consumer`（D）/ `kv_both`。角色写入 params 的 `do_remote_prefill/do_remote_decode` 解释方式，并决定 push 模式下能力边界（如 consumer 禁 PP>1，base_worker.py:436-440）；`is_kv_transfer_instance` 决定是否初始化 connector。NIXL 推荐显式 producer/consumer。

---

## 8. 端到端时序（pull 模式）

```
Proxy            P Scheduler      P Worker (NIXL)        D Scheduler      D Worker (NIXL)
 |  ① req+do_remote_decode            |                       |                |
 |---------------->|                 |                       |                |
 |                 | ② schedule,     |                       |                |
 |                 |   forward(全量)  |                       |                |
 |                 |---------------->| ③ KV 写入已注册显存     |                |
 |                 |   (无逐层save)   |                       |                |
 |                 | ④ finish:       |                       |                |
 |                 | request_finished|                       |                |
 |                 | →延迟释放+params |                       |                |
 |  ⑤ resp{kv_transfer_params:       |                       |                |
 |     remote_block_ids,engine_id,   |                       |                |
 |     host,port,remote_req_id}      |                       |                |
 |<----------------|                 |                       |                |
 |  ⑥ req+params                     |                       |                |
 |---------------------------------------------------------->|                |
 |                 |                 |                       | ⑦ get_num_new_ |
 |                 |                 |                       |   matched_tokens|
 |                 |                 |                       |   =(N,async)    |
 |                 |                 |                       | ⑧ alloc+update_ |
 |                 |                 |                       |   state_after_  |
 |                 |                 |                       |   alloc → WAITING|
 |                 |                 |                       |   _FOR_REMOTE_  |
 |                 |                 |                       |   KVS           |
 |                 |                 |  ⑨ zmq GET_META 握手   |                |
 |                 |                 |<========================================|
 |                 |                 |  (compat hash 校验,    | ⑩ add_remote_  |
 |                 |                 |   NixlAgentMetadata)   |    agent       |
 |                 |                 |                       | ⑪ start_load_kv|
 |                 |                 |                       |   make_prepped_|
 |                 |                 |   ⑫ RDMA READ          |   xfer(READ)   |
 |                 |                 |<========================================|
 |                 |                 |  ⑬ notif(req_id:tp)    | (后台进行,D继续 |
 |                 |                 |    D READ 完成         |  decode其它req)|
 |                 | ⑭ get_finished→ |                       | ⑭ get_finished→|
 |                 |  finished_sending|                      | finished_recving|
 |                 | ⑮ 释放 blocks    |                       | ⑮ req 解除阻塞  |
 |                 |   (或 lease 超时/|                       |   →正式 decode  |
 |                 |    HB 心跳续租)  |                       |   首 token 产出 |
 |  ⑯ stream decode 输出             |                       |                |
 |<----------------------------------------------------------|                |
```

## 9. 关键设计决策

1. **为什么 pull（READ）为主**：P 侧 prefill 是延迟热点，零拷贝"写完即可发"不占用 P 任何传输线程/拷贝；D 天然异步（请求排队期间就能传），传输与 decode overlap。P 只需被动等 notif 释放 blocks。
2. **为什么 block 级而非 token 级**：KV pool 按 block 组织，descriptor=region×block 的 scatter-gather 列表一次 RDMA 搞定；block_id 空间各自独立，靠 params 显式映射，彻底解耦两侧分配器。
3. **为什么异步 + lease**：D 侧握手/READ 都是后台的，调度器用 WAITING_FOR_REMOTE_KVS + finished_recving 做状态机；P 侧 blocks 延迟释放（lease 30s + HB 心跳续租 + 时钟对齐）容忍 D 排队与跨机时钟漂移，防止 D 读到被覆写的 blocks。
4. **为什么握手走 zmq 旁路**：NIXL agent metadata（含 base addr、desc 布局）只需每对 engine 交换一次，zmq REQ/ROUTER 足够；且与 engine_id 绑定、带 compatibility hash，可安全做多实例 XpYd。
5. **为什么不逐层（NIXL）**：整请求一次 READ 已能与其它请求的 decode overlap，逐层收益小而要付出 PIECEWISE graph 代价；逐层接口留给需要"传算流水"的 LMCache 类 connector。
