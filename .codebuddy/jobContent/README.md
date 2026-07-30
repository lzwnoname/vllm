# 工作内容归档索引

## 文档列表

| 日期 | 文档 | 主题 | 关联资料 |
|------|------|------|---------|
| 2026-07-29 | [./2026-07-29/vllm-pd-kv-transfer-nixl-deep-dive.md](./2026-07-29/vllm-pd-kv-transfer-nixl-deep-dive.md) | vLLM V1 PD 分离 KV 传输深度调研（NixlConnector pull/push、调度状态机、时序图） | vllm 源码 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/`、`vllm/v1/core/sched/scheduler.py` |
| 2026-07-28 | [./2026-07-28/kuaishou-sglang-mock-interview.md](./2026-07-28/kuaishou-sglang-mock-interview.md) | 快手电商推理框架实习（SGLang二开）模拟面试题与参考答案 | SGLang 源码 `python/sglang/srt/`；vLLM 源码 |
| 2026-07-05 | [./2026-07-05/vllm-pd-separation-source-guide.md](./2026-07-05/vllm-pd-separation-source-guide.md) | vLLM PD分离（Disaggregated Prefill/Decode）源码导读 | vllm 源码 `vllm/distributed/kv_transfer/` |
| 2026-07-05 | [./2026-07-05/vllm-parallel-strategies-analysis.md](./2026-07-05/vllm-parallel-strategies-analysis.md) | vLLM 分布式并行策略源码分析（TP/PP/DP/EP/SP/CP/EPLB） | vllm 源码 `vllm/config/parallel.py`, `vllm/distributed/parallel_state.py` |
| 2026-07-06 | [./2026-07-06/vllm-async-scheduling-analysis.md](./2026-07-06/vllm-async-scheduling-analysis.md) | vLLM 异步调度（Async Scheduling）源码分析 | vllm 源码 `vllm/v1/core/sched/async_scheduler.py` |
| 2026-07-06 | [./2026-07-06/vllm-moe-communication-analysis.md](./2026-07-06/vllm-moe-communication-analysis.md) | vLLM MoE 并行通信策略分析 | vllm 源码 `vllm/model_executor/layers/fused_moe/` |
| 2026-07-06 | [./2026-07-06/vllm-chunked-prefill-analysis.md](./2026-07-06/vllm-chunked-prefill-analysis.md) | vLLM Chunked Prefill 实现分析 | vllm 源码 `vllm/v1/core/sched/scheduler.py` |

## 维护记录

| 日期 | 操作 | 说明 |
|------|------|------|
| 2026-07-29 | 新增 | 新增 PD 分离 KV 传输 NIXL 深度调研报告 |
| 2026-07-28 | 新增 | 新增快手电商 SGLang 实习模拟面试题（含参考答案与源码冲刺清单） |
| 2026-07-06 | 新增 | 新增 MoE 并行通信策略分析 |
| 2026-07-06 | 新增 | 新增异步调度源码分析 |
| 2026-07-05 | 新增 | 新增 vLLM 分布式并行策略源码分析 |
| 2026-07-05 | 新建 | 初始化文档索引；新增 PD 分离源码导读 |
