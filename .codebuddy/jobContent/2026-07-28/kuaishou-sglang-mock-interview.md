# 快手电商 · 推理框架实习生（SGLang 二次开发）模拟面试题与参考答案

> 基于简历（刘志炜）定制。岗位方向：基于 SGLang 做二次开发，服务电商在线推理场景。
> 结构：自我介绍 → 简历项目深挖 → SGLang 架构 → 推理框架核心 → CUDA/Triton 算子 → 分布式/性能 → 量化 → 手撕代码 → 行为面 → 反问。

---

## 0. 面试节奏预判与自我介绍模板

**预判流程（约 60min）**：自我介绍 3min → 简历项目深挖 25-30min（权重最高）→ 框架/八股开放题 15min → 手撕代码 10-15min → 行为+反问 5min。

**自我介绍模板（90s，背熟后改成自己的话）**：

> 面试官好，我是刘志炜，浙大计算机本科、中山大学智能工程学院研一，NOIP 提高组一等奖出身，算法和工程基础比较扎实。
> 之前在腾讯微信做扩展开发实习，负责广告组件框架，做过首帧加载优化，把首帧耗时从 817ms 压到 44ms，这段经历让我建立了"先 profiling 再优化、用数据验证"的性能工程方法论。
> 个人方向上我聚焦 LLM 推理系统：一是基于 nano-vllm 做了 KV Cache 管理和请求调度的扩展，包括滑动窗口/周期压缩驱逐、chunked-prefill 混合组批、Gated Delta Attention 模型接入、AWQ W4A16 量化接入；二是我有一个 Triton/CUDA 算子库项目，手写过 FlashAttention V2、AWQ Dequant GEMM、Tiled GEMM 这些推理核心算子。
> 我对 vLLM 的调度器和 KV cache 管理读过源码、动手改过，SGLang 也在读。希望能把这些经验用到快手电商的推理框架优化上。

---

## Part 1 简历深挖：KVC-nano-vllm（必考区）

### Q1. 整体介绍一下这个项目，架构上你动了哪几层？

**考察点**：是否真正理解推理框架全貌，而非只改了局部。

**参考答案**：
nano-vllm 是教学级框架，结构是 `LLMEngine → Scheduler → ModelRunner → KV Cache Manager`。我在三层做了扩展：
1. **KV Cache 管理层**：新增两种驱逐策略（滑动窗口、周期性压缩），核心是 block 粒度的释放/回收逻辑和 refcnt 维护；
2. **调度层**：实现 chunked-prefill + prefill/decode 混合组批，改造 `schedule()` 的 token 预算分配；
3. **模型层**：接入 Qwen3.5 Text Dense（含 Gated Delta Attention 的 state 管理）和 AWQ W4A16 量化权重加载。
最大难点是驱逐策略与 prefix caching / RoPE 的正确性交互，以及 hybrid 模型（full attention + GDN）两类缓存的统一管理。

**追问**：代码量多少？测试怎么做的？（准备：单元测试对比驱逐前后 logits、长上下文 needle-in-haystack 测试、与 HF 输出逐 token 对齐）

---

### Q2. 滑动窗口驱逐 vs 周期性压缩驱逐，分别怎么实现的？为什么吞吐能提升 10%？

**考察点**：对 KV cache 驱逐经典方法（StreamingLLM / H2O / SnapKV）的理解。

**参考答案**：
- **滑动窗口**：只保留最近 W 个 token 的 KV + 最前几个 attention sink token（StreamingLLM 发现初始 token 吸引大量注意力分数，直接丢弃会导致 softmax 分布崩溃）。实现上是维护每个请求的有效 block 区间，窗口滑动时把滑出的 block refcnt 减一放回 free list。
- **周期性压缩**：每 1024 个 decode step 触发一次，对超长上下文按 block 粒度做重要性筛选——用近期窗口内 token 对历史 KV 的累积注意力分数做 top-k（H2O/SnapKV 思路），保留重要 block，其余驱逐。
- **吞吐提升来源**：4K+ 长 prompt 场景下 KV cache 是并发瓶颈，驱逐后单请求 KV 占用下降 → 同时驻留的请求数变多 → batch 变大、GPU 利用率上升，吞吐提升约 10%。注意这是用"近似"换吞吐，需要在精度可接受的场景使用。

**追问**：驱逐和 prefix caching 冲突吗？（冲突：被驱逐的 block 可能正被缓存复用，实现上要么禁止驱逐 shared/refcnt>1 的 block，要么同步失效 hash 表项）

---

### Q3. 驱逐 KV 之后 RoPE 位置编码怎么处理？有没有正确性问题？

**考察点**：位置编码与 KV cache 驱逐的交互，最容易被深挖的点。

**参考答案**：
- KV cache 里存的是**已经施加过 RoPE 旋转的 K**（rotate 发生在写入 cache 之前），所以驱逐中间一段 KV 后，剩余 K 的相对位置关系保持不变，不需要重新编码——这是 RoPE 相对位置性质带来的便利。
- 真正要注意的是**"位置空洞"**：窗口滑出后，保留下来的 token 位置 id 不连续。对 full attention + RoPE 来说，attention 计算只用相对位置 (i-j)，空洞本身不引入数值错误，但语义上模型没见过训练分布中"中间缺一段"的模式，属于近似误差。
- 如果实现上为了省显存把 KV 重排压实（compaction），则必须同时维护逻辑位置 id，不能让 kernel 用物理偏移当位置用。

**追问**：Sliding Window Attention 模型（如 Mistral/Gemma）和这个有什么区别？（SWA 是模型结构本身只算窗口内注意力，驱逐是"无损"的；对 full attention 模型做窗口驱逐是近似）

---

### Q4. 为什么是每 1024 个 decode step 压缩一次？这个周期怎么定的？

**考察点**：工程权衡意识。

**参考答案**：
压缩本身有开销（要算一遍近期 token 对历史的注意力分数），频率太高会吃掉收益；太低则长上下文期间 KV 占用降不下来，并发收益滞后。1024 是按"典型请求 decode 长度的分数倍"选的——大部分请求在几百到几千 step 内结束，1024 保证长请求至少触发 1-2 次压缩，且单次压缩开销摊薄到千级 step 后可忽略。实际上这个值应该可配置，并按工作负载（平均输出长度、显存水位）调。

**追问**：更优策略？（按显存水位动态触发：free block 低于阈值即压缩，而非固定步数——vLLM/SGLang 的抢占/回缩都是水位驱动的）

---

### Q5. Gated Delta Attention 是什么？"按请求粒度分配 state slot"具体怎么做的？

**考察点**：对线性注意力/混合模型缓存的理解，这是 SGLang/vLLM 当前热点（Qwen3-Next、Jamba、Bamba 等 hybrid 模型）。

**参考答案**：
- **Gated DeltaNet（GDN）**是线性注意力变体：不存每个 token 的 KV，而是维护一个**固定大小的循环状态** `S ∈ R^{d_k × d_v}`（每个 head 一份），每步按 delta 规则更新：`S_t = α_t · S_{t-1}(I − β_t k_t k_tᵀ) + β_t k_t v_tᵀ`，输出 `o_t = S_tᵀ q_t`。另外 q/k/v 还过一个短卷积（kernel=4），所以还有一份 **conv state**（保存最近 3 个 token 的 q/k/v）。
- **缓存管理差异**：full attention 的 KV cache 随序列长度增长 O(L)，按 block 分页分配；GDN 的 state 是 **O(1) 固定大小**，与序列长度无关——所以按请求粒度分配：请求准入时分配一个 state slot（conv_state + ssm_state 两块固定 buffer，slot id 与请求绑定），decode 每步原地更新，请求结束释放 slot。
- **prefill 处理**：长 prompt 不能逐 token 循环，用 chunked 并行形式（chunk size 64，WY 表示/chunked delta rule scan）算输出的同时得到最终 state 写入 slot。
- hybrid 模型里 full attention 层走普通 paged KV pool，GDN 层走 state pool，调度器要为两类资源同时记账（SGLang 的 HybridLinearKVPool、vLLM 的 MambaManager 是同样思路）。

**追问**：GDN 层能做 prefix caching 吗？（很难：state 是全文压缩摘要，无法像 KV block 那样按前缀共享；工程上只对 full attention 层做 prefix cache，GDN 层重算）

---

### Q6. chunked-prefill 怎么实现的？为什么 TTFT 降 12%？对 TPOT 呢？

**考察点**：调度器核心机制，SGLang/vLLM 都有。

**参考答案**：
- **实现**：调度器每步有 token 预算（`max_num_batched_tokens`），prefill 请求不再一次性整段算完，而是按 chunk（如 2048 token）切片，和 running 队列里的 decode 请求**混在同一个 batch** 里跑；没算完的 prompt 下轮继续切。
- **TTFT 降低的原因**：原来一个 8K prompt 的 prefill 会独占整个 step（几十~几百 ms），期间 decode 全部停摆、新请求排队；chunk 化后长 prompt 被摊到多个 step，新到达请求不用等整段 prefill 结束就能插进来，排队延迟下降 → 平均 TTFT 降 12%。
- **对 TPOT**：混合组批让 decode 不再被长 prefill 卡住出大 bubble，TPOT 的毛刺（P99）显著改善；单步内 prefill token 会摊薄 decode 的算力，平均 TPOT 略有上升，用 `max_num_batched_tokens` 预算控制这个 trade-off。

**追问**：chunk 大小怎么选？（太小：kernel 启动/调度开销占比上升，prefill 总时长变长；太大：decode 饥饿依旧。一般取让单 step 耗时 ≈ decode 单步耗时的 2-5 倍，如 2048/4096/8192，要实测）

---

### Q7. AWQ W4A16 接入的坑？34% 显存、4897.9 tok/s 怎么测的？

**考察点**：量化落地的真实工程经验，数字必须能自圆其说。

**参考答案**：
- **接入坑**：① 权重布局——AWQ 是 group-wise（通常 g=128）int4 + fp16 scale/zero，pack 成 int32，加载时要按 kernel 要求的交错布局重排（qweight/qzeros/scales 三个张量）；② dequant 要 fuse 进 GEMM，逐权重反量化再 matmul 会比 FP16 还慢；③ 有些层不量化（lm_head、norm、embedding），state dict 要做混合精度映射；④ tied weights 处理。
- **34% 显存**：端到端（权重+KV+激活，固定 batch 和上下文）对比 FP16 baseline。权重本身从 16 bit→约 4.75 bit（int4 + g128 的 scale/zero 开销）省 ~70%，但 KV cache 和激活不变，所以端到端是 34% 而非 70%——这个数字是合理的，要能说清构成。
- **吞吐测试**：固定输入/输出长度（如 in=1024/out=512），batch 扫多档，开 continuous batching 打满，取稳定段 tokens/s = 4897.9。报告时要带硬件型号、模型、batch、上下文长度四个前提。

**追问**：为什么 decode 阶段 W4A16 收益大、prefill 反而可能变慢？（decode 是 memory-bound，瓶颈在权重搬运，int4 让权重读取量 ÷3.4，dequant 的少量计算被掩盖；prefill 是 compute-bound，dequant 增加额外指令，收益小甚至为负）

---

### Q8. nano-vllm 和生产级 vLLM/SGLang 差在哪？把你的功能移植到 SGLang 要注意什么？

**考察点**：是否读过生产框架源码，知道教学框架简化掉了什么。

**参考答案**：
差距：① 生产框架是**多进程架构**（调度、detokenize、worker 分离，ZMQ/共享内存通信），nano-vllm 单进程；② 生产有 prefix caching（SGLang radix tree / vLLM block hash）、CUDA Graph、torch.compile、TP/PP/DP/EP 分布式；③ 调度器有抢占/回缩、优先级、多级 KV pool（SWA/MLA/Mamba 混合）；④ 完善的 metrics、容错、PD 分离。
移植注意事项：驱逐策略要接入 radix cache 的淘汰逻辑并处理共享 block 的 refcnt；chunked-prefill 已有现成机制（`chunked_prefill_size`），我做的混合组批逻辑要对齐它的 token 预算接口；GDN state 管理要对齐 SGLang 的 `HybridLinearKVPool/MambaPool` 的 slot 语义，而不是自己造一套。

---

## Part 2 简历深挖：Triton/CUDA 算子库

### Q9. 手写 FlashAttention V2 的核心流程？online softmax 为什么省显存？

**参考答案**：
- 核心：把 Q 按 BLOCK_M 分块放 shared memory/寄存器，外层循环 K/V 的 BLOCK_N 块；每块算 `S_ij = Q_i·K_jᵀ·scale`，**不物化完整的 N×N 注意力矩阵**，而是用 running max `m_i` 和 running sum `l_i` 做 online softmax：新块到来时 `m_new = max(m_old, rowmax(S_ij))`，`α = exp(m_old − m_new)`，把已有累积 `acc` 和 `l` 乘 α  rescale，再累加 `exp(S_ij − m_new)·V_j`；最后 `O = acc / l`。
- 省显存：朴素实现要存 S 和 P（N²×2 bytes），FA 只存 O(N) 的 O/m/l，显存从 O(N²) 降到 O(N)，8K 序列从 256MB/head 降到 KB 级；同时减少 HBM 读写次数带来 2-4x 提速。
- Triton 实现要点：causal 场景把 K/V 块分"完全可见（不 mask）"和"对角线（需 mask）"两段处理，避免对全部块做 mask；`tl.dot` 用 tensor core；num_warps/num_stages 调参。

### Q10. FA2 相比 FA1 改进了什么？你的 1.8x 是相对什么 baseline？

**参考答案**：
- FA2 改进：① 减少非 matmul FLOPs（把 softmax 的除法、scale 移到循环外，非 matmul 部分是 tensor core 时代的瓶颈）；② 并行维度从 (batch, head) 增加到 (batch, head, **Q 序列块**)，长序列小 batch 时占满 SM；③ warp 级分工优化：warps 沿 Q 切分，每 warp 独立处理一段 Q 与全部 K/V，减少 shared memory 通信和同步。
- 1.8x 的 baseline 必须讲清楚：相对自己实现的朴素 Triton attention（物化 S/P）还是 torch SDPA？面试时如实说，并给出测试形状（B/H/N/D、dtype、causal）。（如果是 vs PyTorch naive，1.8x 偏保守合理；vs SDPA 1.8x 则非常亮眼，准备好被追问 profiling 数据）

### Q11. AWQ Dequant GEMM 怎么 fuse 的？

**参考答案**：
- 布局：int4 权重 8 个 pack 进 int32，kernel 内解包（shift + mask），按 group(g=128) 取 scale/zero，`w = (q − zero) × scale`，然后直接喂给 `tl.dot`/mma，**dequant 在寄存器里完成，不落显存**——这就是 fuse。
- 关键优化：① 解包用位运算 + 查表（LOP3 技巧）减少指令；② scale/zero 沿 K 维按 group 边界对齐加载，放 shared memory；③ K 维分块流水（num_stages），把 int4 加载和 dequant+dot 重叠。
- 收益来源：decode 时 GEMM 是 skinny（M=1~几十），瓶颈在把整份权重从 HBM 读进来，int4 读取量只有 fp16 的 1/3.4，dequant 计算被访存掩盖 → 1.6x。

### Q12. Tiled GEMM 怎么做到 CUTLASS 83%？差的 17% 在哪？

**参考答案**：
- 做法：block tile 128×128、thread tile 8×8（256 线程）；global→shared 用 `float4`/cp.async 向量化+多阶段流水（double/triple buffering）掩盖访存；shared memory 布局做 XOR swizzle（或 +padding）消除 bank conflict；寄存器 double buffer 让 FFMA 与下次 shared 加载重叠；epilogue 边界谓词处理。
- 差距 17%：CUTLASS 用了 tensor core（我的是 CUDA core FFMA）或更深的 software pipeline（cp.async 多阶段 + warp specialization），以及 swizzle 更精细。若我是 SGEMM 对比 cuBLAS 83% 属正常高水平；面试时讲清对比对象和数据类型。
- 要继续逼近：上 mma.sync（HMMA16816）、warp-level tile、pipeline 类封装。

### Q13. 什么是 bank conflict？你项目里怎么消除的？

**参考答案**：
- shared memory 分 32 个 bank（4B/bank），一个 warp 内 32 个线程若访问落在同一 bank 的不同地址，访问被串行化成 N 拍（N-way conflict）。
- 消除手段：① 数组最后一维 +1 padding 错开 stride；② XOR swizzle：`col ^= (row % 8)` 类打乱；③ 调整线程到数据的映射让同 warp 访问连续（coalesced 且跨 bank）。
- Steiner 项目里：候选 block 表放 shared memory，多个线程按步长并发探测，原始布局 8-way conflict，padding + 重排映射后消除，配合循环展开拿到这部分的加速。

### Q14. Steiner System 搜索：180x 加速怎么拆解？120G→20G 怎么压的？

**参考答案**（诚实拆解，这是体现 profiling 能力的好题）：
- 加速构成：GPU 并行本身（搜索树按层切到上万线程，二分查找分配任务）贡献大头；bank conflict 消除、循环展开、递归改显式栈迭代各贡献百分之几十到几倍；最终相对 CPU 单线程基线 180x。**准备一张纸能写出乘积拆解**（如 并行 40x × 访存优化 2x × 指令优化 2.2x ≈ 180x）。
- 显存压缩：搜索状态原本是稀疏集合的显式存储（120G），改成 bitset（每个候选三元组 1 bit）+ 前缀索引，压到 20G；SQS(16) 再配合更强剪枝（同构排除/排序不变性）拿到 273x。
- 面试价值点：算法剪枝 >> 工程微优化，两个项目数字说明我理解"先降复杂度、再压常数"。

---

## Part 3 SGLang 架构与二次开发（岗位直接相关，重点准备）

### Q15. SGLang 整体架构？一个请求从 HTTP 到返回经过哪些组件？

**参考答案**：
- 多进程架构：`HTTP Server / TokenizerManager`（asyncio，做 tokenize、会话管理、流式转发）→（ZMQ IPC）→ `Scheduler`（核心：radix cache 管理、组批、调度、launch forward；TP>1 时每 rank 一个 scheduler 进程，SPMD 锁步运行）→ `DetokenizerManager`（增量 detokenize、流式回传）。
- 请求路径：HTTP → TokenizerManager 编码 → Scheduler 入 waiting 队列 → 命中 radix cache 拿到已算前缀 → 组 prefill batch（可能 chunked）→ TpModelWorker 前向 → 采样 → 进入 decode 循环（每步一个 token，continuous batching）→ DetokenizerManager 增量解码 → SSE 流式返回。
- 配套组件：`RadixCache`（prefix 复用）、`ReqToTokenPool / TokenToKVPool`（KV 显存池）、`CudaGraphRunner`（decode 图捕获）、grammar backend（结构化输出）、disaggregation 模块（PD 分离）、sgl-router（多实例路由，cache-aware）。

### Q16. RadixAttention 原理？和 vLLM 的 block-hash prefix cache 有何区别？

**参考答案**：
- **RadixAttention**：所有请求的 token 序列组织成一棵 radix tree（边=token 片段，节点存对应的 KV pool 索引）。新请求做最长前缀匹配，命中部分直接复用 KV；未命中部分算完后 split 节点插入树中。节点带 last_access 时间戳，显存不足时按 **LRU 从叶子淘汰**。
- **对比 vLLM**：vLLM 是扁平的 `hash(block tokens) → block` 映射，O(1) 查找、block 粒度 LRU（free queue 顺序）；SGLang 树结构额外支持：① **cache-aware 调度**——waiting 队列按前缀匹配长度排序（lpm 策略），把共享前缀的请求排在一起最大化命中；② 多轮对话/agent 场景天然表达分支历史；③ 前缀不必 block 对齐（page_size=1 时 token 粒度）。
- 代价：树的 match/insert/split 在 CPU 热路径上，单请求开销比 hash 查表高；vLLM 胜在简单快。两者都能跨请求共享。

### Q17. SGLang 的 overlap scheduler 是干什么的？

**参考答案**：
- 问题：正常循环里"GPU 跑完 step t → 同步等结果 → CPU 采样/更新请求状态/组 batch t+1 → 再 launch"是串行的，CPU 那段（Python 调度 + radix cache 操作 + batch 元数据构建，可达数 ms）期间 GPU 空转。
- overlap scheduler 让 **GPU 执行 batch t 的同时，CPU 准备 batch t+1**：batch t 的 next token 通过异步拷贝 + future 机制延迟解析，调度器先按"假设成功"推进，结果在下轮真正消费前解析校验。
- 收益：消除 GPU bubble，小 batch/decode 场景吞吐提升明显（官方 ~10-30% 级别）。vLLM V1 的 async scheduling 是同一思路。
- 代价：实现复杂（错误处理、abort 请求的回滚），spec decode 与 overlap 叠加时更复杂。

### Q18. SGLang 怎么做结构化输出？开销在哪？

**参考答案**：
- JSON Schema/regex 先编译成 **FSM**（后端用 xgrammar；早期是 Outlines）；每步 decode 由 FSM 当前状态算出**合法 token 的 bitmask**，作为 logits processor 把非法 token 置 -inf，采样保证输出恒合法。
- **jump-forward 优化**：当后续片段无歧义（如固定的 `", "` 或括号），直接快进追加 token，跳过模型调用。
- FSM 状态按（schema, 已生成前缀）缓存，避免每步重编译。
- 开销：bitmask 计算在 CPU 每步一次（可 overlap），整体延迟增加通常 <5%；收益是电商场景（商品信息抽取、工具调用、接口参数生成）免去重试和解析失败。

### Q19. SGLang 的 PD 分离怎么实现的？

**参考答案**：
- prefill 实例和 decode 实例分开部署，中间用 **KV transfer** 连接（后端可选 Mooncake 或 NIXL，走 RDMA/GPUDirect）。
- 流程：请求先到 prefill 实例 → decode 实例侧先**预分配 KV 池空间**并注册（bootstrap server，每 rank 一个端口）→ prefill 算完后**逐层把 KV 推给 decode**（与计算 overlap）→ decode 侧收齐（kvcache receipt）后开始 decode。
- 收益：TTFT 与 TPOT 解耦，两类节点独立扩缩容（prefill 用算力强的、decode 用显存带宽强的）；长 prompt 不再阻塞 decode 集群。
- 挑战：传输与计算的重叠编排、block 布局一致性（page size/后端要匹配）、故障恢复、多实例路由（mini_lb / sgl-router 感知角色）。

### Q20. 什么是 DP attention？为什么 DeepSeek MLA 需要它？

**参考答案**：
- MLA 的 KV 是压缩潜变量（c_kv 512 + k_rope 64 = 576 维/token/层）。TP 下每个 rank 要拿**完整的潜变量**才能投出自己那部分 head 的 K/V，所以潜变量 KV 在 TP rank 间是**复制的**——TP 开得再大，每卡 KV 显存不降。
- DP attention：attention 部分按**数据并行**切请求——每 rank 只存自己那份请求的 KV，KV 显存 ÷DP 数；attention 算完后 gather hidden states 给后面的 MoE（EP/TP）部分，算完再 scatter 回去。
- 这是 SGLang 服务 DeepSeek 的关键特性（`--enable-dp-attention`），配合 EP MoE（DeepEP、EPLB 负载均衡）才能把 V3/R1 的每卡 KV 占用压下来、并发拉上去。

### Q21. SGLang 的投机采样（EAGLE）流程？什么时候用？

**参考答案**：
- EAGLE：一个单层的 draft 模型，输入 target 模型的 hidden states + token embedding，自回归展开一棵草稿树（top-k 扩展、限深度限节点数）；target 模型一次前向 + **tree attention mask** 并行验证整棵树，按采样/贪心规则接受最长一致前缀 + 1 个 bonus token。SGLang 支持 EAGLE/EAGLE-3，DeepSeek 场景用 MTP（nextn 头）。
- 收益：decode 从"一步一 token"变"一步多 token"（平均接受长度 2-4），低并发下延迟降 30-60%。
- 代价：draft 前向开销 + tree attention 变长 + 拒绝 token 的算力浪费。**batch 大（compute-bound）时反而掉吞吐**——电商在线低峰 latency-sensitive 场景开，高峰打满时关或降级。

### Q22. 如果要你在 SGLang 接入一个新 hybrid 模型（full attention + 线性注意力），改哪些地方？

**参考答案**（这是把简历 Q5 迁移到岗位的送分题）：
1. `models/` 新增模型定义：config 解析、权重名映射、forward；注册进 model registry；
2. **KV 池选型**：hybrid 用 `HybridLinearKVPool`（full attention 层走 paged KV pool，线性层走 MambaPool 式的 state slot），`model_runner` 里按层类型组装；
3. attention backend：线性层复用/新增 mamba 类 kernel（conv + chunked scan prefill + recurrent decode）；
4. radix cache：只对 full attention 层做 prefix cache，线性层 state 不参与，注意 page 对齐与 slot 分配时机；
5. CUDA Graph 捕获路径验证（hybrid 模型的 capture 有特殊约束）；
6. 正确性：与 HF 实现逐 token 对 logits → 精度测试 → benchmark（TTFT/TPOT/吞吐）。

---

## Part 4 推理框架核心

### Q23. continuous batching 原理？和 static batching 对比？

**参考答案**：static batching 等 batch 内所有请求都结束才一起返回并换下一批，GPU 大量空转（生成长度方差大时浪费严重）。continuous batching（Orca 提出）把调度粒度从"请求"降到"**iteration**"：每个 decode step 结束都检查——有请求完成就移除并立即补新请求进来。配合 KV 分页管理，GPU 始终打满，吞吐可提升数倍（Orca 论文相对 FasterTransformer 量级提升）。

### Q24. PagedAttention 解决什么问题？block 大小怎么选？

**参考答案**：
- 解决：传统按 max_len 预分配连续显存 → 内部碎片（长度不确定）+ 外部碎片（大小不一的分配）→ 实测浪费 60-80%。PagedAttention 借鉴 OS 虚拟内存：KV 按固定大小 block（页）分配，逻辑 block 经 block table 映射到物理 block，按需增长、用完释放。
- block 大小权衡：小（16）→ 碎片少、prefix 共享粒度细，但 block table 项多、kernel 循环次数多；大 → 反之。vLLM 默认 16 是实测平衡点；SGLang radix cache 下 page_size 可为 1（树本身管细粒度）。

### Q25. KV cache 显存计算（必考计算题）

**公式**：`bytes/token = 2(K,V) × num_layers × num_kv_heads × head_dim × dtype_size`

| 模型 | 参数 | 每 token KV | 16K 上下文 |
|---|---|---|---|
| LLaMA-3-8B | 32 层, 8 KV头(GQA), dim 128, fp16 | 2×32×8×128×2 = **128KB** | ≈ 2GB |
| Qwen2.5-72B (TP8) | 80 层, 8 KV头, dim 128, fp16 | 2×80×8×128×2 = 320KB，每 rank ÷8 = **40KB** | 每卡 ≈ 640MB |
| DeepSeek-V3 (MLA) | 61 层, 潜变量 576, fp16 | 61×576×2 ≈ **69KB**（TP 下各 rank 复制） | ≈ 1.1GB |

反向题也要会：80G 卡、权重 16G、利用率 0.9 → KV 池 ≈ 56G → 能容纳 token 数 = 56G/128KB ≈ 43.7 万 token。

### Q26. prefill 和 decode 的特性差异？

**参考答案**：prefill 是 compute-bound（M=L 的大 GEMM，tensor core 打满），一次处理整段 prompt，决定 **TTFT**；decode 是 memory-bound（M=1~batch 的 skinny GEMM，每步读全部权重 + 读全部历史 KV，算力利用率常 <5%），逐 token 生成，决定 **TPOT**。推论：① 优化手段不同——prefill 靠并行/算子/量化计算，decode 靠减访存（量化权重、KV 量化、CUDA Graph、spec decode）；② 调度上不能互相阻塞（chunked prefill / PD 分离）。

### Q27. 请求抢占怎么做？

**参考答案**：显存不足时调度器要牺牲部分请求。vLLM V0：recompute（丢弃 KV 回 waiting）或 swap（KV 搬到 CPU 内存）；V1 只保留 recompute（配合 prefix cache 重算代价小）。SGLang：retract_decode——把运行最久/代价最大的请求逐出 running，释放其末尾 KV，前缀仍留在 radix tree 里，后续重新调度时命中缓存。要点：抢占要保证公平（避免饿死）、尽量避免级联触发。

### Q28. 投机解码什么场景有用、什么场景有害？

**参考答案**：有用：低 batch（memory-bound，有大量空闲算力）、接受率高的场景（代码、模板化文本、电商话术），延迟可降 30-60%。有害：batch 打满（compute-bound，验证的 tree attention 和草稿开销是纯浪费）、接受率极低（draft 质量差）。所以生产上常按 batch size 阈值动态开关。

### Q29. TTFT / TPOT / 吞吐 / goodput？电商在线场景关心哪个？

**参考答案**：TTFT=首 token 延迟（prompt 排队+prefill）；TPOT/ITL=相邻 token 间隔；吞吐=系统总 tokens/s；goodput=满足 SLO（如 TTFT<2s, TPOT<100ms）的最大请求速率。电商导购/客服是在线交互：用户首屏体验看 **TTFT P99**，阅读流畅度看 **TPOT**（人阅读速度 ~10-20 字/s，TPOT <100ms 即可），成本看 goodput。通常以 SLO 约束下的 goodput 为优化目标，而不是裸吞吐。

### Q30. prefix caching 的命中粒度与淘汰？多轮对话收益？

**参考答案**：命中粒度：vLLM=block（16 token）对齐；SGLang radix tree 可到 token 级（page_size=1）。淘汰都是 LRU。多轮对话：第 n 轮的 prompt 包含前 n-1 轮全部内容，prefix cache 命中后每轮只需 prefill 新增部分，TTFT 大幅下降——电商客服/导购多轮场景收益极大，这也是 SGLang radix + cache-aware 调度的主战场。

---

## Part 5 分布式与性能优化

### Q31. TP / PP / DP / EP 怎么切、怎么选？

**参考答案**：
- **TP**：层内切（QKV/MLP 列切+行切），每层 2 次 all-reduce，通信量大 → 限单机 NVLink 内（≤8）；降显存 + 降延迟。
- **PP**：层间切，P2P 传激活，有 bubble（microbatch 缓解）→ 跨节点用。
- **DP**：整模型复制，各跑不同请求，无通信 → 提吞吐（推理侧常配合 DP attention）。
- **EP**：MoE 专家切到不同卡，token 路由后 all-to-all 收发 → DeepSeek 类大 MoE 必用，配 DeepEP 内核 + EPLB 专家负载均衡。
- 选型：单机 8 卡能放下 → TP8 最省事；MoE → EP(+DP attention)；跨节点 → TP×PP 组合。

### Q32. MoE 的 EP 有哪些坑？

**参考答案**：① 专家负载不均（热门专家所在卡拖尾）→ EPLB 动态冗余/迁移专家；② all-to-all 通信与计算重叠（DeepEP 的 dispatch/combine 分 normal/low-latency 模式，prefill 用大吞吐模式、decode 用低延迟模式）；③ 小 batch 时 EP 通信占比上升，可能不如 TP；④ 显存要预留通信 buffer。

### Q33. CUDA Graph 为什么加速 decode？限制？

**参考答案**：decode 一步要 launch 几百个小 kernel，每个 launch CPU 开销 5-10μs，Python 框架更慢，GPU 在 kernel 间隙空转。CUDA Graph 把整个 step 录制成一张图，一次 launch 重放 → 消除 launch gap（decode 提速可达 10-30%）。限制：shape 必须静态 → 按 batch size 档位各 capture 一张、运行时 pad 到最近档；输入输出用静态 buffer（拷贝进图地址）；图内不能有 CPU 同步/动态分配；prefill（变长）一般不入图；每张图占显存（vLLM/SGLang 用 graph pool 共享缓解）。

### Q34. decode 一步的时间花在哪？怎么 profiling？

**参考答案**：memory-bound：大头是 GEMM 读权重（占 60-70%）+ attention 读 KV（随上下文增长）+ 其余（norm/RoPE/采样 <10%）。方法论：nsys/torch profiler 看 kernel 时间分布；算 arithmetic intensity 对照 roofline 验证是否贴访存带宽；先看有没有 CPU 间隙（没开 CUDA Graph 时）；再逐 kernel 优化。

### Q35. NCCL all-reduce ring vs tree？小消息怎么优化？

**参考答案**：ring：每卡传 2(N−1)/N 份数据，带宽最优（大消息），延迟随 N 线性；tree/double-tree：延迟 logN（小消息优）。decode 的 TP all-reduce 是小消息（batch×hidden×2B，几十 KB），延迟敏感 → 用 **custom allreduce**（vLLM/SGLang 都有：one-shot 直接 NVLink P2P 读写聚合，或 NVLS/NVLink SHARP 在交换机内归约），绕开 NCCL 通用路径。

---

## Part 6 量化

### Q36. AWQ 原理？与 GPTQ、SmoothQuant 区别？

**参考答案**：AWQ：观察到 ~1% 的权重通道对激活异常值敏感（salient），按激活幅度找出来后对其做 per-channel 放大（等效保护），再做 group-wise(g=128) 非对称 int4 量化；不依赖逐列误差补偿，校准快、泛化好。GPTQ：基于 OBQ，逐列量化并用 Hessian 逆把误差补偿到未量化列，精度好但校准慢、顺序敏感。SmoothQuant：把激活的量化难度按 per-channel  scale 迁移到权重上（W 除 s、X 乘 s），从而激活也能 int8 → 支持 W8A8。

### Q37. W4A16 / W8A8 / FP8 怎么选？

**参考答案**：W4A16：decode memory-bound 场景，显存/带宽收益最大，精度有损，需 dequant kernel；W8A8(int8)：prefill/compute-bound 也受益（int8 tensor core），精度损失略大，要校准；FP8(W8A8 e4m3，H100+)：精度几乎无损，是当前主力，per-tensor 或 blockwise(128×128) scale；Blackwell 上 NVFP4。电商在线 decode 重 → W4A16 或 FP8 KV + FP8 GEMM 组合常见。

### Q38. 量化模型上线前怎么验证？

**参考答案**：离线：perplexity（wiki/c4）+ 下游任务集（MMLU/CMMLU/业务评测集）对比 FP16 基线，业务侧加抽样人工评估；线上：shadow 流量对比输出分布/logprob 偏移 → 小流量灰度看业务指标（转化率、解决率）+ 延迟/成本指标 → 全量，保留一键回滚。

---

## Part 7 手撕代码（高频，准备到能默写骨架）

### C1. LRU Cache（LeetCode 146）——必会，会引申到 prefix cache 淘汰

要点：哈希表 + 双向链表（get 移到头部、put 满了删尾部）。讲完主动引申：vLLM free queue 按 LRU 驱逐 prefix block、SGLang radix tree 按 last_access 淘汰叶子，都是这套思想，只是粒度从 entry 变成 block/节点，且要处理共享引用（refcnt>0 不可淘汰）。

### C2. 简化 continuous batching 调度器

```python
class Scheduler:
    def __init__(self, max_batched_tokens, max_seqs, kv_pool):
        self.waiting, self.running = deque(), []
    def step(self):
        # 1) 先保障 running：预算内每请求 1 个 decode token
        budget = self.max_batched_tokens
        for r in self.running: budget -= 1
        # 2) 从 waiting 准入 prefill（chunked：只取 budget 允许的 token 数）
        while self.waiting and budget > 0 and len(self.running) < self.max_seqs:
            r = self.waiting[0]
            take = min(r.remaining_prompt, budget)
            if not kv_pool.can_alloc(take + r.max_output): break  # 准入检查
            kv_pool.alloc(take); r.remaining_prompt -= take
            budget -= take
            if r.remaining_prompt == 0: self.running.append(self.waiting.popleft())
            else: break  # 该请求 prompt 没切完，下轮继续
        # 3) 显存不足则抢占队尾 running 请求（recompute）
        while kv_pool.will_oom():
            victim = self.running.pop(); kv_pool.free(victim)
            self.waiting.appendleft(victim)
        return self.running
```
要点：decode 优先、token 预算、准入检查、抢占回退，讲清 starvation 与公平性。

### C3. Paged KV block manager

要点：free list（deque）+ `refcnt[block_id]` + `hash(block_tokens)→block_id` 的 prefix 表；alloc：先查 hash 命中（refcnt+1 复用）否则取 free block；free：refcnt−1，归 0 时入 free list 尾部（但 hash 保留，命中即复活；free list 头部被淘汰时才删 hash）；get/append 满 16 token 才固化成可共享 block。这就是 vLLM `BlockPool` 的迷你版。

### C4. Triton decode attention（online softmax）骨架

```python
@triton.jit
def decode_attn(Q, K_cache, V_cache, O, block_tables, seq_lens,
                scale, BLOCK_N: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)                      # 一个 (bs*head) 一个 program
    q = tl.load(Q + pid * D + tl.arange(0, D))  # [D]
    m_i, l_i, acc = -float("inf"), 0.0, tl.zeros([D], tl.float32)
    seq_len = tl.load(seq_lens + pid // H)
    for start in range(0, seq_len, BLOCK_N):    # 按 KV block 流式遍历
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < seq_len
        # 经 block_tables 间接寻址取物理 block 的 K/V
        k = tl.load(K_cache + phys_addr(offs), mask=mask, other=0.)   # [BLOCK_N, D]
        s = tl.sum(q[None, :] * k, 1) * scale                         # [BLOCK_N]
        s = tl.where(mask, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None] if False else s - m_new)
        v = tl.load(V_cache + phys_addr(offs), mask=mask, other=0.)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        l_i = l_i * alpha + tl.sum(p, 0)
        m_i = m_new
    tl.store(O + pid * D + tl.arange(0, D), acc / l_i)
```
要点：讲清 indirect load（block table 查物理页）、online softmax 三行更新、mask。

### C5. top-k / top-p 采样

要点：top-k：logits 取 topk，其余 −inf，softmax 后采样；top-p(nucleus)：按概率降序排序、累积和 > p 的尾部砍掉、重归一化；温度先除 T；工程上用 gumbel-max trick（argmax(logits/T + G)）避免显式采样，spec decode 的 verify 也用它。

---

## Part 8 行为面

### B1. 腾讯实习最有挑战的事？817→44ms 怎么做的？

**答题框架（STAR + 方法论）**：广告首帧慢 → 埋点 trace 拆解耗时（布局计算阻塞主线程 X ms + 图片加载 Y ms + 网络 Z ms）→ 三个优化对应三个瓶颈：JS 引擎**预布局**（加载期提前算好布局）、**MMKV 高度缓存**（同尺寸内容直接命中免计算，注意缓存清理策略防膨胀）、**封面图预渲染存储** → 灰度对比验证 817→44ms。收尾强调方法论：**先测量定位瓶颈、缓存与异步化是通用武器、用数据验证收益**——这套方法同样适用于推理优化（先 profile 找 bubble，再上 CUDA Graph/overlap/缓存）。

### B2. 客户端经历和推理框架岗有什么关系？（必被质疑）

**参考答案**（诚实 + 迁移）：领域确实不同，但 ① 性能工程方法论完全通用：trace 拆解 → 瓶颈定位 → 缓存/异步/预计算三板斧 → 数据验证；② 客户端资源受限（内存/主线程）与推理显存受限是同一类约束思维；③ 框架开发经验（通用组件接口、自闭环测试环境）直接对应推理框架的模块化与可测性；④ 个人在课余系统补了推理方向（nano-vllm 改造 + 算子库 + 读 vLLM/SGLang 源码），这是主动转型而非临时起意。

### B3. 为什么做推理框架？为什么快手电商？

要点：推理是 LLM 落地的成本与体验瓶颈，系统优化杠杆大；自己喜欢"离硬件近、能用数字衡量收益"的工作（竞赛+CUDA 项目佐证）。快手电商：真实高并发在线场景（导购/客服/内容生成），对延迟和成本敏感，工程挑战真实；团队基于 SGLang 二开，能接触生产级问题（PD 分离、cache 优化、新模型接入）。

### B4. 时间安排？

每周 4-5 天、连续 3-6 个月（简历一致）；可表达：希望 3 个月内 own 一个完整模块（新模型接入或某类 kernel 优化），能长期更好。

### B5. 读过 SGLang 哪些源码？（诚实作答 + 展示路径）

建议提前按附录 A 读一遍，面试时说：读过 scheduler 的 event loop 与 overlap 机制、radix cache 的树管理、mem_cache 的 KV pool 抽象，正在看 eagle worker 和 PD 分离。切忌假装精通——被深挖时承认边界，并说出"如果让我接手，我会从这里入手"。

---

## Part 9 反问环节（选 2-3 个）

1. 团队目前线上推理的主要瓶颈在 TTFT、TPOT 还是成本？最近在攻克什么优化方向？
2. PD 分离和 prefix cache 在电商业务里的实际命中/收益情况？多轮导购场景 cache 命中率大概什么水平？
3. 实习生 3-6 个月能独立 own 的模块一般是什么粒度？有没有 mentor 带？
4. 新模型接入（如下一代 Qwen/DeepSeek）从发布到上线的流程和周期大概怎样？

---

## 附录 A：面试前冲刺清单（SGLang 源码阅读路径，按优先级）

| 优先级 | 文件 | 看什么 |
|---|---|---|
| P0 | `python/sglang/srt/managers/scheduler.py` | event_loop_normal / event_loop_overlap、get_next_batch_to_run、run_batch、process_batch_result、retract_decode |
| P0 | `python/sglang/srt/mem_cache/radix_cache.py` + `radix_tree.py` | match_prefix / insert / split / evict（LRU） |
| P0 | `python/sglang/srt/managers/schedule_batch.py` + `schedule_policy.py` | batch 组织、PrefillAdder、lpm 策略、chunked prefill |
| P1 | `python/sglang/srt/mem_cache/memory_pool.py` | ReqToTokenPool、MHATokenToKVPool、MLATokenToKVPool、SWA / HybridLinear pool |
| P1 | `python/sglang/srt/model_executor/cuda_graph_runner.py` | capture_bs、pad、静态 buffer |
| P1 | `python/sglang/srt/layers/attention/` | flashinfer_backend / triton_backend / fa3 / trtllm_mla 的 forward_extend vs forward_decode |
| P2 | `python/sglang/srt/speculative/eagle_worker.py` | draft → verify → accept 流程、tree_mask |
| P2 | `python/sglang/srt/disaggregation/` | mooncake / nixl、bootstrap、逐层 KV 传输 |
| P2 | `python/sglang/srt/constrained/` | xgrammar backend、bitmask logits processor |
| P3 | `sgl-kernel` 仓库 | custom allreduce、moe、quant kernel 的接入方式 |

**配合动作**：① 把 vLLM 的 `v1/core/sched/scheduler.py`、`kv_cache_manager.py` 对照读（已有基础）；② 跑通 `sglang.launch_server` + 官方 benchmark，开 `--enable-radix-cache` 对比 TTFT；③ 准备一张自己画的"SGLang 请求生命周期图"。

## 附录 B：数字/公式速查卡

- KV bytes/token = `2 × layers × kv_heads × head_dim × dtype`；LLaMA-3-8B = 128KB/token
- AWQ g128 实际位宽 ≈ 4 + 32/128 = 4.25~4.75 bit/param；8B 模型 FP16 16GB → W4 ≈ 4.5GB
- FA 显存 O(N) vs 朴素 O(N²)；计算量不变 O(N²d)，省在访存
- ring allreduce 数据量 2(N−1)/N·S；decode all-reduce 消息 ≈ batch×hidden×2B（几十 KB）
- CUDA Graph：消除 kernel launch gap（每个 5-10μs × 每步数百 kernel）
- 连续批处理收益来源：消除"最短请求等最长请求"的空转
- spec decode 净收益 ≈ 接受长度 ÷ (1 + draft成本 + verify额外成本)，低 batch 才为正
- TTFT ≈ 排队时间 + prefill 时间；TPOT ≈ 单 decode step 时间
