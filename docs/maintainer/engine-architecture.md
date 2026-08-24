# NInfer Engine 架构

本文是 NInfer 当前 Engine 架构、执行所有权和请求生命周期的唯一顶层维护者权威。它描述已经落地的
系统，而不是迁移历史、备选设计或未来功能草案。

本文回答四个问题：

1. 一个请求从产品入口到 GPU 执行和结果发布经过哪些边界；
2. Scheduler、ResourceManager 和 Program 分别拥有哪类决策与状态；
3. admission、batch transaction、continuation 和 response lifetime 如何闭合；
4. 当前边界如何保持单 GPU 小并发场景的正确性和性能。

ResourceManager 的checkpoint、Device/Host replica和materialization contract由
[资源调度与上下文缓存架构](resource-scheduling-and-context-cache.md)定义；Paged KV的页几何、物理layout、
allocator和consumer contract由[Paged KV Context Store](paged-kv-cache.md)定义。模型数学、Artifact、Op和
外部协议也分别由文末列出的专项文档定义；本文不复制这些内容。

---

## 1. 产品模型与范围

### 1.1 当前工作负载

NInfer 的运行期模型是：

- 一张 GPU；
- 一个 resident model instance；
- Engine 启动时固定 `max_concurrency=1..8`；
- 有界 FIFO ingress；
- 无 request preemption；
- 每个 round boundary 将全部 decode-ready requests 组成一个 compact batch；
- Text、Vision、MTP、prefix reuse、CLI 和 serving 通过同一个公共 `ninfer::Engine` 路径；
- 35B-A3B target 还可以在 text-only 模式使用 DFlash。

`max_concurrency` 是同时 active 的请求上限，不是 shared KV capacity 的等分因子。单个请求可以使用
shared pool 的大部分容量，只要当前 active、catalogued 和transaction reservation的aggregate unique
occupancy仍然可满足。

### 1.2 当前非目标

以下能力不属于当前架构：

- 大规模或抢占式 continuous batching；
- priority/QoS、多租户公平性或跨 Engine 调度；
- 多 GPU sequence placement；
- active request preemption或swap；
- weight offload、跨Engine context storage或后台不可观察的cache worker；
- runtime model discovery 或字符串驱动的通用执行图；
- 为未注册模型、其他 GPU 架构或未来 storage tier 预留占位接口。

这些能力未来可以改变产品模型，但不能被当作解释当前代码的隐含前提。

### 1.3 设计中心

当前架构围绕三个原则建立：

1. **请求级 policy 与模型执行分离。** Engine 决定“谁在何时运行”，Program 决定“选中的工作如何在
   GPU 上完成”。
2. **逻辑资源与物理状态分离。** ResourceManager 维护 admission ledger 和 continuation catalog，Program
   维护 physical lane、KV、model state 和 epoch。
3. **所有可观察变化发生在 boundary transaction 之后。** GPU 产生的 token 先是 provisional，只有
   Frontend policy、Runtime state 和资源 ledger 全部提交后才向 consumer 发布。

---

## 2. 四层架构

NInfer 的请求路径分为四层：

```text
Gateway
  │  product/protocol request, response transport
  ▼
Frontend
  │  PreparedPrompt, OutputSession
  ▼
Engine
  │  scheduling decision, logical resource choice, output transaction
  ▼
Runtime
     physical state transition, model execution, CUDA work
```

这四层是所有权边界，不要求公共 API 中出现四个同名对象。公共 `ninfer::Engine` facade 同时暴露
Frontend 的 `prepare` 和 Engine 的 `submit/generate`；HTTP server、CLI 和直接 C++ caller 都通过这条
产品路径进入。

### 2.1 Gateway

Gateway 是产品适配层，包括 HTTP serving、CLI 和其他调用公共 Engine 的入口。它负责：

- 协议解析、鉴权、连接和 streaming transport；
- message/tool/media 输入到公共 `PromptInput` 的转换；
- media acquisition 的 URL/path/data 处理；
- API error、usage 和 response schema；
- transport/request lifetime 与 Gateway 自身的容量限制。

Gateway 不选择 physical lane，不检查 prefix continuation，不参与 batch formation，也不解释模型状态。
直接使用 C++ `Engine` 时可以没有独立 Gateway 对象，但边界仍然成立。

### 2.2 Frontend

Frontend 拥有 target family 的输入和输出语义：

- tokenizer、chat template、Vision preprocessing 和 MRoPE prompt construction；
- `PreparedPrompt` 的 owning representation 与 prefix identity；
- stop policy、thinking/content channel、增量 detokenization 和最终文本；
- 独立于 presentation mode 的 thinking phase/cap 状态；
- `OutputSession::preview_model`、`preview_control`、`preview_terminal` 与 `commit_preview`。

Frontend 产生两个关键对象：

```text
PreparedPrompt  = 已完成输入语义转换、可被 Runtime 规划的 owning prompt
OutputSession   = 一条请求独占的、可 preview/commit 的输出语义状态
```

Frontend 不持有 admission ledger、physical sequence、KV 或 CUDA Graph，也不决定请求顺序。

### 2.3 Engine

Engine 是请求控制平面。它负责：

- bounded outstanding capacity 和 FIFO pending queue；
- `RequestRecord`、active slots、deadline 和 cancellation；
- Scheduler policy；
- ResourceManager 的逻辑资源与 continuation catalog；
- prefill/decode/target-control boundary orchestration；
- OutputSession 与 Runtime 之间的 commit transaction；
- generation accounting、event queue 和最终 response completion；
- Runtime failure 后的 Engine-wide cleanup。

Engine 理解 request、queue、budget、finish reason 和 publication，但不理解 transformer layer、KV plane、
MTP state selector 或 graph capture 实现。

### 2.4 Runtime

Runtime 的入口是 exact package 暴露的 `Program` contract。它负责：

- physical lane 与 lane epoch；
- active sequence、catalogued continuation/checkpoint 和 target persistent state；
- Main/backend KV allocation、frontier 和 block-table binding；
- prefill、ordinary decode、MTP/DFlash speculative schedule；
- provisional model writes、accepted-prefix commit 和 rollback；
- workspace、request transient arena 和 CUDA Graph；
- generation timings、speculative statistics 和 physical memory summary。

Runtime 不维护 FIFO、protected head 或 backfill policy，不决定 continuation 是否值得保留，也不直接发布
用户输出。

### 2.5 Core、Ops 与 Startup

`src/core` 和 `src/ops` 是 Runtime 的实现基础，不是额外的请求层：

- Core 提供 device/tensor/view、arena、CUDA Graph RAII、paged storage 和 raw state transfer；
- Ops 提供数学或状态转换闭合的 CUDA 实现；
- target family Runtime 组合这些能力形成完整 Program；
- exact package 提供注册 identity、weights binding、immutable model view 和 execution leaves。

Startup 是从 Artifact 到一个可执行 Engine 的构建流程，也不是第五层。运行期调度只在所有 startup
资源和 physical address 已冻结后开始。

---

## 3. Package 与启动边界

### 3.1 Compile-time package contract

公共 Engine 在构造时读取 `.ninfer` identity，并从 closed registry 选择 exact package。当前 27B 与
35B-A3B package 是 peer compile-time variants；Qwen3.6 family 提供共享算法，但 Engine worker 不在运行期
分派 family 类型。

每个 package 向通用 EngineCore 提供同一组语义别名：

```text
PreparedPrompt / OutputSession / PublishedOutput
RequestBasePlan / AdmissionPlan
SequenceHandle / ContinuationHandle / PendingBatch
PrefillProgress / StartResult / CommitResult / FinishResult
Program
```

这些 owning、target-dependent 类型通过 package alias 到达 Engine。`src/runtime/contract` 只保存
`AdmissionResources`、`CommitDecision`、`RetentionDecision` 等 package-neutral value contracts。

因此，EngineCore 可以针对 package 编译，却不包含 Qwen layer、backend state 或 artifact identity 分支；
Program 可以针对 exact Variant 编译，却不获取 serving request 或 Scheduler policy。

### 3.2 Startup flow

Engine 构造按以下顺序闭合：

```text
validate EngineOptions
  -> read .ninfer identity
  -> select exact registered package
  -> build artifact load plan and target SequencePlanner
  -> preflight KV capacity against planned weight bytes
  -> materialize weights/resources and construct immutable LoadedModel
  -> resolve final KV capacity from current device memory
  -> finalize SequencePlan
  -> construct Frontend and Program
  -> construct EngineCore, Scheduler and ResourceManager
```

`SequencePlan` 是 target 对完整 physical runtime 的启动期规划，包含 fixed state、KV pool、workspace、
request transient 和 graph 所需容量。`Program::admission_capacity()` 产生运行期 ledger 的总容量；
ResourceManager 不从模型 geometry 重算它。

### 3.3 Engine-lifetime resources

权重、KV slabs、fixed per-lane state、device block tables、shared workspace、request transient backing 和
decode graph storage 都在启动期建立。请求期间可以改变 logical ownership、page mapping 和 frontier，但不
重新建立这些 device allocations。

`memory_summary()` 和 `reset_memory_peaks()` 通过 Engine 的 execution ownership 与 Program mutation 串行，
不会并发观察正在提交或清理的 physical state。

---

## 4. Engine 内部所有权

### 4.1 唯一 authority

| 状态或决策 | 唯一 authority |
|---|---|
| protocol、connection、response transport | Gateway |
| prompt construction、stop/output 语义 | Frontend |
| pending queue、RequestRecord、active slots、event publication | EngineCore |
| FIFO head、protection epoch、backfill、prefill owner、round membership | Scheduler |
| active/catalog ownership、unique-occupancy ledger、candidate/pressure choice | ResourceManager |
| physical lane epoch、model/KV state、PendingBatch、GPU execution | Program |

“唯一 authority”不表示只有一个对象读取该信息，而是只有表中的 owner 可以定义其状态转换。其他组件通过
summary、grant、choice 或 opaque capability 观察它。

### 4.2 EngineCore

EngineCore 持有：

- FIFO `pending_`；
- 固定大小的 active `slots_`；
- Scheduler 与 ResourceManager；
- 一个 Engine worker thread；
- outstanding、runtime stats 和 response event queues；
- Program 的稳定引用。

EngineCore 是 orchestration owner。它调用 Scheduler 取得 boundary decision，调用 ResourceManager 完成逻辑
资源转换，再调用 Program 执行 physical work。它不能绕过 ResourceManager 直接以 raw lane 修改 Program-owned
physical state。

### 4.3 Scheduler

Scheduler 拥有请求顺序与时间 policy：

- 观察 FIFO snapshot 和当前 head identity；
- 建立、推进和清除 protected-head epoch；
- 发出 head/backfill `AdmissionGrant`；
- 维护唯一 staged-prefill lane；
- 在每个 boundary 选择 admission、prefill 或 decode；
- 从 decode-ready slots 构造 frozen `RoundMembership`；
- 维护 projected service-work accounting。

Scheduler 不持有 AdmissionPlan、ContinuationHandle 或 eviction list。它只看到请求 identity、资源摘要、
reuse 后的 service work 和当前 active projection。

### 4.4 ResourceManager

ResourceManager 拥有 logical resource policy：

- active lane ownership与private/shared continuation catalogs；
- Device/Host State、Main/Backend KV的aggregate unique-occupancy ledger；
- continuation identity、revision、Session/Shared/Sparse indices与retention observations；
- root/private/shared candidate、pressure、replica-transition与capture选择；
- Program terminal acknowledgement和resource delta校验。

ResourceManager 不保存 physical epoch，不解释 KV page mapping，也不运行模型。Physical epoch 只由 Program
铸造和验证。

### 4.5 Program

Program 是 physical execution owner。它维护固定数量的active control lanes、StateImage/Device KV/Host KV
containers、execution metadata和唯一context transaction。AdmissionPlan由同一Program的inspect path产生并
绑定source/destination capability revision；SequenceHandle、ContinuationHandle、SharedPrefixHandle和
PendingBatch携带显式owner/epoch或transaction identity。Capability owner、epoch或transaction不匹配属于内部
invariant failure，而不是正常cache miss。

Program 同时是 Runtime transaction 的 mutation owner。`PendingBatch` 不携带反向可调用的 Program 指针；
只有 `Program::commit` 或 `Program::abort_pending` 可以消费它。

---

## 5. 四类生命周期

请求生命周期不是一个从 HTTP 到 CUDA 的大 enum。当前系统有四类相互关联、但归还条件不同的生命周期。

### 5.1 Gateway request lifetime

HTTP serving 在 media acquisition 和 prompt preparation 之前取得独立 `RequestLifetime` permit。它覆盖：

```text
request accepted
  -> media acquisition
  -> Frontend preparation
  -> Engine submission/wait
  -> protocol response processing
  -> Gateway lifetime object destruction
```

该 permit 控制 serving 端整体工作量，不等同于 Engine outstanding capacity。CLI 或直接 C++ caller 不需要
复制这个 serving-specific lifetime。

### 5.2 Engine model request lifecycle

已提交到EngineCore的`RequestRecord`使用以下模型状态；`capture_pending`是叠加在已发布Active状态上的
transaction flag：

```text
WAITING -> MATERIALIZING -> PREFILL -> DECODE_READY -> MODEL_FINISHED
```

| 状态 | Runtime binding | 允许的下一步 |
|---|---|---|
| `WAITING` | 无 lane、SequenceHandle 或 active entitlement | plan/inspect、取消、超时、admit |
| `MATERIALIZING` | destination reservation和context transaction；尚无published active slot | progress/abort materialization |
| `PREFILL` | 已安装 lane、SequenceHandle、budget 和 active entitlement | advance prefill或active capture |
| `DECODE_READY` | 同一 active binding | 加入下一次 compact decode batch |
| `MODEL_FINISHED` | 无 active sequence；可能已发布独立catalog continuation | response completion/consumer drain |

只有materialization terminal acknowledgement被ResourceManager采用后，Engine才安装sequence、budget和
active entitlement，并按返回的PrefillProgress进入`PREFILL`或完成finalization。Program已经reserve但尚未
publish期间保持`MATERIALIZING`，不会出现“物理destination已预留、逻辑却伪装成Active”的中间状态。

Successful terminal row 的顺序是 `commit -> finish -> MODEL_FINISHED`。取消和错误使用各自明确的 Runtime
release 路径后才进入 `MODEL_FINISHED`。

### 5.3 Engine response lifetime

Engine response completion 由两个独立 latch 表达：

```text
response_done       worker 已产生最终 result 或 error
consumer_released   wait() 已结束，或未消费 GenerationHandle 已被销毁
```

二者可以任意先后。只有第一次观察到两者同时为真时，`capacity_released` 才被置位并 exactly once 归还
Engine outstanding capacity：

```text
release_capacity := response_done && consumer_released && !capacity_released
```

这使以下路径都闭合：

- worker 先完成，consumer 随后读取结果；
- consumer/sink 先放弃，worker 在下一个 boundary 取消请求；
- 未调用 `wait()` 的 GenerationHandle 直接析构；
- error/fail-all 与 consumer release 并发到达。

`MODEL_FINISHED` 不等于 consumer 已释放。反过来，consumer 放弃也不表示 Runtime 已经安全释放。

请求的 `requested_output_tokens==0` 是公共 Engine 的非执行旁路：它直接返回 immediate result，不创建
RequestRecord、不进入 pending queue，也不占用 Engine outstanding permit。

### 5.4 Active lane 与 continuation lifecycle

Active lane与catalog descriptor是两个独立维度。每个lane只处于`Free | Materializing | Active`；private
continuation descriptor从root admission开始随lineage保留，active时不同时构造第二个catalog object。Shared
prefix有独立的定容catalog，允许多个active owners引用同一immutable full-page prefix。

```text
Waiting
  -> Materializing(destination reservation + exact source/pressure claims)
  -> Active(SequenceHandle + completion entitlement)
  -> Catalogued private endpoint | Released
```

Materialization、active capture和inactive replica transition共享Program唯一context transaction，但各自具有
独立terminal acknowledgement。一个private source可以被destructive Move消费，也可以在Fork时继续
catalogued；shared source只能Fork。Catalogued continuation已脱离RequestRecord和active slot，其State/KV
replicas可独立位于Device、Host或Both，不绑定原active lane。

### 5.5 Runtime local transaction

Prefill finalization 和 decode 都可能产生一个 move-only `PendingBatch`：

```text
Produced -> commit(PendingBatch&&)
         -> abort_pending(PendingBatch&&)
```

Program 同一时刻只有一个 unresolved pending transaction。PendingBatch 中的 row membership 和 token spans
引用 Program-owned boundary storage，只在 token 被 commit/abort 消费前有效。析构 PendingBatch 不发起
CUDA 或 Program mutation；Engine 的控制流负责线性消费。

`commit` 入口接管 token 后，caller 不再调用 `abort_pending`。若 commit 抛出，Program 消费 transaction、
释放 frozen members 并使相应 sequence capability 失效；Engine 根据冻结的 entitlement 清理 logical ledger，
随后进入 Engine-wide failure。

---

## 6. Admission 与逻辑资源

### 6.1 Bounded ingress

`Engine::submit` 同步建立 queue membership。Engine 同时限制：

- `max_concurrency`：active lanes；
- `max_pending_requests` 所导出的 bounded outstanding capacity；
- `pending_timeout`：未完成 admission 的 deadline。

队列满时 submit 直接返回 `Overloaded`，不建立一个无界等待对象。已经取得 outstanding permit 后，只有
第 5.3 节的双 latch handshake 才归还它。

### 6.2 完整生命周期 entitlement

ResourceVector按层级和resource class逐维记账：

```text
Device = (active_lanes, state_slots, main_kv_pages, backend_kv_pages)
Host   = (state_slots, kv_bytes)
```

`active_entitlement`表示请求从publication到terminal的完整completion guarantee，而不是下一次GPU unit的
增量。它覆盖active lane、可写State destination、最终Main/Backend KV范围、selected speculative backend和
page rounding。请求一旦published Active，正常执行不再因另一个请求增长而失去completion capacity。

Demand同时描述reserve peak和terminal ownership delta：

```text
reservation_credit / reservation_added / physical_peak_additional
final_removed / final_added / active_entitlement
```

Ledger按unique physical ownership聚合；共享page/state被多个catalog或active reference持有时只计一次。
Program在reserve前给出完整peak，在terminal acknowledgement中返回精确delta；ResourceManager只在
acknowledgement通过后发布logical ownership。

### 6.3 Candidate inspection

Scheduler先选定当前有资格admit的request；ResourceManager随后建立有界shortlist：

- SessionIndex指向的private endpoint/rewrite；
- marker与SharedPrefixIndex命中的shared stable prefixes；
- matching private long anchors与bounded anonymous private candidates；
- 永久存在的root/no-capture fallback。

Program对每个候选执行target exact verification，并返回opaque AdmissionPlan、资源demand、reuse path/frontier、
remaining service work和typed transfer requirements。Session key、hash、marker或KV token match都不能替代
verification。Inspection不修改Program、catalog或ledger。

### 6.4 Candidate selection

ResourceManager使用实测prefill/transfer成本与有界retention observation比较候选，并在资源不足时组合Program
提供的preserving degradation或eviction pressure option。比较对象包含完整reservation peak和publication后
unique occupancy，不用“释放后最终净值”代替物理可行性。

Choice固定source/victim IDs与revisions、pressure effects、destination、demand、projection和cost observation。
Program在reserve前重新验证全部capabilities。无候选当前可行时返回temporary blocked；root entitlement本身
永久不可行才是配置/内部错误，不把资源不足伪装成cache miss。

### 6.5 Materialization 与 ledger adoption

Accepted choice的mutation顺序是：

```text
revalidate exact source/victims and reserve demand
  -> Program reserve_materialization
  -> progress zero or more transfer/physical steps
  -> Program terminal acknowledgement
  -> ResourceManager atomically adopts source/victim/resource deltas
  -> publish Active SequenceHandle
  -> Engine installs RequestRecord binding
  -> Scheduler commits AdmissionGrant
```

NeedsTransfer request在此期间处于`Materializing`，不占active slot的published ownership；transfer完成后才能
成为Prefill或DecodeReady。Ready candidate也经过同一transaction contract，只是可以在一个boundary完成。
Terminal acknowledgement adoption前不能Finalize，adoption后恰好Finalize一次。Mutation开始后的Program异常
不是可重新选择另一个候选的cache miss；worker进入全局cleanup。

---

## 7. Scheduler 与 GPU boundary

### 7.1 Scheduling unit

Scheduler 调度的是完整 GPU execution unit，而不是 kernel：

- 一个materialization、active-capture或replica-transition progress unit；
- 一个 staged prefill chunk；
- 一个包含全部 decode-ready rows 的 decode round。

一个 unit 内 membership、page mapping 和 physical state selector 冻结。Admission、cancellation、slot recycle、
continuation claim/eviction 和 output commit 都发生在 unit 之间的 boundary。

### 7.2 Boundary order

worker 在每个 boundary 执行：

```text
expire/cancel waiting requests
  -> progress/finalize the unique context transaction when present
  -> freeze active cancellation snapshot
  -> abort boundary-cancelled active requests
  -> build compact decode membership
  -> consume one pending admission check when its fairness gate is open
  -> choose prefill / decode / replica transition / wait
  -> run at most one GPU unit
  -> commit Runtime and output state
  -> publish stats/events
```

只有 worker 修改 RequestRecord 模型状态、Scheduler、ResourceManager 和 Program。`execution_mutex` 同时串行
Program introspection 与 fail-all cleanup。

### 7.3 Prefill 与 decode policy

当前 policy 同时满足：

- 最多一个 request 拥有 staged prefill；
- prefill 和 decode 都存在时，不连续用 prefill 饿死 decode；
- pending admission check 与 decode 都存在时，只在decode之后给该check一次公平机会；
- materialization publication后才把request安装为active prefill/decode owner；
- active capture在自己的transaction完成前阻止该request越过记录的post-capture phase；
- admission 暂时不可行时，已有decode membership继续执行，且不在后续decode boundary重复inspection。

EngineCore用一个合并的invalidation latch记录尚未处理的admission check。Waiting queue变化、active
entitlement释放、prefill gate重新打开以及context transaction terminal会置位；普通decode commit、未完成的
prefill chunk、输出和统计发布不会置位。`TemporarilyBlocked`消费当前check后保持静默，直到新的相关事件发生；
任何accepted choice仍在同一worker boundary立即reserve，不跨boundary缓存未pin的choice。

当 `RoundMembership` 建立后，它包含所有且仅包含当前 decode-ready slots。Program 收到 compact
`SequenceHandle[B]` 和 `RoundBudget[B]`；inactive lanes 不以 padding row 加入本轮。

### 7.4 FIFO head protection

FIFO head 在root entitlement本身可行、但因当前active incumbents暂时无法admission时进入protection epoch。
Scheduler 冻结：

- head 的 cold/base entitlement；
- 当前 active incumbents 的 identity、resources 和 remaining service work；
- 按 projected completion 排序后，使 head 首次可行的 donor prefix；
- 最后一个 donor 的 projected distance，作为 temporal credit。

Protection 只解决“允许哪些 later request 暂时 backfill”，不改变 ResourceManager 对候选 physical lane 的
选择。

Backfill 分两类：

- **Persistent-safe**：即使该 request 持续占用资源，冻结 donor 释放后 head 仍逐维可行；
- **Temporal**：request 的 candidate-specific service work 不超过当前 donor frontier distance 和剩余
  temporal credit，预计会在 head 需要资源前完成。

当 head 在忽略 temporal borrowers 后已经可行，epoch 转入 Drain，不再接纳新的 backfill；已有 temporal
borrowers 完成后 head 获得 admission。FIFO head 变化会清除旧 protection，避免把过期 projection 应用到
另一个请求。

### 7.5 Service work

Service work 是 Scheduler 的完成距离单位：

- 每个materialized start/prefill unit消耗一个quantum；
- decode 消耗该 row 本轮实际 commit 的非取消 token 数；
- prefix reuse 通过 candidate summary 减少剩余 work；
- speculative round 的 produced width 不直接等于 service work，accepted token 数才推进请求。

它是小并发 backfill 的确定性 projection，不是通用延迟预测器或 GPU 时间模型。

---

## 8. Runtime 与输出事务

### 8.1 Planning 与 prefill

Program contract 的请求路径为：

```text
plan_request
  -> inspect_admission
  -> compose/revalidate/reserve/progress materialization
  -> advance_prefill (zero or more)
  -> optional active capture transaction
  -> decode (zero or more)
  -> commit / abort_pending
  -> finish / abort / replica transition / release
```

`plan_request`产生与lane无关的BasePlan和root demand。`inspect_admission`将它与destination、optional
private/shared source及checkpoint facts合成move-only AdmissionPlan；ResourceManager可再组合pressure effects，
但不能改写Program给出的source ownership或terminal delta。

Materialization publication产生SequenceHandle和首个PrefillProgress；后续每次advance prefill运行一个GPU
unit。未完成progress保持`PREFILL`；完成progress携带单row PendingBatch，走与decode相同的output
transaction。Exact prefix hit可以省略suffix prefill，仍通过finalization unit产生正确anchor和pending
output。

### 8.2 Decode 与 PendingBatch

Decode round 的 membership 在调用 Program 前冻结。Program 对 ordinary、MTP 或 DFlash 运行一次完整
whole-model unit，并返回 row-aligned provisional tokens：

```text
PendingBatch
  rows[B]        frozen SequenceHandle membership
  tokens         Program-owned ragged token view
  row_counts[B]  每行 licensed extent（ordinary 可以省略为 1）
  row_stride      physical row stride
```

Produced token 不是 committed output。MTP/DFlash 可以产生多个 proposal，但每行仍独立决定 accepted prefix；
Program 在 commit 前保持能将全部持久状态折叠到该 prefix 的 boundary state。

### 8.3 Preview 与 commit 顺序

每个 PendingBatch 使用以下顺序：

```text
freeze cancellation snapshot
  -> OutputSession preview_model / preview_terminal for every row
  -> stage accepted token IDs in RequestRecord
  -> Program::commit(PendingBatch, decisions)
  -> ResourceManager applies released-row ledger transitions
  -> finish every successful terminal row
  -> commit GenerationBudget and service work
  -> OutputSession::commit_preview
  -> append/publish per-request output events
  -> complete terminal responses
```

所有 Runtime commit、terminal finish 和 logical resource transitions 在本批第一次 output publication 之前
完成。因此 consumer 不会看到只写入 provisional KV/model state、尚未提交的 token。Publication 本身按请求
event queue 独立进行，不把多个 consumer 合并成一个 transport transaction。

OutputSession preview 可以更新自己的 staged decoder state，但在 `commit_preview` 前不发布。Preview 失败时，
Engine 回滚 staged token IDs，调用 `abort_pending` 释放整个 frozen membership，然后进入 Engine-wide
failure；不会逐 row 对 unresolved PendingBatch 调用普通 abort。

### 8.4 Commit decision 与 disposition

非取消 row 的合法 decision 满足：

```text
1 <= accepted_tokens <= produced_tokens
nonterminal => accepted_tokens == produced_tokens
terminal    => 可以接受 produced prefix
```

取消 snapshot row 使用：

```text
accepted_tokens = 0
terminal        = true
cancelled       = true
```

Program 为每行返回一个 disposition：

| disposition | Runtime 结果 | Engine 后续动作 |
|---|---|---|
| `Active` | 同一 SequenceHandle 继续 active | 保持 `DECODE_READY` |
| `Finishable` | accepted state 已提交，handle 只可 finish | 调用 `finish` |
| `CancelledReleased` | provisional rollback，sequence 已释放/失效 | ledger `Active -> Free`，不调用 finish |

MTP/DFlash acceptance、KV frontier、Linear Attention state、backend state、RNG/anchor 和 checkpoint 都在
Program commit 中推进到同一个 accepted frontier。Engine 不分别提交这些 target states。

### 8.5 Thinking cap 与 target control transaction

`ExecutionOptions::thinking.budget` 是 accepted model-origin thinking token 的可选正数上限。
cap 激活后，Frontend 的语义 tracker 不依赖 raw/reasoning presentation mode：它逐个解析已接受模型
token 的完整字节，自然观察 `</think>`，并只在 thinking 仍打开且计数准确到达上限时让
`OutputDecision` 返回 `ApplyTargetControl`。未配置 cap 时该 tracker 保持 dormant，不增加默认 decode
路径的逐 token detokenization。Scheduler 在 tracker armed 时把模型 round license 限制为
`min(total_remaining, budget-used)`，因此 MTP/DFlash 不会跨过该边界提交 speculative extent。

自然 close、stop token/string、总 output/context limit 和 cancellation 都优先于 control。达到边界的
nonterminal request 进入 `CONTROL_READY`，不再属于普通 decode membership。Prompt-finalization 的第一个
token 也可能到达边界；若它同时产生 active-capture offer，capture transaction 的 post state 必须保存
`CONTROL_READY`。

Qwen Frontend 在资源构造时把规范 control suffix 编码一次，不向 Engine 暴露固定 token ID 或固定 token
数。当前 suffix 与 [Qwen 官方 thinking-budget 机制](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md)
一致，固定为：

```text
\n\n Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n
```

引导语位于 close marker 之前，因此属于 reasoning；`</think>` 只切换输出 phase，不是 stop token。Control
membership 中每行携带 tokenizer 动态编码得到的完整 span，事务顺序为：

```text
freeze cancellation snapshot
  -> OutputSession::preview_control(exact span)
  -> stage every control token ID in RequestRecord
  -> Program::append_forced_tokens(compact membership)
  -> commit GenerationBudget and service work
  -> OutputSession::commit_preview
  -> publish reasoning deltas
  -> CONTROL_READY -> DECODE_READY
```

`preview_control` 忽略 caller stop policy 的截断作用，但仍通过 presentation decoder 完成 reasoning close；
Program 成功前没有 control output 可见。Control span 消耗 generated-token、context 和 service-work 容量，
但不调用 sampler、不推进 RNG 或 sampling occurrence counter。

Program 把一行 control 前的 sequence invariant
`S=E+1, ledger.back()=unexecuted anchor` 推进为短 no-sample continuation。对 forced tokens
`F[0..K-1]`，模型执行 `ledger.back(), F[0], ..., F[K-2]`，并把完整 F 追加到 ledger。提交后
`E'=E+K, S'=E'+1`，最后的 `F[K-1]` 成为下一普通 decode 的 anchor。同一 family 算法更新 Main KV、GDN
state/fork、tail hidden、position/rope delta、prefix identity、MTP KV 与 draft reset，以及 DFlash
feature/context frontier。控制预算值本身不进入 prefix identity；已提交的 forced token IDs 进入。

若有效输出容量 `M` 大于 budget `B`，planning 要求 `M-B >= K+1`；额外一个槽位只提供一次 post-close
模型执行机会，不保证可见正文或工具调用。`M<=B` 时普通 output/context limit 可以先结束请求。

### 8.6 Finish 与 retention

允许reuse的成功请求在finish时把active mutable state冻结为private endpoint；root-only或request-level
reuse关闭时直接释放。顺序是：

```text
Program::finish(sequence)
  -> return Released | Catalogued + continuation summary/handle + exact resource delta
  -> ResourceManager validates terminal delta and checkpoint completeness
  -> reclassify active private descriptor as catalogued, or release it
  -> atomically publish/replace optional SessionIndex endpoint
```

Private descriptor在admission时已经保留，finish不临时搜索catalog slot。Typed rewrite、long anchor和shared
stable prefix由更早的committed boundary通过ActiveCapture发布；它们与endpoint一起组成完整checkpoint
summary。一个checkpoint只有State、Main/Backend KV、hidden/position和selected speculative state全部对应同一
frontier时才可发布。

`finish`、`abort`、`release_continuation`、`release_shared_prefix`和`abort_pending`是consuming operations，
并通过`ConsumeStatus`与resource delta区分成功和owner/epoch mismatch。Mismatch不伪装成空资源成功；Engine
将其提升为invariant failure，由Program teardown cleanup兜底。

---

## 9. Cancellation、错误与清理

### 9.1 Cancellation boundary

Engine 观察到 cancellation flag 后使用 boundary 语义：已经提交的 GPU unit 先完成，再在下一 boundary
选择对应路径。外部 `CancellationView` 由 consumer wait loop 轮询并转换成该 flag，因此“外部取消发出”与
“Engine 已观察取消”不是同一个时刻。

| 被观察时的状态 | Runtime 动作 | Output 动作 |
|---|---|---|
| `WAITING` | 无 Runtime state | staged `preview_terminal(Cancelled)` 后完成 |
| active，尚未产生本轮 PendingBatch | `abort(sequence)` | release 成功后 commit/publish terminal preview |
| `CONTROL_READY`，control transaction 尚未开始 | `abort(sequence)` | 不插入 control；release 后 commit/publish terminal preview |
| 已属于 frozen PendingBatch | `commit` 为 `CancelledReleased` | ledger release 后 commit/publish terminal preview |

同一请求的 terminal preview 恰好执行一次。若 cancellation 在本轮 snapshot 之后到达，而 row 已被正常
terminal decision 结束，保留原 finish reason；若 row 仍 nonterminal，则在下一 boundary abort。

GenerationHandle abandonment 设置 cancellation，并独立完成 `consumer_released` latch；它不从 consumer
线程直接调用 Program。

### 9.2 Request-local errors

在 Runtime mutation 前可归属于单个 request 的错误只结束该 request，例如：

- prompt/context 超过公开或 shared-capacity contract；
- queue timeout；
- waiting cancellation；
- request planning 对 represented input 的拒绝；
- submit/outstanding overload。

这类错误不清空其他 active requests。

### 9.3 Engine-wide failures

以下故障表示 Engine/Runtime state 已无法安全继续：

- Program 在 GPU unit 或 mutation 开始后抛出；
- PendingBatch layout、row disposition 或 resource acknowledgement 不一致；
- physical capability owner/epoch mismatch；
- Output preview 已持有 unresolved PendingBatch 时出现无法局部恢复的错误；
- finish、commit 或 logical ledger invariant 被破坏。

worker 在 `execution_mutex` 所有权内执行 fail-all：

```text
mark Engine failed and detach pending queue
  -> reset Scheduler
  -> Program::fail_all_cleanup
  -> clear ResourceManager catalog/ledger
  -> complete every active and waiting response with error
```

Engine failure 后新的 submit 返回 unavailable。Cleanup 的目的不是回滚业务 policy，而是使所有 physical
state、capability、response latch 和 capacity permit 到达可终止状态。

---

## 10. Physical execution 与性能结构

### 10.1 Fixed lanes 与 compact rows

Program 在 startup 建立 `max_concurrency` 个 physical control lanes。Lane 是 persistent state 与 stable
metadata 的 home；compact batch row 是一次 decode round 的临时 ordinal。二者通过 row selectors 映射：

```text
compact row b -> SequenceHandle -> physical lane -> block-table row/state selectors
```

因此，active lanes可以稀疏，GPU launch仍使用精确`B`的compact tensors。Catalogued continuation不占
active control lane；materialization把选中的Device-ready或Host-backed checkpoint安装到一个free lane的稳定
execution metadata中。

### 10.2 Workspace 与 request transient

Program 拥有两类复用内存：

- whole-model/shared workspace：服务 prefill/decode schedule 和 Ops；
- request transient arena：服务单请求 preparation-to-prefill bridge，尤其 Vision 临时张量。

它们的 backing 在 startup 分配，Program 在明确的 request phase activate/deactivate。Engine 不持有 device
arena，也不向 Program 传 raw transient region。一个 staged prefill owner 与一个 Program mutation owner
使该 arena 不需要变成 per-request allocator。

### 10.3 Paged KV

Growing Main/backend KV 使用 shared page pools 和启动期固定 block-table matrices。ResourceManager 只按
page-rounded entitlement 记账；Program 与 KV store 维护 allocation mapping 和 valid/provisional frontier。

一次 GPU unit 内 mapping 稳定，boundary 才能 materialize、commit、truncate、claim 或 release。Prefix reuse
需要完整 continuation state；KV token/page match 本身不足以证明可恢复性。详细 contract 见
[Paged KV Context Store](paged-kv-cache.md)。

### 10.4 Exact-B CUDA Graph

Program 为当前 backend 的每个合法 compact batch size `B=1..max_concurrency` 建立 decode graph families，
并按 target frontier/topology profile 选择 replay。Graph key 描述执行 topology，不包含 request identity、
physical page IDs 或 active lane 集合。

Graph capture 使用 representative fixed-address storage；运行期变化通过 stable buffers、row selectors、
frontiers 和 block tables 更新。Scheduler 的 maximal membership 与 Runtime 的 exact-B graph 共同避免 inactive
row padding 和 per-request decode launches。

### 10.5 Speculative backends

MTP 与 DFlash 是 Program 内的 closed execution schedules，不是独立 Engine。Scheduler 对 ordinary/MTP/DFlash
都只看到一个 decode unit、candidate service work 和最终 accepted tokens。

Speculative backend 可以扩大每行 provisional width，但不会改变以下边界：

- batch membership 在 round 前冻结；
- 每行 output policy 独立；
- Program 一次 commit 全部 target state；
- 只有 accepted tokens 推进 GenerationBudget 和 service work；
- partial terminal prefix 与最终 committed/published continuation frontier 一致。

---

## 11. End-to-end 请求路径

一个普通非零输出请求的完整路径为：

```text
Gateway/product adapter
  -> Engine::prepare(PromptInput)
  -> Frontend produces PreparedPrompt
  -> Engine::submit reserves outstanding capacity
  -> RequestRecord enters FIFO WAITING
  -> Program plan + ResourceManager inspect
  -> Scheduler grants admission
  -> ResourceManager claim/evict/start
  -> RequestRecord adopts Active sequence as PREFILL
  -> zero or more prefill units
  -> finalization PendingBatch preview/commit
  -> DECODE_READY
  -> repeated maximal compact decode + commit
  -> optional CONTROL_READY + deterministic no-sample target control + DECODE_READY
  -> terminal Program finish
  -> optional catalogued private endpoint and shared/checkpoint references
  -> OutputSession commit and response_done
  -> GenerationHandle consumer release
  -> exactly-once outstanding capacity release
```

Prefix reuse 只改变 admission candidate、prefill work 和最终 continuation path，不创建另一条 request
lifecycle。MTP/DFlash 只改变 Program 的 unit 内部和 PendingBatch row width，不创建第二套 Scheduler 或
publication path。

---

## 12. 当前扩展边界

### 12.1 Host backing 与 replica transition

Inactive private/shared checkpoints可以分别持有Device、Host或Both replicas。ResourceManager选择logical
residency、成本与pressure intent；Program拥有State/KV transfer correctness、physical layouts、events和
terminal acknowledgement。Host State按完整image计费，Host KV按实际extent bytes计费；active request不被
swap，只有inactive ownership参加background-free replica transition。

### 12.2 Fork、COW 与 shared prefix

Private continuation保持single-claim，但可以从immutable source Fork到新的active StateImage；shared prefix只
能Fork。Main/Backend KV的immutable full pages可以跨catalog和active owners共享，非page-aligned branch对部分
tail执行private COW。ResourceManager ledger按unique occupancy计费，Program阻止任一writer覆盖surviving
checkpoint保护的prefix。Linear Attention、hidden和selected backend StateImage仍按完整image Fork，不能用KV
page refcount替代完整continuation state。

### 12.3 保持边界清晰

新增能力时使用以下判断：

- 改变请求优先级或 GPU unit 顺序：Scheduler；
- 改变 logical residency、entitlement 或 candidate value：ResourceManager；
- 改变模型状态、KV、copy/transfer 或 GPU 执行：Program；
- 改变 stop/detokenization/输出 channel：Frontend；
- 改变协议、连接或 response schema：Gateway。

一个功能跨越多个边界时，应以显式 summary/choice/capability/transaction 连接，而不是让其中一个组件读取
另一个组件的私有状态。

---

## 13. 代码映射与相关权威

### 13.1 当前代码映射

| 架构职责 | 主要实现位置 |
|---|---|
| 公共 Engine facade 与 zero-output bypass | `include/ninfer/engine.h`, `src/runtime/engine/engine.cpp` |
| RequestRecord 与 response latch | `src/runtime/engine/request_record.h`, `engine_core.h` |
| Scheduler | `src/runtime/engine/scheduler.h`, `admission_policy.*` |
| ResourceManager 与 ledger/catalog | `src/runtime/engine/resource_manager.h` |
| package-neutral contracts | `src/runtime/contract/types.h` |
| exact package registry/startup | `src/targets/registry.*`, `src/targets/<package>/` |
| Qwen3.6 Frontend | `src/targets/qwen3_6/impl/frontend/` |
| Qwen3.6 Program family Runtime | `src/targets/qwen3_6/impl/runtime/` |
| physical primitives 与 arenas | `src/core/` |
| semantic CUDA Ops | `src/ops/`, `include/ninfer/ops/` |
| HTTP Gateway | `src/serve/` |
| shared product input adapter | `src/product/prompt_input/` |

路径表用于定位当前 authority，不把文件拆分本身当作外部 contract。重构可以改变局部文件组织，但不能在
没有同步更新本文的情况下改变第 4 节的所有权。

### 13.2 相关文档

- [Paged KV Context Store](paged-kv-cache.md)：page ownership、capacity、physical layout、frontier 和
  consumer contract；
- [Op development](op-development.md)：Op 语义所有权、正确性和性能准入；
- [Qwen3.6-27B model](qwen3.6-27b-model.md) 与
  [Qwen3.6-35B-A3B model](qwen3.6-35b-a3b-model.md)：模型数学与持久状态语义；
- [Artifact container](artifact-container.md)、[storage layouts](storage-layouts.md) 与
  [tensor formats](tensor-formats.md)：`.ninfer` 持久格式；
- [CLI](../cli.md) 与 [HTTP serving](../serving.md)：用户行为和外部协议；
- [Performance](../performance.md)：公开测量方法和结果。
