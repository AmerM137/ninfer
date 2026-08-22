# Device-only Private Continuation 纵向闭环实施计划

**文档状态：** Step 3 实施完成
**实施范围：** 全局 private continuation catalog、Device state/KV 逻辑存储、资源事务与执行闭环
**相关架构：** [Resource scheduling and context cache](resource-scheduling-and-context-cache.md)
**实施记录：** [Resource Scheduling 与 Context Cache 实施记录](resource-scheduling-and-context-cache-implementation-record.md)

## 1. 文档用途

本文是 Resource Scheduling 与 Context Cache 落地工作的 Step 3。它定义第一个可独立运行的
纵向闭环：一个请求从 Scheduler 获得 admission 资格后，ResourceManager 可以从全局 private
continuation catalog 选择 endpoint、typed rewrite 或 root，Program 将所选逻辑资源
materialize 为任意空闲 lane 上的 active sequence，执行完成后再发布新的 private
continuation。

本阶段同时切换逻辑 ownership、物理 binding 和资源账本，不保留“新 catalog + 旧 lane-resident
Program”或“新逻辑 store + 旧 `PrivateKVMapping`”的混合生产路径。完成后，现有 Text、Vision、
MTP、DFlash 和 exclusive prefix reuse 仍通过同一个公共 Engine 路径运行。

[正式架构文档](resource-scheduling-and-context-cache.md)继续是最终目标 authority。本文只固定
本阶段的实现边界、切换顺序和验证证据，不修改 Host、shared prefix 等后续目标设计。

## 2. 本阶段完成定义

Step 3 完成时，以下路径形成一条闭合的生产链路：

```text
Scheduler selects one waiting request
    -> ResourceManager enumerates global private checkpoints and root
    -> Program performs exact target inspection
    -> ResourceManager selects source, destination lane and whole-continuation victims
    -> Program materializes a Device-only active sequence transactionally
    -> RequestRecord adopts SequenceHandle
    -> ResourceManager adopts the exact resource delta
    -> existing prefill / compact decode / speculative commit executes
    -> terminal committed state is published as a new private continuation
```

具体交付为：

1. 用全局、固定容量的 private continuation catalog 取代按 lane 固定的 resident catalog；
2. lane 只保留 active execution/control 语义，continuation 可以在任意空闲 lane 恢复；
3. 将 Device state slot 纳入完整 request-lifetime 资源向量和统一账本；
4. 在 Program 内新增 `StateImageStore`、typed `LogicalKVPageStore` 和
   `KVAddressSpaceStore`；
5. 删除固定 `current(lane)/rewrite(lane)` state authority，改为动态 StateImage binding；
6. 删除 `PrivateKVMapping`，由 logical page/address-space objects 持有 ordered mapping、
   Device replica 和 growth reservation；
7. 建立 root、private endpoint、private typed rewrite 的 exact inspection 和 readiness；
8. 建立 `ResourcePlan -> MaterializationTransaction -> SequenceHandle` 的线性消费路径；
9. 实现 Device-only `FullReset`、private `Move`、active `Freeze/Fork` 和 finish publication；
10. 保持现有 public Engine、CLI、OpenAI/Anthropic serving 与 artifact contract 不变。

生产切换完成后删除被替代的 `LogicalLaneState::Resident`、lane-indexed
`ContinuationHandle`、`StateImageSlots`、`PrivateKVMapping` 和 Program `Resident` lifecycle；
不提供 compatibility alias 或旧行为 fallback。

## 3. 明确不在范围内的内容

以下内容不属于 Step 3：

- Program 对 `HostStatePool` 或 `HostKVArena` 的创建、replica publication、offload 或 restore；
- `Snapshot`、`Restore`、D2H/H2D transaction、异步 transfer stream 或跨 boundary ticket；
- shared stable prefix、跨 address space logical-page sharing、tail COW 或 shared state Fork；
- SessionKey、caller stable markers、long anchors、SharedPrefixIndex 或新的产品输入字段；
- endpoint-only/rewrite-only 的部分降级、partial KV eviction 或 replica demotion；
- online cost learning、hit-frequency/recency policy、后台 retention worker 或主动整理；
- Scheduler 的 FIFO、protected-head、backfill、prefill-owner 和 compact-decode policy 重写；
- public `ContextCacheOptions`、Host 容量选项、CLI/serving 参数或 schema 修改；
- artifact、权重、tensor format、attention 数值公式或 KV codec 修改；
- benchmark 工程、吞吐提升声明或固定百分比性能 gate；
- 对正式架构文档或其他长期 current-behavior 文档的统一改写。

Step 1/2 已实现的 Host 物理容器和传输原语继续保留，但本阶段生产 Program 不持有 Host
replica。`NeedsTransfer` 作为 Scheduler/ResourceManager 的最终 readiness 语义被保留，本阶段
不会产生该结果。

## 4. 本阶段固定决策

### 4.1 Device state 容量

设 Engine 最大 active concurrency 为 `C`。本阶段固定：

```text
H = C
Device StateImage slots = C + H = 2C
```

这与当前 `2C` StateImage payload bytes 相同，不增加 state 显存；变化的是 ownership：全部
slots 进入一个同构动态池，不再永久分成每 lane 的 current/rewrite 两半。

其语义固定为：

```text
C slots: active-concurrency guarantee
H slots: full concurrency 时仍可被 active rewrite 或 global hot checkpoint 使用的容量
```

空闲 active capacity 可被 catalogued checkpoints 借用。由于 Step 3 没有 shared pin、Host
transfer 或跨 boundary materialization，boundary 上所有非 source 的 catalogued private
continuations 都可立即整体回收；admission 仍需保证 active requests 已取得的完整 entitlement
不可撤销。

本阶段不暴露 `H` 配置。以后将 `H` 改为启动选项只改变 capacity construction，不改变本阶段
建立的 store、handle、ledger 或 materialization contract。

### 4.2 Device 资源向量

现有只覆盖 lane/Main KV/Backend KV 的 `AdmissionResources` 被一个明确的 Device 资源向量
替代，概念形式固定为：

```cpp
struct DeviceResources {
    uint32_t active_lanes;
    uint32_t state_slots;
    uint32_t main_kv_pages;
    uint32_t backend_kv_pages;
};
```

其含义均为 unique physical occupancy 或已取得的排他 reservation：

- active entry 的 `active_lanes == 1`；
- catalog entry 的 `active_lanes == 0`；
- active `state_slots` 包含当前 mutable image、仍保留的 rewrite/source image，以及本请求
  后续 checkpoint rotation 已取得的 destination entitlement；
- active KV pages 包含 mapped Device replicas 与尚未 materialize 的 future-growth
  reservation；
- catalog KV pages 只包含该 continuation 独占的 mapped Device replicas；
- endpoint 与 rewrite 共用同一 address space，其 pages 在 entry footprint 中只计一次。

Program 返回已按真实物理粒度取整的向量。ResourceManager 和 Scheduler 不从 token 数重新推导
page 或 state 数量。Host state slots 与 Host KV bytes 等到 Host replica 进入生产路径时由独立
Host resource vector 加入 materialization demand，不在本阶段放置恒零字段。

总 Device capacity 固定为：

```text
(C active lanes, 2C state slots, Kmain pages, Kbackend pages)
```

所有加减使用扩大后的整数做 checked arithmetic；underflow、overflow 或 Program
acknowledgement 不一致属于 Engine invariant failure。

稳态 boundary 的统一 ledger 为：

```text
sum(active full-lifetime entitlements)
+ sum(catalog unique footprints)
+ synchronous transaction reservations
<= Device capacity
```

`ResourceDelta` 使用 `removed/added` 表达 ownership ledger transition：

```text
used_after = used_before - removed + added
```

Private Move 的 source footprint进入 `removed`、同一批physical resources所属的新active
entitlement进入 `added`；这表示唯一资源从catalog重分类到active，不表示先释放再重新分配
payload。Victim acknowledgement另行确认哪些physical resources确实被释放。

Candidate还携带Program给出的`ResourceDemand`，其中区分：

```text
active_entitlement       publish后的完整request-lifetime ownership
prepublish_additional    source仍完整时需要独立取得的transaction destinations/reservations
source_conversions       publish点可从source原子重分类或转成growth reservation的资源
```

`prepublish_additional`不包含Move复用的source slots/pages，也不能提前使用尚未提交的source
release；它与steady-state `ResourceDelta`是两个不同的容量事实。

### 4.3 固定控制面容量

当前产品满足 `C <= 8`。Step 3 使用以下启动时固定容量：

```text
ResourceManager catalog slots                  = 2C
Program continuation descriptor slots          = 2C
StateImage logical-object slots                 = 2C
Main KV address-space slots                     = 2C
Backend KV address-space slots, when enabled    = 2C
Logical Main page descriptors                   = Main Device pool page capacity
Logical Backend page descriptors, when enabled  = Backend Device pool page capacity
Main ordered-membership cells                   = 2C * Main per-sequence table capacity
Backend ordered-membership cells, when enabled  = 2C * Backend per-sequence table capacity
```

每个有效 private continuation 或 active lineage 都拥有一幅 endpoint/current StateImage，
optional rewrite再占一幅。因此 `2C` state capacity 同时给出了上述 lineage objects 的上界；
第二幅 image只会减少可同时存在的 continuation 数，不会要求更多 descriptor。

Catalog slot 具有：

```text
Vacant
Catalogued
Claimed                 // synchronous materialization 内部状态
ReservedForActive       // 预留给该 active lineage 的 finish publication
```

从 catalog source 恢复时复用其 catalog slot；从 root 启动时在第一次 physical mutation 前
预留一个 vacant slot。若存在 `2C` 个 `Catalogued` entries，它们各自独占一幅 endpoint image，
Device state pool已经没有free slot；root plan因而会整体逐出一个或多个entries，释放的其中一个
catalog slot作为publication reservation。由此terminal publication不再依赖host allocation或
临时扩容。

Candidate、victim、transaction acknowledgement 和 compact row metadata 使用固定数组。
Token ledger、target prefix identity 和 PreparedPrompt 的变长 owning buffers 可以在
admission 第一次 physical mutation 前完成 reserve/move；decode、commit、finish 和
publication 不扩展这些 buffers。

Program另有一个固定的materialization staging record，对应唯一同步transaction ticket。Root
启动时，它在victim descriptor尚未释放前暂存已经reserve完成的identity/ledger ownership；
ResourcePlan预先指定一个vacant descriptor，或指定一个将在同一transaction中整体消费的victim
descriptor作为destination。Staging record不会形成第二个published continuation，victim释放后
才把root ownership move进正式descriptor。Catalog publication slot与KV address-space slots使用
相同的conditional reservation规则，不要求在`2C`之外增加第`2C+1`个对象槽位。

### 4.4 本阶段 retention policy

本阶段保留现有产品行为：

- 正常成功请求发布一个 private continuation；
- cancellation、abort 或 Engine failure 不发布新 continuation；
- `allow_prefix_reuse=false` 只禁止该请求选择已有 continuation，不禁止其成功结果成为后续
  `allow_prefix_reuse=true` 请求的 source；
- catalog pressure 只整体逐出 inactive private continuation；
- 不在 finish 时主动降级或迁移其他 entry；
- 不实现 endpoint-only 或 rewrite-only victim action。

这使现有“cold source request 后由下一请求验证 reuse”的真实路径保持不变，同时把 victim
policy 限定在本阶段已经能够原子闭合的 ownership 单元。

## 5. Ownership 与对象模型切换

### 5.1 ResourceManager ownership

ResourceManager 持有：

- 固定容量 global private continuation catalog；
- `ContinuationId`、logical revision、catalog state 和 policy-visible summary；
- Free/Active lane ledger；
- catalog/active/transaction 的 Device resource ledger；
- candidate、whole-continuation victims 和 destination lane 的选择；
- Program-minted `ContinuationHandle` 的 custody；
- active lineage 对应的 reserved finish-publication catalog slot。

Catalog entry 不保存 raw state slot、physical page ID、block-table row、content epoch 或 target
prefix payload。Policy summary 只包含 checkpoint kinds/frontiers、unique Device footprint、
rebuild work，并与ResourceManager采用它时的catalog revision绑定；Program capability generation
仍只存在于opaque handle/activation中。

### 5.2 Program ownership

Program 持有：

- continuation descriptors，以及 target token ledger、exact prefix identity 和 backend
  frontier metadata；
- `StateImageStore`；
- Main/optional Backend `LogicalKVPageStore` 与 `KVAddressSpaceStore`；
- Device state/page replicas、growth reservations 和 execution rows；
- active lane 到 continuation/state/address-space 的 physical binding；
- state src/dst selectors、Fork resolution 和 CUDA Graph binding；
- checkpoint exact inspection、materialization、prefill/decode/commit 和 finish capture。

Program 是 physical occupancy、capability generation、KV content epoch/coverage 和 exact
resource delta 的唯一 authority。

### 5.3 Lane 与 continuation 分离

Program 的 lane storage 被拆成两部分：

```text
ActiveLane[C]
    request control / lifecycle / sampling / pending round
    SequenceHandle epoch
    ActiveContinuationBinding

ContinuationStore[2C]
    target identity / token ledger / frontier metadata
    endpoint + optional rewrite StateImage handles
    Main + optional Backend KV address-space handles
```

Catalogued continuation 不占 `ActiveLane`，也不持有 execution-table row。Materialization 选择
最低编号的空闲 lane 作为 destination；source continuation 的历史 lane 不再存在于 handle 或
candidate contract 中。

`SequenceHandle` 继续是 Program owner + active lane + lane epoch capability。
`ContinuationHandle` 改为 Program owner + continuation descriptor index + generation capability，
不含 lane。二者字段保持 private，不能由 ResourceManager 拼装。

### 5.4 Private continuation payload

本阶段一个 Program-owned private continuation 固定包含：

```text
ContinuationRecord
├── exact target identity and committed token ledger
├── Main KVAddressSpaceHandle
├── optional Backend KVAddressSpaceHandle
├── EndpointCheckpoint
└── optional typed RewriteCheckpoint
```

Checkpoint 只保存其 StateImage handle、typed KV requirement、frontier 和 kind。Endpoint 与
rewrite 引用同一 address spaces 的不同 committed prefixes。

支持的 checkpoint kinds 为现有：

```text
SessionEndpoint
TurnClosure
ResponseReplay
```

本阶段没有 SessionKey index；`SessionEndpoint` 表示 private terminal endpoint 的语义类型，
候选仍通过全局 catalog shortlist 后由 Program 做完整 target exact verification。

## 6. StateImageStore 与执行 binding

### 6.1 Logical StateImage identity

在 Qwen3.6 family runtime 内新增 Program-private `StateImageStore`。它位于
`StateImageDevicePool` 之上，使用 Step 1 的完整-image physical operations，但不把 raw slot
暴露给 ResourceManager。

每个 logical image object 记录：

```text
opaque StateImageHandle generation
one Device physical slot
Free | ActiveMutable | CheckpointImmutable | ReservedDestination role
source/destination pin state
```

Step 3 每幅有效 logical image 恰好有一个 Device replica；Host replica metadata尚未加入。
release 同时使 logical handle generation 和 physical slot generation 失效。StateImage handle
只在所属 Program lifetime 内有效。

Store 提供的语义操作为：

```text
reserve/reset mutable destination
Move checkpoint -> active mutable
Freeze active mutable -> checkpoint immutable
begin Fork(source checkpoint, reserved destination)
commit/abort Fork
release logical image
resolve validated physical src/dst selectors
```

这些操作不理解 request fairness、cache value 或 victim policy。

### 6.2 ActiveStateBinding

每个 active sequence 持有：

```cpp
struct ActiveStateBinding {
    StateImageHandle read;
    StateImageHandle write;
    bool fork_pending;
};
```

Program 在每个实际推进target state的execution invocation建立 per-row：

```text
state_src_slot[B]
state_dst_slot[B]
```

普通 prefill/decode 和 private Move 完成后：

```text
src == dst
```

active checkpoint capture 后的首个完整 state transition：

```text
src = immutable checkpoint
dst = reserved active destination
```

只有 Program commit确认 destination 已包含当前 Variant 的完整 committed state 后，才把
binding 收敛为 `src == dst` 并解除 source pin。Exact-hit finalization若没有形成完整新 state，
`fork_pending` 保持到后续实际 transition。

### 6.3 Qwen state consumer 切换

所有推进完整 state 的 production paths 使用同一 binding：

- Text prefill 的 convolution 与 recurrent state 接收独立 input/output slots；
- ordinary compact decode 的 GDN batch update接收 per-row src/dst selectors；
- MTP/DFlash target verify 的 record path从 src读取；
- accepted replay/fold把完整 accepted recurrent/conv state写入 dst；
- continuation hidden由 dst selector发布；
- rejected speculative suffix不推进 committed binding；
- CUDA Graph ingress携带 runtime selector values，不捕获固定 slot literal。

现有 scalar distinct-state convolution/GDN contract直接复用。Batched GDN update 和 replay
fold contract扩展为独立 src/dst；数值公式、represented inputs 和 committed result不变。

普通 rows 的 src/dst值相同，selector在 kernel inner recurrence之外解析。不能通过整幅
`StateImageDevicePool::copy_slot` 实现普通 decode 或 GDN Fork。

### 6.4 DFlash local fixed state

DFlash local cyclic K/V 不能由一次新 token transition完整重写。开始 active Fork 时，Program
只把该 fixed component 从 source image复制到 destination image；随后：

```text
GDN state       read source, write destination
continuation hidden        write destination
DFlash local K/V           read/write copied destination
```

这保持 DFlash window history正确，同时避免复制包含全部 GDN recurrent state 的整幅约
150 MiB image。Component copy在同一 Program stream中先于首个 destination unit，source pin
保持到该 unit commit。

### 6.5 Rewrite rotation

active request需要建立或替换 typed rewrite 时使用：

```text
current active S0 -> Freeze as new rewrite
old rewrite/free entitlement S1 -> new active destination
next committed transition: S0 -> S1
terminal: S1 -> new endpoint
```

若此前没有 rewrite，S1来自 admission已经取得的第二个 state-slot entitlement；若存在旧
rewrite，则在新 committed rewrite已经建立后复用旧 rewrite slot。过程中最多占用该 active
plan声明的两幅 images，不引入第三幅临时 image。

Rewrite frontier位于一个prefill unit内部时，Program在同一unit内按精确frontier划分state
transition subpasses：先把frontier hidden发布到S0并Freeze S0，再让剩余suffix从S0读、向S1写。
如果本unit没有frontier后的state transition，Fork保持pending到下一次prefill/decode；如果有，
该subpass完成并提交后立即收敛到S1。实现不能退回到`copy_slot(S0, S1)`保存整幅GDN state，
也不能为了ownership简化而把一个既有融合prefill unit暴露成新的Scheduler unit。

选择 endpoint恢复时，endpoint image使用 Move成为 active，仍匹配的 rewrite继续保留。选择
rewrite恢复时，旧 endpoint被 destructive consume，KV suffix被截断，rewrite image使用 Move
成为 active。

`StateImageSlots` 和 `current(lane)/rewrite(lane)` helper在生产 cutover后删除。Graph capture
使用临时 store reservations，capture完成并同步后归还，不绕过 allocator写固定 lane slots。

## 7. Logical KV page 与 address space

### 7.1 Store placement

Logical KV 层由 Qwen3.6 family Program拥有，不下沉为新的通用 core inference framework：

```text
Main LogicalKVPageStore + KVAddressSpaceStore
optional Backend LogicalKVPageStore + KVAddressSpaceStore
    -> Step 2 DeviceKVPagePool / KVExecutionTablePool
```

Core 继续只拥有物理 page、reservation、execution row 和 transfer primitives。Main 与 Backend
使用相同逻辑 contract，但保留独立 store、capacity、frontier 和 resource dimension。

### 7.2 Logical page object

每个 logical page object记录：

```text
opaque LogicalKVPageHandle + capability generation
canonical content epoch
committed coverage in page columns
one DeviceKVPageLease
address-space reference count
write-protected committed prefix
```

Step 3 没有 Host replica或跨 address-space sharing，因此有效 page的 reference count只会是
`1`；仍然由 store而非 continuation副本维护。正常 append不改变 content epoch，只在 Program
commit后推进 coverage。Speculative bytes即使已经写入 physical page，也不进入 canonical
coverage。

Private rewrite restore若在 partial tail内发生 destructive divergence：

1. Program先确认旧 endpoint已被 transaction consume，且没有其他 surviving checkpoint需要
   被覆盖的 suffix；
2. address space frontier回退到 rewrite requirement；
3. page committed coverage收缩，content epoch递增；
4. 同一 Device replica按新 epoch重新认证，旧 suffix bytes失去内容语义；
5. 后续 writer只在新的 committed frontier之后追加。

本阶段不创建 COW page。任何会要求第二个 writer或跨 address-space reference的操作均不进入
该实现。

### 7.3 KVAddressSpace

每个 address space object记录：

```text
opaque KVAddressSpaceHandle + generation
typed-pool identity
ordered LogicalKVPageHandle membership
logical/committed frontier
remaining Device growth reservation while active
optional KVExecutionRowLease while active
endpoint/rewrite protected requirements
```

Ordered membership使用 Program启动时固定 descriptor storage，不在 page materialization、
commit或finish时增长 host container。Address space从 catalog进入 active时才取得 destination
lane对应的 execution row并发布 block table；finish时先解除 row，再进入 catalog。Row不拥有
pages，catalogued continuation不占 row。

`PrivateKVMapping` 的三项职责被分别接管：

```text
ordered physical leases      -> LogicalKVPageStore replicas
remaining growth reservation -> active KVAddressSpace
execution row                -> active KVAddressSpace binding
```

### 7.4 Checkpoint validity 与 footprint

本阶段所有有效 checkpoints 均为 Device-only。Program只有在以下条件成立时返回 exact
candidate：

- checkpoint StateImage handle有效且为 immutable Device image；
- Main/Backend required logical pages均存在 epoch一致、coverage足够的 Device replica；
- target identity和typed frontier关系精确匹配 request；
- `ResourceDemand`的每个维度均不超过Program启动时的总Device capacity。

Continuation summary的 KV footprint按 address space unique mapped pages计数，而不是 endpoint
与 rewrite requirement之和。Active summary使用 mapped pages加 remaining growth reservation。

## 8. Runtime contract 与 exact inspection

### 8.1 Common contract types

Runtime common层新增或替换以下语义类型：

```text
DeviceResources
ResourceDemand
ResourceDelta { removed, added }
Readiness { Ready, NeedsTransfer, TemporarilyBlocked, PermanentlyInfeasible }
CheckpointKind / CheckpointRef
ContinuationId / Revision
```

Package-owned opaque类型为：

```text
RequestBasePlan
ProgramActivationPlan
SequenceHandle
ContinuationHandle
MaterializationOutcome
```

`ProgramActivationPlan` 是 move-only、owner-bound、source-generation-bound且
destination-lane-epoch-bound的capability。它可以表达：

```text
FullReset
MoveEndpoint
MoveTypedRewrite
```

它不把 StateImage handle、logical page handle或physical ID暴露给 Runtime common层。

### 8.2 RequestBasePlan

`plan_request`继续在 represented request上执行一次，返回 root/full-reset 的：

- prompt/output limits；
- exact Main/Backend full-lifetime page entitlement；
- 本请求需要的一幅或两幅 active state entitlement；
- root service-work quanta；
- rewrite capture意图；
- `allow_prefix_reuse`。

State entitlement由目标 plan精确给出：没有 surviving/captured rewrite的路径为一幅；需要保留
或建立 typed rewrite的路径为两幅。Agent rotation保证不需要第三幅 image。

### 8.3 Catalog summary 与候选枚举

ResourceManager catalog保存 Program在 publication时返回的 revisioned summary：

```text
unique Device footprint
endpoint frontier and rebuild-work summary
optional rewrite kind/frontier and rebuild-work summary
```

它不保存 exact prefix identity。对 Scheduler已选择的一个 request，ResourceManager：

1. 选择最低编号空闲 destination lane；
2. 若 request允许 reuse，枚举每个 `Catalogued` entry的 endpoint和optional rewrite；
3. 始终加入 root candidate；
4. 将每个 `CheckpointRef` 与 `ContinuationHandle const&`交给 Program exact inspection。

Catalog最多 `2C` entries，因此候选数上界为：

```text
1 root + 2 * (2C) private checkpoints <= 33
```

无需把 source与每个空闲 lane做笛卡尔积：state/KV已与 lane解耦，所有空闲 lanes在资源语义
上同构；destination只在选定后绑定最低编号 lane epoch和execution rows。

### 8.4 ResumeAssessment

Program inspection无副作用。Exact match返回：

```text
CheckpointRef
reused prompt tokens
PrefixReusePath
ProgramActivationPlan
ResourceDemand, including full active entitlement and prepublish additional reservation
source unique DeviceResources
remaining service-work quanta
source capability generation required by the plan
```

Identity mismatch只淘汰该 candidate；它不是错误，也不修改 continuation。Owner/generation
mismatch表示内部 capability invariant失效，不能当作普通 miss。

ResourceManager在接收assessment时另行捕获catalog `ContinuationId/revision`；Program不读取或
生成ResourceManager revision。

Root assessment不依赖 catalog，作为稳定 fallback和 Scheduler head-protection demand。由于本
阶段没有 shared physical pages，private hit不会使 request的最大 future KV entitlement小于
root；它只改变已有 source footprint的重分类、剩余 prefill work和可能保留的 rewrite state。

## 9. Candidate、victim 与 readiness

### 9.1 只读选择

Inspection和selection完全只读。一个 `ResourcePlan` 固定：

```text
destination lane + lane epoch
selected checkpoint or root
ProgramActivationPlan
source ContinuationId/revision, if any
whole-continuation victim IDs/revisions
active Device entitlement
prepublish additional Device reservation and source conversions
expected ledger `removed/added` vectors and victim physical releases
service-work / victim-loss cost
intended finish-publication catalog slot
```

它不持有 ResourceManager自行选择的 raw slot/page/row IDs，也不跨 GPU boundary缓存。

### 9.2 Feasibility equation

对 Device-only private Move，ResourceManager同时验证两个方程。Source仍保持published时的
transaction peak为：

```text
used
- committed whole-victim physical releases
+ prepublish_additional
<= Device capacity
```

publish后的稳态为：

```text
used
- selected source catalog footprint
- whole victim catalog footprints
+ selected active entitlement
<= Device capacity
```

Source subtraction表示同一 unique resources在 publish时从 catalog ownership重分类为 active
ownership，不表示 publication前可以破坏 source。Victims在 materialization中先完成 logical
eviction并返回 exact release acknowledgement，随后其 capacity才可分配给 destination。
Source suffix pages、endpoint state slot或rewrite state slot只有在Program声明为
`source_conversions`且publication metadata transition可以原子完成时，才可直接变成active
ownership或growth reservation；它们不进入prepublish release credit。

ResourceManager 对所有非 source catalog entries枚举 bounded victim subsets，并要求同一subset
同时满足transaction-peak与steady-state方程。`C <= 8`、catalog
capacity `2C <= 16`，因此最多检查 `2^16` 个简单资源向量组合；该工作只发生在 admission
boundary，不进入 decode/prefill GPU hot path。

### 9.3 初始确定性 policy

本阶段不引入未校准的在线成本模型，选择固定为：

1. 先只比较不需要 non-source victim的 exact candidates；
2. 若该集合非空，选择 remaining service-work quanta最小者；
3. 若全部需要 victim，枚举满足缺口的 whole-entry subsets，并最小化：

   ```text
   remaining service-work quanta
   + sum(victim rebuild-work quanta)
   ```

4. 同成本时依次选择 reused prompt tokens更多、victim数量更少、ContinuationId序列更小、
   endpoint优先于rewrite的方案。

Program使用与 Scheduler service-work相同的 quanta尺度报告每个 entry从 root重建其最深
surviving checkpoint的 work；whole-entry victim loss取该值，不把同一 lineage的 endpoint和
rewrite重复相加。此 policy在所有 candidates均为Device-ready、state恢复均为private Move的
Step 3中可实行；Host transfer与shared fan-out加入后由正式架构成本模型替换。

### 9.4 Readiness

ResourceManager返回四态结果：

- `Ready`：已有空闲 lane，且通过零个或多个同步 whole-entry evictions可以取得完整 active
  entitlement；
- `TemporarilyBlocked`：root plan本身可装入 Engine，但当前 active requests占用的不可回收
  entitlement或active-lane上限使本 boundary无法发布；
- `PermanentlyInfeasible`：root/full-reset完整 entitlement超过 Engine总 capacity；
- `NeedsTransfer`：Step 3不可达，留给 Host replica阶段。

Catalog entries不会把一个本可运行的 root request永久阻塞，因为本阶段它们均可整体回收。
Scheduler继续使用 root demand建立 protected-head epoch，使用 `Ready` choice的 actual active
demand和service work判断backfill，不持有 source或victim信息。

## 10. MaterializationTransaction

### 10.1 Engine-visible sequencing

本阶段materialization同步完成，不跨 worker boundary。状态为：

```text
Inspected -> Reserved -> ReadyToPublish -> Published -> Finalized
                         \-> Aborted
```

没有 `Transferring` 状态实例，也没有异步 event ownership。

### 10.2 Reserved

同一 Engine worker在第一次 Program physical mutation前完成：

1. 再次检查 request cancellation、destination lane、catalog IDs/revisions和publication slot；
2. Program统一验证 activation、source和全部 victim handles的owner/generation；
3. source entry进入 `Claimed`，victim entries从普通candidate discovery隐藏；
4. ResourceManager checked验证prepublish peak、完整active entitlement和expected delta；
5. 为root的staging record、最终continuation/catalog/address-space slots、state/KV logical
   objects和active identity/ledger预留全部host-side bookkeeping；被选victim提供的slot此时只被
   transaction条件占用，不覆盖victim内容；
6. Program建立唯一materialization ticket。

任何 host allocation或vector reserve在第5步结束前完成。失败时尚无 physical mutation，choice
可丢弃，request按错误类型结束或重新inspection。

### 10.3 Device preparation

Program按以下顺序执行 Device-only preparation：

1. 完整验证后，消费并释放plan中的whole-continuation victims；
2. root path取得新的 continuation、StateImage destination和typed KV address spaces；private
   path只建立source的transaction-owned staged view，source objects保持原published内容；
3. 对endpoint Move准备role/address-space adoption；对rewrite Move准备endpoint suffix release、
   partial-tail epoch切换和typed-frontier回退，但不提前提交这些destructive changes；
4. 取得Main/Backend future-growth transaction reservations，publication前不把它们写入source
   address spaces；
5. 取得destination execution rows并从logical address spaces发布block tables；
6. 建立ActiveStateBinding、request control和target metadata；
7. FullReset只清零新mutable StateImage；private Move不执行完整StateImage D2D copy；
8. 复核Device stores的actual occupancy与plan expected delta。

所有CUDA操作进入现有Program stream。后续首个prefill unit在同一stream执行，因此无需为
顺序正确性增加全局同步；若publish前出现无法确认的CUDA状态，按Engine-wide failure处理。
Selected source的state roles、address-space frontier、logical-page epoch/coverage和suffix ownership
在本节始终保持原值；transaction abort因此仍能原样返回source。

### 10.4 Publication 与 Engine adoption

最后一次cancellation检查通过后，Program commit materialization ticket，并在唯一publication
point执行source Move、rewrite truncate/epoch transition和transaction reservation adoption。
这些metadata转换使用Reserved阶段已经验证并预留的storage；commit后返回：

```text
SequenceHandle
exact ResourceDelta { removed, added }
active DeviceResources
source/victim consumption acknowledgements
```

ResourceManager先核对结果与已经reserved的plan，但将最终Active ledger transition封装在一个
move-only `PublishedActivation` adoption token中。随后同一worker、无分配地完成：

```text
Program physical publish
-> RequestRecord installs SequenceHandle/lane/budget
-> ResourceManager adopts exact delta and lane Active record
-> selected publication catalog slot becomes ReservedForActive
-> Scheduler commits AdmissionGrant
```

这些步骤在worker之外不可见。RequestRecord安装和ResourceManager adoption所需storage均已
预留；publish后的mismatch或未能consume adoption token进入Engine-wide failure，不重新选择
root或另一个source。

Materialization本身不执行模型prefill。完成上述Active adoption后，Engine立即调用现有
`advance_prefill`执行一个完整GPU unit，再返回`RanGpuUnit`。因此Scheduler的“successful
admission包含一个GPU unit”行为不变，同时Active publication point不再包裹在首个模型unit
内部。

### 10.5 Aborted

本阶段只允许在Program publish前形成已闭合的 `Aborted` outcome：

- destination state/KV reservations和execution rows已释放；
- selected private source仍然有效，并以完整 `ContinuationHandle + summary`返回
  ResourceManager；
- 已经为capacity preparation完成的whole-victim evictions不回滚，并带回exact released
  resources；
- 没有SequenceHandle或半发布Active binding；
- materialization ticket已消费。

ResourceManager先采用victim delta，再将source从 `Claimed`恢复为 `Catalogued`并推进logical
revision。Cancellation在第一次mutation前没有victim side effect；mutation开始后观察到的合法
pre-publish cancellation可以保留已经提交的victim evictions，但不破坏source。

## 11. Active execution 与 finish闭环

### 11.1 Prefill/decode/commit

现有GPU scheduling unit、compact batch、PendingBatch和output transaction保持不变。变化仅在于：

- lane通过ActiveContinuationBinding访问continuation/state/address spaces；
- 每个推进target state的invocation从binding生成state src/dst和KV table row selectors；
- Program commit同时推进state binding、logical KV committed coverage、target ledger和backend
  frontiers；
- rejected speculative suffix不进入StateImage checkpoint或logical-page canonical coverage；
- cancelled commit释放active continuation及其publication-slot reservation，不发布catalog entry。

ResourceManager active ledger保存该sequence完整entitlement；正常执行中不调整或借出其中的
state slot、Main pages或Backend pages。

### 11.2 Checkpoint capture

Program只在target state、Main KV、optional Backend KV、token ledger和prefix identity已经共同
commit到同一typed frontier后发布rewrite或endpoint checkpoint。

Runtime commit显式接收Program返回的Fork resolution。若一次provisional generation最终接受
零个state transitions，`fork_pending` destination不成为committed state，read source继续表示
当前frontier；nonterminal request保持pending binding，terminal request在finish时释放
destination并从完整read source发布endpoint。

Typed rewrite capture使用第6.5节rotation，并更新同一 address spaces的protected requirement。
如果desired rewrite frontier不晚于selected reuse frontier、而source又没有精确匹配该desired
checkpoint，Program固定选择现有`DeferCapture`：本请求不创建该较早checkpoint，source中其他
仍合法的rewrite按inspection plan保留。不能通过复制一个不对应该frontier的state伪造capture。

若terminal unresolved Fork的read source同时就是同frontier的new rewrite，finish只发布一幅
endpoint image并省略这个内容完全相同的optional rewrite；不让两个logical checkpoints伪装成
两幅独立images，也不复制整幅state来保留一个对recoverability没有增量的duplicate frontier。

### 11.3 Finish

Terminal accepted rows继续先全部完成Program commit，再逐行finish，最后统一发布用户输出。
ResourceManager拥有retention decision，正常成功路径选择private publication：

```text
Program Freeze committed active state as endpoint
-> release unmapped KV growth reservations
-> unbind execution rows
-> return ContinuationHandle + revisioned summary + exact delta
-> ResourceManager checks catalog publication reservation
-> Active -> Catalogued
```

若active lineage来自catalog source，保留同一 `ContinuationId`并推进revision；root lineage在首次
成功publication时取得新ID。Endpoint和optional rewrite的unique footprint逐维不大于active
entitlement，且`active_lanes`从1变为0。

Cancellation和abort消费active binding、释放全部unique resources及reserved catalog slot。
Destructive private Move之前的旧endpoint不恢复，也不从可能不一致的active state捕获新
checkpoint。

## 12. Scheduler 与错误语义

### 12.1 Scheduler保持不变的authority

Scheduler仍唯一负责：

- FIFO head与protected-head epoch；
- persistent/temporal backfill；
- staged prefill owner；
- admission/prefill/decode boundary顺序；
- compact decode membership与service-work accounting。

它只接收root或selected candidate的 `DeviceResources`、service work和readiness，不读取catalog
handle、checkpoint kind、victim set或physical stores。

### 12.2 Request-local结果

以下结果发生在第一次Program mutation前，按request处理：

- represented prompt或requested capacity非法；
- root plan `PermanentlyInfeasible`；
- waiting cancellation或queue timeout；
- exact identity mismatch导致某个candidate被淘汰；
- 合法catalog revision变化导致choice丢弃并重新inspection；
- host-side admission bookkeeping在mutation前无法建立。

`TemporarilyBlocked`不结束request，由现有head protection/backfill路径等待后续boundary。

### 12.3 Engine-wide failure

以下情况不能降级成cache miss或full reset：

- Program capability owner/generation mismatch；
- state role/pin、logical-page reference、epoch或coverage invariant被破坏；
- actual resource delta与selected plan或ResourceManager ledger不一致；
- Program physical publish后RequestRecord/ledger acknowledgement不一致；
- CUDA launch/copy/commit状态不可信；
- PendingBatch、finish或cleanup consumption contract失败。

Engine沿用现有唯一worker fail-all路径：Program清空active/catalog physical stores，
ResourceManager清空logical ledger和handles，所有requests完成为error。

## 13. 热路径与内存约束

Step 3 的性能结构固定为：

- 普通decode始终`state_src == state_dst`；
- 普通decode、prefill和commit不执行完整StateImage copy；
- private resume使用Move，不执行完整StateImage copy；
- 只有active Freeze后的首个transition使用`src != dst`；
- DFlash Fork只复制local cyclic component；
- selector在kernel inner recurrence之外解析；
- exact-B CUDA Graph topology不依赖具体StateImage slot或continuation ID；
- block tables、state selectors和round ingress继续使用稳定地址；
- Device/pinned memory仍只在Program startup分配；
- decode、commit、finish和Graph replay不新增heap allocation；
- candidate/victim枚举使用固定数组，只发生在admission boundary。

State payload仍为`2C` images，KV physical capacity与Step 2相同。本阶段不以吞吐提升为目标，
也不接受通过每次resume或checkpoint capture复制完整GDN state来换取较小代码改动。

## 14. 预计代码边界

### 14.1 Runtime common / Engine

预计修改：

- `src/runtime/contract/types.h`
- `src/runtime/engine/resource_manager.h`
- `src/runtime/engine/request_record.h`
- `src/runtime/engine/engine_core.h`
- `src/runtime/engine/admission_policy.h/.cpp`
- `src/runtime/engine/scheduler.h`
- 对应 ResourceManager/admission tests

这里完成Device resource vector、global catalog/ledger、readiness、ResourcePlan adoption和
Engine materialization sequencing，不加入target数学或physical IDs。

### 14.2 Qwen3.6 family runtime

预计修改或新增：

- `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/runtime.h`
- `src/targets/qwen3_6/impl/runtime/program.h`
- `src/targets/qwen3_6/impl/runtime/program_impl.h`
- `src/targets/qwen3_6/impl/runtime/request_plan_impl.h`
- `src/targets/qwen3_6/impl/runtime/layouts_impl.h`
- `src/targets/qwen3_6/impl/runtime/state_image_store.h`
- `src/targets/qwen3_6/impl/runtime/logical_kv_store.h`
- Text/MTP/DFlash schedule、context、ingress/egress和Graph preparation consumers

Target packages继续只alias/instantiate family runtime，不复制Program或store实现。

### 14.3 Ops

预计修改现有semantic Ops：

- batched GDN recurrent update的src/dst selector contract；
- GDN replay fold的src/dst row contract；
- 受影响wrapper、launcher、kernel和独立数值测试。

只扩展同一数学state transition的destination表达，不建立context-cache专用重复kernel。

### 14.4 不修改

本阶段不修改：

- `HostStatePool` / `HostKVArena` production contract；
- attention、paged-KV append和KV codec数学；
- public Engine headers中的新cache配置；
- CLI、serving schema、artifact/converter；
- 正式架构文档。

实际受旧lane-resident ownership影响的consumer均随cutover修改；是否触及上述每一个预计文件
不是完成条件。

## 15. 实施顺序

### Phase 1：Common resource contract 与 bounded stores

- 用 `DeviceResources`贯穿RequestPlan、Scheduler snapshot和ResourceManager ledger；
- 建立fixed global catalog和active publication-slot reservation；
- 新增Program continuation descriptor store、StateImageStore和typed logical KV/address-space
  stores；
- 先用focused store tests闭合handle generation、roles、coverage和容量守恒。

阶段内不切换生产request lifecycle。

### Phase 2：State binding cutover

- 将SequenceState中的固定current/rewrite Tensor views替换为StateImage handles；
- 扩展batched GDN和replay fold的src/dst contract；
- 切换Text、ordinary、MTP、DFlash、hidden publication和Graph selectors；
- 实现Freeze/Fork resolution及DFlash local component copy；
- 删除`StateImageSlots`。

阶段内以独立Op numerical tests和Qwen runtime mechanism tests验证。

### Phase 3：Logical KV/address-space cutover

- 将Main/Backend mappings迁入LogicalKVPageStore/KVAddressSpaceStore；
- 切换reservation、incremental materialization、block-table publication、commit coverage、trim、
  rewrite truncate、finish和cleanup；
- 删除`PrivateKVMapping`和任何第二份ordered physical-page ownership。

阶段内以logical-store test和现有runtime mechanism test验证。

### Phase 4：Program continuation 与 materialization

- 拆分active lane control和continuation record；
- 改造ContinuationHandle为lane-independent capability；
- 实现root/endpoint/rewrite exact inspection；
- 实现Program materialization ticket、Published/Aborted outcome和finish publication；
- 将当前`start_request`拆为materialize与首次`advance_prefill`。

### Phase 5：ResourceManager/Engine production cutover

- 切换global candidate/victim选择和four-state readiness；
- 切换source claim、publication slot、exact delta和PublishedActivation adoption；
- 切换terminal finish、cancel、abort和fail-all cleanup；
- 删除lane Resident lifecycle及旧release/retention路径；
- 执行第16节验证并更新总实施记录状态。

## 16. 验证与完成条件

### 16.1 Focused store 与 policy evidence

新增一个Qwen family logical-resource focused test，使用小型production layouts覆盖这些真实
转换：

1. StateImage动态slot allocation与lane无关，released generation不可复用；
2. private Move不复制source，Freeze/Fork保持source immutable，并在commit后收敛到
   `src == dst`；
3. rewrite rotation在两幅images内完成；
4. logical KV append只在commit后推进coverage，rejected suffix不发布；
5. endpoint/rewrite共享一个address space且footprint不重复计页；
6. rewrite truncate推进content epoch并使旧suffix失去语义；
7. catalogued address space不占execution row，任意active lane可重新publish mapping。

重写现有ResourceManager focused test，覆盖：

- source continuation与destination lane解耦；
- Device state/Main/Backend各维容量判断；
- 先无victim、后whole-entry victim subset的确定性选择；
- source claim/abort恢复、victim committed release和exact delta adoption；
- root active的finish-publication slot在catalog pressure下仍已预留。

不为private getters、enum值、构造函数、已删除compatibility或相同stale check的每个入口增加
重复测试，也不引入Host/shared/COW fake behavior。

该focused test固定落在：

```text
tests/targets/qwen3_6/test_context_store.cpp
target: ninfer_qwen3_6_context_store_test
```

### 16.2 State transition numerical evidence

修改现有测试而不是建立重复suite：

```text
ninfer_gated_delta_net_test
ninfer_gdn_replay_fold_test
```

在既有独立FP32/FP64 oracle与真实registered shapes上增加distinct src/dst cases，验证：

- output与destination final state满足现有numerical criterion；
- source state bitwise不变；
- `src == dst`与当前in-place contract一致；
- batched mixed selectors写入各自destination且不交叉。

### 16.3 Focused build/test set

最终focused集合限定为：

```bash
cmake --build build -j --target \
  ninfer_resource_manager_test \
  ninfer_qwen3_6_context_store_test \
  ninfer_gated_delta_net_test \
  ninfer_gdn_replay_fold_test \
  ninfer_qwen3_6_runtime_mechanisms_test

ctest --test-dir build --output-on-failure \
  -R '^(ninfer_resource_manager_test|ninfer_qwen3_6_context_store_test|ninfer_gated_delta_net_test|ninfer_gdn_replay_fold_test|ninfer_qwen3_6_runtime_mechanisms_test)$'
```

`ninfer_qwen3_6_context_store_test`只覆盖新logical ownership/state transitions；既有runtime
mechanisms target继续覆盖layout、prefix identity和family control。Step 1/2 physical-container
tests只有在对应物理contract被实际修改时才加入，不机械重复全部历史gate。

### 16.4 Representative real execution

生产Program、prefix ownership、state binding和typed KV lifecycle均被切换，因此运行现有三个
代表性real routes：

```text
ninfer_qwen3_6_27b_prefix_real_test
ninfer_qwen3_6_35b_a3b_real_test
ninfer_qwen3_6_35b_a3b_dflash_real_test
```

它们分别覆盖27B Text/Vision/MTP/typed rewrite、35B MTP/private append和35B DFlash
endpoint/rewrite/cyclic state。复用仓库已有artifacts和环境变量，不新增identity、dtype、协议或
concurrency矩阵。

验收使用现有exact reuse path/count、greedy output、speculative commit和CUDA execution断言。
不把一次CLI smoke、完整测试套件或benchmark叠加为额外gate；若real test因明确的本机artifact
或显存前提无法运行，记录未覆盖的具体route与原因。

### 16.5 Structural review

执行：

```bash
git diff --check
```

并审查以下production事实：

- ResourceManager catalog不再按lane索引continuation；
- ContinuationHandle不含lane；
- Program没有`Lifecycle::Resident`、`StateImageSlots`或`PrivateKVMapping`；
- state/KV physical facts只有Program stores一份authority；
- normal decode没有完整StateImage copy，Graph不绑定固定slot；
- catalog/transaction/publication不在commit后扩容host bookkeeping；
- 没有Host replica、shared prefix、COW、partial eviction或compatibility fallback混入；
- 正式架构文档未被本阶段执行计划改写。

本阶段没有性能提升claim，因此不设置吞吐百分比gate。若代表性执行出现明确、可重复且能归因
于新增selector或copy的热路径回退，应在完成前修正；不为此扩张benchmark矩阵或引入独立
profiling campaign。

### 16.6 最终完成条件

Step 3 只在以下事实同时成立时记为实施完成：

1. 第2节纵向链路由唯一生产实现闭合；
2. lane、continuation、StateImage和KV address-space ownership符合本文固定关系；
3. ResourceManager与Program exact resource delta在start/finish/cancel/evict路径全部闭合；
4. Text、Vision、MTP、DFlash和typed rewrite的代表性真实路径通过；
5. 旧lane-resident与直接physical-mapping authority已删除；
6. 已知范围内没有Host/shared等后续语义的半实现。

完成后只在总实施记录中补充状态、提交和验证摘要；后续 Host replica、shared prefix和policy
阶段另行制定实施计划。
