# NInfer Paged KV Context Store

本文定义 NInfer growing KV 的物理存储与消费合同。它是 typed KV pools、logical pages、
Device/Host replicas、address spaces、reservations、block tables 和 GPU consumer views 的维护者权威。

请求顺序和生命周期见 [Engine 架构](engine-architecture.md)；checkpoint、缓存保留、抢占和资源
回收策略见 [资源调度与上下文缓存](resource-scheduling-and-context-cache.md)。KV Store 兑现选定操作的
物理需求，并向模型 execution unit 提供稳定的直接访问视图。

---

## 1. 物理模型

Growing KV 使用一组启动时固定的 homogeneous pools。每个 pool：

- 存储具有同一 frontier 和 lifetime 的全部 planes；
- 使用固定的原 token page size、plane order、各 plane 行跨度和 page-group count；
- 拥有独立的 physical page-ID namespace 与 free capacity；
- 为每条 active sequence 提供一个 logical address space；
- 由 consumer 通过 block table 直接寻址。

一个 request 的 KV 不要求物理连续，也不与 control lane 固定绑定。所有 active 与 inactive address spaces
共享 pool capacity。普通 reservation 保障下一单元增量；暂停恢复的 reservation 覆盖重建至旧
frontier 及首个真实新单元，并跨 chunk 持有。

Paged storage 覆盖按上下文增长的 KV。DFlash local cyclic state、Vision/query temporary K/V 和其他固定
state 具有不同 lifetime，由其各自的 StateImage 或 workspace contract 管理。

---

## 2. 三种独立粒度

KV 架构区分：

| 粒度 | 含义 | 当前合同 |
|---|---|---|
| allocation granularity | pool 一次取得或释放多少 token payload | 每个 growing pool 为 `P=64` |
| valid-frontier granularity | consumer 可以读取到哪个 logical position | 1 token |
| reusable-state granularity | 哪个 frontier 具有完整模型 continuation | target-defined checkpoint |

Page boundary 不是 Attention mask boundary，也不是 prefix hit boundary。一个 valid frontier 可以位于 page
内部任意 offset。

各 plane 的物理行密度由 `KVPlaneGeometry::tokens_per_row` 单独描述：一行可覆盖 1 或 4 个原 token。
它不改变 allocation、frontier 或 checkpoint 的原 token 域；有效行数按第 4 节从 coverage 推导。

KV Store 可以在任意 token frontier 表示、truncate 或保护 prefix；这不证明模型可以从该位置恢复。
可复用 frontier 必须同时存在完整 StateImage 与 target-defined backend state，具体规则见
[Continuation 与 checkpoint](resource-scheduling-and-context-cache.md#checkpoints)。

---

## 3. Typed pool set 与容量

### 3.1 Pool set

模型配置和 selected speculative backend 在启动时确定 pool set：

```text
ordinary:
    Main Text

MTP:
    Main Text
    MTP

DFlash / DFlash2:
    Main Text
    Draft Full, when the selected draft config contains full-attention layers
```

Speculative backends 在一个 Engine 内互斥，因此当前最多有两个 growing pools。官方 DFlash2
配置的五层全部是 local attention，只需要 Main growing pool，其 draft context 使用 cyclic storage。

| Pool | 内容 | 逻辑 frontier |
|---|---|---|
| Main Text | target full-attention K/V 与其 code/scale planes | target materialized KV frontier |
| MTP | MTP persistent K/V 与其 code/scale planes | MTP KV frontier |
| Draft Full | selected draft 的 full-context K/V | draft context frontier |

Main Text 与 MTP 使用 Engine 选择的 BF16、INT8-G64、FP8-E4M3FN-row256、NVFP4-G16 或 K8V4
KV profile；Draft Full 使用自己的 BF16 profile。`BFloat16` 名称下的物理 layout 为 BF16 K、FP16 V，
写入端将 BF16 V 一次转换为 FP16。K8V4 是封闭的非对称 profile，不是运行时 bit-width 组合：K 固定为
FP8-E4M3FN-row256，V 固定为 NVFP4-G16。

`PagedKVStorageLayout` 将选定的 closed profile 解析为 K/V data/scale plane schema；target planner 按
layer 展开该 schema 并确定 plane ordinal。Common pool implementation 仍只接收已展开的
`KVPageGeometry`、plane inventory 和 capacity，不解释 storage mode。
上述 Qwen3.5 K/V data 与 scale planes 均使用 `tokens_per_row=1`；通用存储层同时支持第 4 节的
四-token 行几何，具体 plane 的数学含义与消费支持由模型和 Op 定义。

### 3.2 Main capacity

设：

- \(S\)：单条 sequence 的 `max_context`；
- \(C\)：`max_concurrency`；
- \(P=64\)：Main page size；
- \(L=\lceil S/P\rceil\)：单 address space 的 logical page capacity；
- \(M\)：Main pool 的 physical page-group count。

可用范围为：

\[
M_{min}=\max(L,C)
\]

\[
M_{max}=C\,L
\]

\(M_{min}\) 分别满足单请求独占时达到 \(S\)，以及 C 个请求各需一个最小页的几何下界。
\(M_{max}\) 是全部 active requests 同时达到 per-sequence ceiling 时的物理上界。

`kv_capacity` 的 explicit policy 解析为：

\[
M=\left\lceil K_{main}/P\right\rceil
\]

并要求 \(K_{main}\ge S\) 且 \(M\in[M_{min},M_{max}]\)。

### 3.3 Automatic capacity

Automatic policy 在权重加载后，以当前可用显存 \(F\) 和要求保留的 headroom \(R\) 解析一次 \(M\)。
模型 planner 根据已绑定参数和选定执行域提供 affine `SequenceCapacityCurve`：

\[
B(M)=B_{min}+(M-M_{min})B_{step}
\]

其中 \(B(M)\) 是该 Main capacity 对应的完整 runtime Device reservation，包含：

- Main 与 selected backend typed pools；
- active/checkpoint State storage；
- block tables 与固定 persistent state；
- unified workspace；
- CUDA Graph allowance。

Automatic 选择：

\[
M=
\min\left(
M_{max},
M_{min}+
\left\lfloor
\frac{F-R-B_{min}}{B_{step}}
\right\rfloor
\right)
\]

并要求 \(F\ge R+B_{min}\)。当 \(M_{min}=M_{max}\) 时直接取该单点。Target 用同一生产 layout builder
生成 \(B_{min}\)、\(B_{step}\) 和最终 layout，并验证 capacity curve；resolver 不复制模型维度公式，也不靠
allocation probing 猜测容量。

最终公开的 Main KV capacity 为 \(M\,P\) token-equivalents。Page rounding 只增加 physical padding，
不会扩大单 sequence 的 logical ceiling \(S\)。

### 3.4 Backend capacity

Selected backend 的 logical page capacity仍为 \(L\)。Physical capacity从 Main \(M\) 推导：

```text
backend off:
    Main physical pages = M

MTP with draft window K:
    Main physical pages = M
    MTP physical pages  = M + C * ceil((K - 1) / P)

DFlash / DFlash2 with full-attention layers:
    Main physical pages        = M
    Draft Full physical pages = M
```

MTP 的额外 pages 只覆盖每条 active row 在一个 speculative round 中相对 Main 的 provisional lead，
不扩大任一 address space 的 logical capacity。Draft Full 没有这种 provisional lead；没有 full layer
的 draft 配置不分配此 pool。

各 pools 物理分离。一个 pool 的 free page 不能变成另一 pool 的 payload。Program 在启动时一次性建立
完整 typed capacity vector，运行期不扩容或重分 pool geometry。

### 3.5 Host capacity

StateImage、Main KV 和 selected backend KV 共用 startup-fixed pinned `HostContextArena`。
`HostKVArena` 在同一 backing 上提供 typed KV allocations；每个 allocation 携带自己的 page layout。
Host capacity 按实际 packed bytes 和 extent geometry 计费，不扩大 Device capacity 或单 sequence context
ceiling。输入、请求 ledger 与其他 CPU 数据有各自的生命周期，不计入这个物理 context backing。

---

## 4. Page group 与物理 layout

### 4.1 Grouping invariant

只有同时满足以下条件的 planes 才进入同一个 pool：

1. 使用相同 logical cache ordinal；
2. 共享同一个 committed frontier；
3. 一起 reserve、materialize、truncate、retain 和 release；
4. 使用相同 page size；
5. 没有独立释放或独立容量复用的语义。

因此 Main layers 的 K、V、code 和 scale 可以共享一个 page-group ID；Main、MTP 与 DFlash Full
必须属于不同 pools。

### 4.2 Page group

一个 pool-local page-group ID \(g\) 同时选择该 logical block 在全部 grouped planes 中的 payload：

```text
page group g
├── layer/plane 0 slice g
├── layer/plane 1 slice g
├── ...
└── layer/plane n slice g
```

Planes 拥有 Engine-lifetime-stable backing，但不要求组成一个连续 blob。一个 page group 是 allocation、
reservation、reference 和 transfer 的最小 Device 单位。

Physical IDs 可以任意排列。Allocator 可以优先返回连续 IDs 改善 locality，但 correctness、admission
和 kernel launch topology 不依赖连续性。固定大小 page groups 不产生 variable-size external
fragmentation，也不需要 Device compaction。

### 4.3 Plane 行跨度与 Device order

Device pool 使用原 token 页跨度 \(P=64\)。每个 `KVPlaneGeometry` 独立给出 dtype、leading extent
\(X\)、head extent \(H\)、alignment 和 `tokens_per_row` \(s\)，物理行数为 \(R=P/s\)。
当前接纳以下组合：

| \(s\) | 每页行数 \(R\) | Device order |
|---:|---:|---|
| 1 | 64 | PageMajor 或 HeadMajor |
| 4 | 16 | PageMajor |

`kv_plane_rows_per_page()` 为 Device 与 Host 规划提供同一检查：\(s\) 只能为 1 或 4，
\(P\) 必须为正且能被 \(s\) 整除，\(R\) 必须能表示为正的 Tensor extent；四-token 行要求 PageMajor。
dtype、\(X\)、\(H\) 和 alignment 仍是各 plane 的存储事实，\(X\)、\(H\) 必须为正。
行跨度不表示 feature 轴的量化 group size；同一 token 的 K/V code 与 scale 各占一行。
行跨度参与 geometry 相等性和 Host layout 选择；Device backing 的实际 shape 必须与该 geometry 一致。

两种 closed orders 为：

\[
PageMajor:[X,R,H,N_{physical}],\qquad HeadMajor:[X,R,N_{physical},H]
\]

Main Text 与 MTP 的 K/V planes 使用 PageMajor；DFlash Full 使用 HeadMajor。
对这些现有 K/V profiles，\(s=1\)，其中：

- \(X=D\) 表示 K/V 或 quantized code plane；
- INT8-G64 scale plane 使用 \(X=D/64\)；
- FP8-E4M3FN-row256 scale plane 使用 \(X=1\)；
- NVFP4-G16 code plane 使用 packed U8 \(X=D/2\)，scale plane 使用 U8 \(X=D/16\)；
- \(H\) 是 KV heads；
- \(N_{physical}\) 是该 pool 的 physical page count。

对 logical position \(p\)：

\[
b=\lfloor p/P\rfloor,\qquad o=p\bmod P,\qquad r=\lfloor o/s\rfloor,\qquad g=block\_table[b]
\]

\(r\) 定位包含该原位置的物理行；该行是否完整且可读仍由第 4.4 节的 coverage 决定。

Page-major 地址为：

\[
address=base+d\,nb_0+r\,nb_1+h\,nb_2+g\,nb_3
\]

Head-major 地址为：

\[
address=base+d\,nb_0+r\,nb_1+g\,nb_2+h\,nb_3
\]

所有 grouped planes 使用同一个 \(g\)，但使用各自的行跨度、leading coordinate 和 Tensor strides。
Feature 轴连续，地址和容量计算使用能覆盖完整 Tensor span 的整数类型。
Exact persistent codec 由 [`kv_cache_append.h`](../../include/ninfer/ops/kv_cache_append.h) 定义，
consumer arithmetic 由 [`softmax_attention.h`](../../include/ninfer/ops/softmax_attention.h) 定义；
allocator 只解释 plane bytes、order 和 page-group identity。

D256 Main/MTP profile 的单 token/head 物理 payload 为：

| profile | K code + scale | V code + scale | K+V |
|---|---:|---:|---:|
| BF16 | 512 B | 512 B | 1024 B |
| INT8-G64 | 256 B + 8 B | 256 B + 8 B | 528 B |
| FP8-E4M3FN-row256 | 256 B + 2 B | 256 B + 2 B | 516 B |
| NVFP4-G16 | 128 B + 16 B | 128 B + 16 B | 288 B |
| K8V4 | 256 B + 2 B | 128 B + 16 B | 402 B |

K/V 的 code 和 scale planes 具有各自的 dtype、leading extent 和 group size；它们仍共享 page-group
identity、frontier 和 lifetime。Capacity curve、Device/Host replica、continuation transfer 和 memory
summary 均从这一 typed plane inventory 计算，不能用 `2 * vector_bytes` 代替 K8V4 的非对称字节数。

### 4.4 Logical position domain

Block table 使用 autoregressive cache ordinal：

```text
logical block b covers positions [b*P, (b+1)*P)
```

RoPE、MRoPE、Vision axes 和 `rope_delta` 是 Attention input metadata，不改变 KV slot ownership。

对于已提交前沿 \(F\)，逻辑页 \(b\) 内的有效原 token 数为：

\[
c=\operatorname{clamp}(F-bP,0,P)
\]

跨度为 \(s\) 的 plane 只有行 \([0,\lfloor c/s\rfloor)\) 完整有效；行 \(r\) 对应原位置
\([bP+rs,bP+(r+1)s)\)。例如 \(c=5,s=4\) 时只有第一行有效，第五个 token 不会提前发布第二行。
所有 planes 共用同一原 token coverage，不维护独立行前沿。Consumer 还须遵守各自的因果读域，
不能把物理行存在或 chunk 末端的 coverage 当作每个 query 的可见范围。

尚未形成完整行的模型状态由其 StateImage 合同保存；页组本身不能替代该状态。
Provisional 写入或复制无效后缀字节均不扩大 committed coverage。

### 4.5 Page payload

各 plane 的完整页 payload 与 page-group payload 为：

\[
PlanePageBytes=sizeof(dtype)\,X\,H\,(P/s)
\]

\[
PageBytes=\sum_{plane} PlanePageBytes
\]

Startup physical bytes 由各 plane slab 的完整 span 与 alignment 得到。不同 pools 的 `PageBytes` 可以不同，
但一个 pool 内的所有 page groups 等价。Device slab、Host packed offsets、传输字节和 capacity query
均使用同一 plane geometry；不能将四-token plane 按 \(P\) 行计费。

Host layout 将每个 plane 的起点按 256 B 对齐，末端按 256 B 对齐形成 `page_stride`。
该 stride 用于 Host allocation；payload 字节不包含 plane 间和页末 padding。

`P=64` 同时满足当前 32/64-key Attention tiles、128-token aligned prefill chunks、有限 block-table
metadata 和 bounded tail slack。Prefix hit granularity不参与 page-size 选择。改变 page size、grouping
或 closed plane order 都是架构变更。

---

## 5. Logical page 与 replicas

### 5.1 Logical page identity

Device page ID 只是当前 Device replica 的物理位置，不是 prefix identity。Program 为每个 pool 维护
generation-checked logical pages：

```text
LogicalKVPage
├── object generation
├── content epoch
├── committed columns [0, P]
├── protected columns
├── address-space references
├── active references and writer state
├── optional Device page-group lease
├── optional Host extent membership
└── transaction pins
```

一个 logical page 的 canonical content 由 `content epoch + committed columns` 标识。
Committed/protected columns 始终计原 token，跨度不同的 planes 由此推导有效行数。
Speculative 或尚未提交的 bytes 不扩展 committed coverage。

同一 logical page 在所有引用它的 address space 中保持相同页序号。Fork 和 view 按原位置共享
前缀，COW 与增长创建新 logical page，截断只删除后缀；一个 address space 内也不重复引用同一页。
因此，核对选中页的完整持有者时，可以在各检查点覆盖内按该页序号查询实际 handle。

Replica 对 checkpoint 所需前 \(n\) 列有效，当且仅当：

\[
replica.epoch=page.epoch
\quad\land\quad
replica.coverage\ge n
\]

### 5.2 Device replica

Device replica 使用 `4` 的 consumer-native plane layout。`DeviceKVPageLease` 独占一个 pool-local
physical page group；generation 防止 release/reuse 后的 stale handle。

Pool 对同一容量单位区分：

```text
allocated page lease
reserved but not materialized page
globally available page
```

\[
allocated+reserved+available=capacity
\]

Materialize 把 reservation 转成 lease；dematerialize 把 lease 还回同一 reservation。普通 unit 结算
释放余额；受保护恢复则保留未来覆盖所需余额，直到真实新进展后归还。Checkpoint 别名和 history
共享引用不重复占物理页。

### 5.3 Host replica

Host replica 使用 logical-order packed `HostKVPageLayout`：

- 不保存 Device page ID 或 block-table holes；
- 每个 page 包含该 typed pool 的全部 grouped plane payload；
- variable-size extent只为实际 page count 付费；
- Main/backend layouts可以在同一 arena 中分配不同 stride 的 extents。

Host arena 是有界 variable-size allocator。State 与不同 KV layouts 竞争同一 backing，彼此没有固定
配额。申请必须满足 alignment 和连续 extent geometry；`free_bytes` 只是占用摘要，不是可分配性的
充分证明。共享 Host extent 按实际 allocation 计一次，pending destination 在 publication 前已经占用容量。

### 5.4 Replica transfer

Device COW 与 D2H/H2D transfer 复制页组的全部 plane payload，按各 plane 的实际行数、bytes 和
offset 搬运原 words。不同密度的 plane 不单独分配、保留或回收。传输可以把相邻 physical IDs
合并成更少的 transfer runs。
Core 按已绑定的 plane geometry 选择 2D copy：短 PageMajor run 可以合并等宽、等间距的 planes，
HeadMajor 可以选择 page 或 head 作为提交外层。只在减少调用数且 pitch 合法时使用合并路径；
Host layout 与 payload 不变。Native 从 Core 返回值记录实际搬运字节和调用数。
一个 replacement replica 的 publication 顺序为：

```text
reserve destination
  -> copy page payload
  -> verify epoch and committed coverage
  -> publish replica
  -> release source when no longer required
```

Copy 进行时 source 与 destination 都被 pin。Copy 完成前，destination 不进入 address space 或 execution
table；唯一有效 source 也不能先释放。

### 5.5 Descriptor lifetime

Logical descriptor 不构成第三份 payload。它可以在 Device-only、Host-only 或 Both placements 下继续存在。
只有当 references、replicas 和 transaction pins 均为零时，descriptor 才能回收并推进 generation。

---

## 6. KV history 与 address space

### 6.1 共享历史与独立视图

一个 `KVHistory` 持有 Main 和可选 backend 的 address spaces。私有 continuation 的当前执行状态及
内部恢复点共享同一个 history，各恢复点分别记录所需 frontier 和 StateImage。

```text
private continuation
├── current sequence ───────────┐
├── input recovery checkpoint ──┼── KVHistory
└── recent boundary checkpoint ─┘   ├── Main address space
                                   └── selected backend address space
```

History 的目录可以继续 append；旧恢复点只读取自己已保护的 prefix。保留多个恢复点不复制完整 KV
目录，也不重复计费前缀页。Program 根据存活恢复点的最大 Main/backend frontier 维护保护范围。

独立分支和公开共享前缀持有自己的 history。它们可以共享完整的物理前缀页，但不共享可变目录的
suffix。相同 token identity 的两个独立计算结果仍是不同内容对象；State 与 KV 必须来自同一实际历史。

### 6.2 Address space

```text
KVAddressSpace
├── logical-block -> LogicalKVPage directory
├── committed frontier
├── checkpoint-protected frontier
├── active/inactive state
├── unit / recovery growth reservation
└── optional execution-row lease
```

目录通过共享不可变 prefix 节点与私有 append path 维护 ordered membership。Block table 是 active
Device mapping 的执行镜像。Inactive history 不占 execution row，也不绑定原 lane。

每个 address space 分别记录：

| 事实 | 含义 |
|---|---|
| membership | 已属于该地址空间的 logical pages |
| committed frontier | 已形成 canonical content 的 token prefix |
| protected frontier | 仍有 checkpoint 需要的最大 coverage |
| growth reservation | 已为 unit 或完整恢复覆盖取得、尚未物化的增量页 |

Membership 可以覆盖 speculative window 等 provisional suffix，其 bytes 在提交前不可作为完整恢复点。
Main/backend 的 frontier 可以不同，它们的语义关系由模型 schedule 确定。

### 6.3 Execution rows

每个 pool 在启动时建立固定地址的 Device block-table matrix：

\[
block\_tables[N_{logical},C]
\]

其中 \(N_{logical}=L\)，\(C=max\_concurrency\)。每个 active address space lease 一行，条目为 I32
pool-local physical page ID。Activation 批量发布 membership 的 Device page IDs；恢复可以租用另一行。
Execution row 不拥有 logical pages、frontier 或 reservation。

## 7. 有限执行单元与生命周期

### 7.1 绑定

从 root 或 checkpoint 初次绑定时，Program 统一准备 State、全部 enabled KV pools 及首个合法 unit。
暂停恢复则准备完整恢复覆盖：旧 frontier、后端规范化/bridge 与首个真实新单元所需的峰值。

```text
lease source and acquire destination
  -> reserve missing replicas, private tails and unit/recovery growth
  -> restore missing Host-only pages
  -> Move or Fork memberships
  -> lease execution rows and publish mappings
  -> publish complete sequence
```

任一 pool 的不足都会阻止完整绑定。Source lease 覆盖所需传输与安装；不可逆接管前失败会清理
destination，保留来源。接管提交后，旧 checkpoint 可能已经消费，后续异常进入 Engine failure
cleanup。一次绑定的 State 和 KV 不从不同计算历史拼接。

### 7.2 Unit reservation 与物化

Program 计算选定 prefill、Replay、decode、control 或 normalization unit 在每个 pool 的最大写入位置。
已有 membership 不重复 reserve；先检查一个调用集合的全部 typed 需求，再安装许可。许可固定 unit
kind、token 参数和 typed frontier，执行必须匹配。Engine 通过逐行申请组成可运行子集。

完整恢复许可与当前 unit 参数分离：address space 持有到恢复覆盖终点所需的 reservation，当前
Replay chunk 只物化其中所需部分。恢复期间每个 unit 都必须落在这个已取得的覆盖内，其他请求
不能占用尚未消费的余额。

`ensure_mapped_to_tokens()` 接收本阶段所需覆盖下界：已有 membership 足够就直接返回；不足时将
该 address 的 reservation 转为 physical pages，并发布 table slice。它不推进 committed frontier，
不裁剪更长的 speculative mapping。所需页数超过 `membership + reservation` 属于许可违约。

Target prefill/verify 保障 Main KV；MTP 和 DFlash Full 分别保障 backend KV。DFlash2 的 local draft
context 使用固定 cyclic state。Ordinary decode 通常只在跨页时物化一个 Main page。

### 7.3 提交与 rollback

Unit 成功后，Program 提交对应 State 与 canonical KV frontiers。Speculative 路径先完成 accepted
prefix 所需 recurrent state、hidden 和 backend context，再发布 frontier。Consumer 不读取 rejected
suffix；partial page 中残留 bytes 不扩大有效范围。

结算时显式 truncate 未提交的尾页，dematerialize 后，普通 unit 释放余额；恢复期间保留到完整
恢复覆盖所需的剩余页。到达旧 frontier 本身不释放，真实新 prefill/token 提交或终态才结束保护。
`truncate` 不能删除 surviving checkpoint 的保护范围，也不能覆盖其他 reader 需要的 partial tail。
仅仅请求更短的
coverage 不会触发裁剪。

### 7.4 暂停、结束与释放

暂停或结束时释放 execution row、active references 和未用 growth reservation。需要保留的恢复点
继续持有完整 State/KV coverage；Snapshot 可以迁移至 Host，或在放弃物理加速副本后通过 Replay 恢复。

移除一个 checkpoint 只撤销它的 State 与 history 引用。其他 checkpoint、active sequence 或 transaction
仍需要的目录和 replicas 保持有效。最后一个 history owner 析构时释放 Main/backend address spaces；
最后一个 physical reference 消失时归还对应 replica。

### 7.5 稳定边界

从 block-table publication 到 GPU unit 完成，selected rows、membership、可读 frontier 和被访问
payload 均保持稳定。Mapping 更新、frontier commit、truncate 和 row recycling 在 GPU 边界完成；
allocator 与 transfer ownership 不进入 kernel。

## 8. Move、Fork 与 active prefix view

### 8.1 私有 history 的接管

私有 continuation 可以接管既有 history，在同一目录上继续 append。内部恢复点保留自己的 State
和 frontier；它们不会强制当前 sequence 在每次续接时复制全部 KV。若从较浅恢复点 rewind，先解除
失效的较深保护，再由 stores 验证并裁剪 suffix。

State 的 Move/Fork 与 KV history 的 Move/Fork 分别判断；完整 source coverage 与实际 reader lease
决定能否复用现有 writer，不能只看逻辑 checkpoint 数量。

### 8.2 独立分支

分支取得独立 history 和执行 row，共享 frontier 前的完整 logical pages。若边界落在 partial page，
先复制所需 tail 到私有 destination，再开始 append。

例如 \(P=64\)、frontier \(F=1000\)：共享前 15 个完整页，为最后 40 个有效 token 复制一页私有 tail。
Checkpoint 仍精确位于 token 1000，allocation 不把 hit frontier 向下取整到 960。

### 8.3 从 active history 导出不可变 prefix

`KVActivePrefixViewReservation` 从停在稳定边界的 active history 导出一个独立视图：

- source 保有 execution row、growth reservation、suffix 与 writer；
- destination 共享完整 prefix pages；
- non-aligned frontier 的 tail 复制到独立 page；
- destination 在复制完成后一次性获得 immutable membership。

导出期间 source 停止执行，reservation 与 source pins 保障内容有效。导出完成不更换 source row，
不截断其 suffix，也不重新发布它的 block table。它用于共享发布和仍在执行中的历史分支。

### 8.4 写保护

Stores 同时检查 protected coverage、active references、writer、epoch 与 transaction pins。
多个只读引用可以共享同一页；append 不覆盖任何 surviving checkpoint 的有效 prefix。
分支要写入共享 partial tail 时先取得私有 COW page。Refcount 本身不赋予写权限。

---

## 9. Speculative 与非 growing KV

### 9.1 Independent pool frontiers

MTP 和带 full layer 的 DFlash backend 使用 Program KV Store 的 backend pool，不建立独立 allocator。

一次 speculative unit 中，Main 与 backend：

- 分别从各自 unit reservation materialize；
- 可以具有不同 mapped/provisional frontiers；
- 分别提交 accepted frontier；
- 分别 trim rejected trailing mappings。

MTP draft 期间 backend mapped extent可以暂时领先 Main；DFlash Full 通常落后于 Main。Provisional lead
不构成 committed checkpoint coverage。Rejected bytes 可以留在 partial page 中，但后续读取前必须被新的
canonical write 覆盖。

### 9.2 Fixed 与 transient K/V

以下 storage 不进入 growing pools：

| Resource | Owner |
|---|---|
| DFlash/DFlash2 local sliding-window K/V | fixed per-sequence StateImage |
| DFlash/DFlash2 boundary-local snapshot | fixed checkpoint StateImage |
| Vision/query temporary K/V | Program workspace |

DFlash/DFlash2 cyclic K/V 使用自己的 `CyclicKVCacheLayerView` 与 modulo/window 语义；它不持有 page ID、
block table 或 growing reservation。模型配置决定其 layer count、heads 和 window。

---

## 10. Consumer contract

### 10.1 Single-sequence view

Single-sequence growing-cache Op 使用 non-owning `PagedKVLayerView`：

```text
PagedKVLayerView
├── k_pages / v_pages
├── optional k_scale_pages / v_scale_pages
├── block_table          I32 [Nlogical]
├── head_dim
├── num_kv_heads
└── storage
```

View 只包含一个 layer 的 plane tensors 与一行 block table。它不包含 allocator handle、request identity、
reservation、ownership 或 frontier。

### 10.2 Batched view

Batched Op 使用 `PagedKVBatchLayerView`：

```text
PagedKVBatchLayerView
├── shared layer planes
├── block_tables         I32 [Nlogical, C]
├── head_dim / num_kv_heads
├── storage
└── table_rows[B]        separate Op input
```

`table_rows[b]` 为 compact row \(b\) 选择对应 active address-space row。Compact batch order、Engine lane
和 table row可以彼此不同。Per-row context length、valid columns和positions由 Op 的其他 typed inputs
提供。

### 10.3 Address translation

当前 \(P=64\)：

```text
logical_block = position >> 6
page_offset   = position & 63
physical_page = block_table[logical_block]
plane_row     = page_offset / tokens_per_row
```

随后使用 `4.3` 的 closed plane order计算元素地址，并检查该 plane row 在有效完整行范围内。
Batched consumer先用 `table_rows[b]` 选择 table row，再执行同一 translation。
K/V views 的现有 codec planes 使用 `tokens_per_row=1`。其他密度的 plane 由对应 Op 的 typed view
表达；storage geometry 本身不扩展 `PagedKVLayerView` 的数学合同。

Wrapper 验证 Tensor dtype、geometry、closed strides、table shape和execution envelope。Caller 保证：

- 本次可能访问的 logical blocks 已 materialize；
- read domain 不超过该 sequence 的 frozen valid frontier；
- writable K/V code 与 scale 在 frontier publication 前全部完成；
- selected table rows 在 GPU unit 内稳定。

### 10.4 Direct paged execution

Growing-cache Ops 直接消费 paged views。Page translation在 page/tile 粒度计算并复用，inner loop不解释
request、pool kind或allocator state。Kernel correctness不能依赖相邻 logical pages映射到相邻 physical IDs。

Paging 不引入 gather-to-contiguous cache或与 context 长度成比例的 staging copy。Route-specific tile、
split、warp和shared-memory方案可以独立优化，只要保持同一 logical Attention、persistent codec与上述
address contract。Op 的数值与性能准入规则见 [Op development](op-development.md)。

---

## 11. CUDA Graph 与 table publication

Plane bases与 block-table matrix base在 Engine lifetime内稳定。跨 replay变化的是：

- table content；
- `table_rows` selectors；
- positions、context lengths和valid counts；
- model state selectors。

Page IDs、request identity和physical contiguity不进入 graph key。

在需要新 mappings 的 execution unit 前，Program：

```text
materialize all required pages
  -> update host-side membership
  -> publish one contiguous table slice
  -> launch/replay consumer on the ordered stream
```

同一 boundary新增多个 pages或重新激活完整 membership时使用批量 publication。Mapping update必须先于
consumer，且 replay in-flight期间不得改写同一 row。

---

## 12. 核心不变量

1. 每个 Device page-group lease 在所属 pool 中至多承载一个 logical page replica。
2. 同一 pool 只组合共享 frontier、lifetime、page size与allocation语义的planes。
3. 一个 pool 的全部 planes 对同一 logical block 使用同一个 page-group ID，行密度不产生第二份页表。
4. Device occupancy 按 allocated leases与reservations计数；logical aliases不重复计费。
5. Logical page identity、content epoch与physical page ID彼此独立。
6. Valid frontier精确到原 token；plane 有效完整行数为 floor(coverage/tokens_per_row)。
   Page boundary不改变Attention或checkpoint语义。
7. Published checkpoint所需coverage不可被writer覆盖；任一logical page至多一个writer。
8. Shared full pages immutable；non-aligned writable tail先建立private COW page。
9. Host/Device replacement在copy与epoch/coverage验证完成后才发布。
10. 普通 unit 结算归还未用容量；恢复 reservation 跨 chunk 保留至真实新进展或终态。
11. 一个GPU execution unit内membership、block tables、replicas与read frontier稳定。
12. Inactive address space不占execution row；execution row不拥有logical pages。
13. Main与backend pools分别reserve、materialize、commit和truncate。
14. Growing-cache consumer只通过paged view和block table访问KV，不取得allocator或ownership authority。
15. Kernel correctness不依赖physical page ID连续性，也不通过gather建立request-contiguous KV。
16. Checkpoint可复用性由完整target continuation证明，KV page存在本身不构成hit。

---

## 13. 实现位置

| 职责 | 主要位置 |
|---|---|
| Device page pools、reservations与execution tables | `src/core/paged_kv_cache.*` |
| closed K/V data/scale plane schema | `src/core/paged_kv_storage.h` |
| 统一 Host backing、typed KV layout 与 allocation | `src/core/host_context_arena.*`, `src/core/host_kv_arena.*` |
| logical pages、replicas 与 references | `src/models/qwen3_5/program/storage/logical_kv_store.h` |
| address spaces、目录与 views | `src/models/qwen3_5/program/storage/kv_address_space.h` |
| history 与 checkpoint 生命周期 | `src/models/qwen3_5/program/storage/checkpoints.cpp`, `sequence.cpp` |
| Host extent membership | `src/models/qwen3_5/program/storage/host_kv_store.h` |
| unit 许可与上下文事务 | `src/models/qwen3_5/program/planning/request_plan.cpp`, `transactions/` |
| model pool layout与capacity curve | `src/models/qwen3_5/program/planning/startup.cpp` |
| public paged consumer views | `src/core/paged_kv_cache.h` |
| growing-cache Ops | `include/ninfer/ops/`, `src/ops/` |

Exact model state 和 backend mathematics 见
[Qwen3.5 model](qwen3_5-model.md)与 [DFlash](dflash.md)；persistent KV codec 和 causal consumer
numerical contract 由上表中的 growing-cache Ops 定义。路径用于定位当前实现，不把文件或类名本身提升为
外部接口。
