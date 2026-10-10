# Qwen4Exp：模型执行与状态提交

本文维护 Qwen4Exp 的层级调用、跨 Op 表示边界、Main/MTP 特征和可恢复状态，以及
[当前执行进度](#implementation-status)。架构、实例参数、权重表示与驻留方案见
[模型与权重](qwen4_exp-model.md)；PLE 的 hash、NF4 行解码、读取服务和设备交接见
[PLE I/O](qwen4_exp-ple-io.md)。

单个 Op 的完整公式、支持域、alias、workspace 和执行约束由 `include/ninfer/ops/`
中的合同维护，本文只规定模型如何组合这些计算。[Engine 架构](engine-architecture.md)
拥有请求事务与发布规则，[Paged KV](paged-kv-cache.md)拥有物理页、replica 和迁移机制。
原生 Qwen4Exp 模型执行尚未接入；下述层流程和提交关系是集成时须落实的设计。

## 1. 共同轴、前沿与所有权

| 符号 | 含义 |
|---|---|
| `T`、`B` | 一次调用每序列的物理列数、实际紧凑 batch 行数；`T>0`，`B=1..8` |
| `M` | 展平列数 `T*B`，`column=t+T*b`；HC 和普通 Linear 消费这个轴 |
| `n[b]` | 本窗口的有效输入前缀长度；Device `valid_columns[B]` |
| `F`、`F0` | 已执行输入前沿、候选窗口的基准前沿；均在原 token 域 |
| `R`、`u[b]`、`a[b]` | 记录容量、连续已记录列数、最终采用的输入列数；`0<=a<=u<=R` |
| `S` | StateImage 槽数；state slot、resident lane、batch row、KV table row 是独立身份 |

Main 和 MTP Backend 各自维护前沿，模型 bridge 建立两者关系。接受长度 `a` 按实际
输入列计数，由 Program 从后端接受结果和 Frontend decision 换算，不能直接使用输出
token 数、预取进度或 ledger 中尚未执行的下一输入。

连续 `[H,T,B]` 与 `[H,M]`、`[H,4,T,B]` 与 `[H,4,M]` 可以零拷贝改 view；合并投影的
子范围保留真实 parent strides。HC 的四条 stream 是 feature 轴，不使 GDN/QSA/MoE
的 token 数或 head 数乘四。MTP 的共享 `fc_hidden` 独立作用于四条 stream 时，Linear
列数才是 `4M`。

HC/Linear 对全部物理列计算，Program 为 padding 提供有限安全输入。QSA/PLE 独立
合同允许 `n=0`；现有 GDN 有效行要求 `1<=n<=T`，因此 Main 在 forward 开始前排除
空行，并在全部层保持同一紧凑成员集合，不在 GDN 层中途单独重排。

Model 持有只读参数和权重 backing。Program 独占可变 state、records、激活、workspace、
页表及 Graph，决定槽选择、前沿和最后消费者。Ops 接收非 owning views，只计算和
读写合同指定的区域；不分配持久设备资源，也不采用或发布请求状态。

## 2. 完整层流程与 HC 边界

Main 的 48 层中，零基层号 `l%4==3` 使用 QSA，其余使用 GDN；每层的 attention block
与 MoE 各有一组 HC。PLE 仅位于零基第 1 层的 attention HC 之前。MTP 有自己的
QSA/MoE 层和 HC 参数，无独立 PLE。具体几何与权重目录见模型文档。

```text
token embedding（先完成视觉特征替换）
  → hc_expand_norm → 四 stream residual h、第一组 normalized z
  → attention hc_mix(z) → block input x、FP32 injection j
  → GDN 或 QSA → block output y
  → hc_combine_norm(h,y,j,MoE norm) → 新 residual、MoE normalized
  → MoE hc_mix → routed experts + gated shared expert → MoE output
  → hc_combine_norm(...,下一层 attention norm) → 下一层
  → 最后一层 combine + final norm → h_final、z_final
  → final hc_mix(z_final) → sample_hidden → 共享 LM head
```

[HC 合同](../../include/ninfer/ops/gated_residual.h)规定 grouped offset norm、低秩
mixing、injection 和 residual 更新的完整数学。模型侧保留以下关系：

- `hc_expand_norm` 逐 word 将 embedding 展开到四条 residual stream，各 stream 使用
  自己的 affine；后续 combine 使用原 residual，不使用 normalized 值替代它。
- 普通 Mix 将 normalized streams 变为一个 2560 维 block 输入和四个 FP32 injection
  系数。系数已经是 `2*sigmoid(logit/4)`；combine 直接消费，不再次激活或缩放。
- `hc_combine_norm` 同时输出 BF16 residual 和下一组 HC 的 normalized streams。
  下一次 norm 必须消费实际产生的 BF16 residual words；下一组 norm 参数与本组
  injection 属于不同参数。寄存器内保留该 BF16 表示不要求额外写回再读取。
- Final Mix 没有 injection 或后续 combine。其 2560 维输出直接进入 LM head，
  不再增加独立 final RMSNorm；`h_final` 是 final norm/mixing 之前的完整 residual。

模型可以在 forward 内暂留 `(h,y,j)`，直到下一组 norm 融合 combine；这三个 borrowed
views 仍须存活到实际消费。PLE、末端特征交接或 checkpoint 需要完整 residual 时，
先物化该边界，不能保存未解释的 pending tuple。进入 PLE 的具体衔接是：

```text
第 0 层 MoE → hc_combine → 完整 BF16 residual
  → PLE Gate / ple_conv_add_norm → h_ple、零基第 1 层 attention normalized
  → attention hc_mix → 本层 GDN
```

MoE 消费 HC 产生的同一个 `x`：router 选择并归一化 top-10 routed probabilities，
routed expert 加权和再与独立 sigmoid 门控的 shared SwiGLU 相加，得到一个 block
输出供 HC 回注。路由/shared 合并 parent 不改变这些依赖；不同 expert 的输入列集合、
NVFP4 标定和驻留时机仍须由完整 MoE 执行合同处理。

### 2.1 Main/MTP 的两种末端特征

HC 没有逐层 token 历史；只有模型选定的末端值进入 continuation。对 `F>0` 的 Main
恢复点，下面两个字段来自同一次实际执行的同一个 token `F-1`：

| 字段 | 表示与用途 |
|---|---|
| `main_sample_hidden` | BF16 `[2560]`；final Mix 输出，供继续采样 |
| `main_mtp_conditioning` | BF16 `[10240]`；启用 MTP 时保存的 pre-final-mixer residual |

两个字段共享恢复点身份、前沿和有效性，不能独立发布。Conditioning 不能由 final
normalized、block output 或 sample hidden 代替或反推。`F=0` 没有有效末端特征；
不启用 MTP 时，Main conditioning 在 final norm 消费后可作为临时激活释放。

Verify 保留 `[2560,T,B]` sample window 和启用 MTP 时的 `[10240,T,B]` conditioning
window。`a>0` 使用同一 selector 选择两个窗口的 `a-1` 列；`a=0` 保留旧恢复点，
不以负 selector gather。可复用
[accepted-hidden 选择合同](../../include/ninfer/ops/speculative_round.h)分别复制两种宽度，
保留原 words，不重跑较短投影或 final Mix 来重建。StateImage 的 Device/Host layout、
clone、checkpoint 和 restore 必须显式表达这两个不同字段。

MTP 第一步消费 Main conditioning，后续 draft step 消费前一步 MTP 的 pre-final
multistream。初始化中，`pre_fc_norm_hidden` 沿完整 10240 维归约，不是四组 HC norm；
之后共享 `fc_hidden` 作用于四条 stream。Embedding 分支独立投影，并通过无 injection
的 `hc_combine_norm` 做 unit-weight broadcast-add 和第一组 HC norm，不需要构造
全一 injection tensor。

每步 MTP 产生 2560 维 sample hidden 供 head、10240 维 multistream 供下一步。
多个待定前缀需要的候选版本保留到选择结束。规范 checkpoint 位于模型提交边界，保存
Main 末端 pair 和已提交 Backend KV/QSA tail；未决 draft 内部特征不单独成为恢复点。
Resume 从保存的 Main conditioning 和正确 Backend 前沿重新进入 bridge。完整
draft/verify/bridge 顺序与输入接受长度换算仍须在原生 MTP 集成时落实。

## 3. GDN 的三条执行路径

HC 输入 `x` 分别进入 QKVZ 与 AB 两个投影分支，后者通过
[gdn_control_proj](../../include/ninfer/ops/gdn_control.h)产生 FP32 `g/beta`。
本实例直接消费 HC 的 `x`，不使用带 input norm 的控制入口。Q/K/V 经四 tap 因果卷积
和 SiLU 后进入 [Gated DeltaNet](../../include/ninfer/ops/gated_delta_net.h)；Z 绕过卷积，
保留至输出 gate。

16 个 Q/K heads 服务 48 个 value heads，维度均为 128。Recurrence 调用传原 Q/K、
`normalize_qk=true` 和 `scale=1/sqrt(128)`，readout 使用当前 token 更新后的 FP32
state。出口随后执行 [sigmoid_gated_rmsnorm](../../include/ninfer/ops/rmsnorm.h)，其
gamma 是直接乘数，既不是 offset norm，也不是 SiLU gate；BF16 结果经 output Linear
得到 2560 维 `y`，再交给 HC combine。两个投影分支到 recurrence 的依赖由模型显式安排。

| 执行路径 | 输入与调用组合 | 状态结果 |
|---|---|---|
| Prefill、Replay、CausalScoring | `B=1` 的连续窗口；`gdn_input_proj → causal_conv1d_silu_split → gated_delta_net`，再走共同输出末端 | 整个窗口后的 conv/recurrent state；长 T 由 recurrence 选择实现，不保存逐 token 完整 state |
| Ordinary decode、单列 control | 实际紧凑 `T=1,B=1..8`；`gdn_input_proj_conv_snapshot → gated_delta_net_batch_update` | 每行 conv 与 recurrent 结果写入同一候选 state slot 的对应层 |
| Target verify | 原始物理 `T=2..16,B=1..8`；`gdn_input_proj_conv_record → gated_delta_net_replay_record` | 基准 state 只读，产生实际 records；接受前缀确定后调用 `GdnReplayFoldPlan` |

投影与卷积的独立、snapshot、record 入口及容量查询见
[GDN input 合同](../../include/ninfer/ops/gdn_input_proj.h)，独立卷积见
[causal convolution](../../include/ninfer/ops/causal_conv1d_silu.h)。分立投影的 BF16 出口是
公开边界；融合投影/卷积不额外固定一个私有 BF16 投影中间值。Record 与 snapshot 的
精确对应不意味着 prefill、不同 chunk 切分或不同物理窗口之间必须 bit-identical。
短窗口限制属于 record/fold 的接口域，不限制 prefill 的正 T。

### 3.1 State 与 exact fold

Main 每个 GDN 层在前沿 `F` 保存 BF16 `[10240,3]` 的卷积前历史和 FP32
`[128,128,48]` recurrent state，`F=0` 均为零。36 个 compact GDN layer indices
映射到实际 Main 层号；MTP 使用 QSA，不另建 GDN state。

Verify 保存四类实际记录：卷积前最新 history 列、卷积/SiLU 后且 L2 norm 前的 K、V、
以及 recurrence 实际消费的 FP32 `g/beta`。恢复不需要 Q/Z/block output，不从卷积
记录重算 K/V，也不重算控制投影。Program 保留产生它们的 `F0`、初始 state slot、
原始物理 `T/B`、batch row 身份和有效长度。

[Replay fold](../../include/ninfer/ops/gdn_replay.h)的 `rows[b]` 始终对应物理 record row
`b`，不能只压缩或重排 selectors 一侧。对 `a>0`，从该次基准重放 `[0,a)`，结果必须
逐 bit 等于**同一完整物理 record-producing block**的对应 snapshot；拒绝后缀不读取，
不能以接受宽度重跑 Linear 或另一条浮点 recurrence 替代。

`a=0` 的 GDN fold 是严格 no-op：不读 records/state，也不填充 destination。Program
应继续采用基准槽，或先将所需基准字段精确复制到独立 candidate StateImage。原位 fold
只有在旧基准已无其他恢复消费者时才能覆盖。跨多个调用的 records 按原窗口保存，必要时
借助中间状态锚点逐段 fold，不能改写成另一种 T/row layout。

## 4. QSA 三阶段、页组与 tail

Main 用 Q8 Q/gate parent 和 BF16 K/V/index parent，MTP 用一个 BF16 joint parent。
[qsa_input_proj](../../include/ninfer/ops/qsa.h)直接产生各自连续的 BF16 分段输出，按原
`T/B` 改 view 后依次进入三个独立 Op；输入投影和 O projection 位于三阶段之外。

| 阶段 | 消费与交接 |
|---|---|
| `qsa_prepare` | 投影、norm/RoPE 参数、位置和只读旧 tail → prepared Q/index-Q、主 KV、完整四 token 块的 compressed keys、实际 raw-key/坐标 records |
| `qsa_indexer` | prepared index-Q 和 compressed cache → Device I32 逻辑块号 `[512,T,B]` 与 counts `[T,B]` |
| `qsa_sparse_gqa` | prepared Q、原 gate logits、selection、已写主 KV → gated BF16 attention；之后 O Linear 产生 HC block output |

三阶段与 tail extraction 的完整数学/API 统一见 [QSA 合同](../../include/ninfer/ops/qsa.h)。
Indexer 输出的是原 token 域的四 token 逻辑块号，与物理 page ID 无关；Sparse GQA 直接
消费前 count 个 IDs 和主 cache，并加上该 query 未完成块的可见尾部，不需要 expanded
token list 或 gathered KV。各 query 只看自己的因果位置，提前写完的候选后缀不因此可见。
完整块参加 TopK，没有“当前块必须保留”的特例；index score 仅用于选块，不加入主 logits。

Cache ordinal 决定因果性、压缩分组和寻址，实际一维/三轴 RoPE 坐标只决定旋转。
跨调用压缩先合并旧 tail 与本次 raw index-K，按四 token 平均并保留合同规定的 BF16
均值 seam，再 norm/RoPE；旋转使用块首实际坐标。不能用已旋转/归一化的 keys 代替
raw pooling，或用 cache ordinal 猜测多模态块首坐标。

Prepare 的 Q/index-Q 出口为 BF16；主 K 的 norm/RoPE 结果直接进入 codec，没有额外
BF16 中间边界。Sparse GQA 从实际 cache 表示读取新旧 K/V；量化 value 的逆变换和全部
split 归约完成后才乘 sigmoid gate，最终输出 BF16。Norm/RoPE 之间、attention/gate
之间的私有 staging 不另成为公共数值边界。

### 4.1 五类 KV 与共同逻辑页

支持 `BFloat16`、`Int8Group64`、`Fp8E4M3Row256`、`Nvfp4Group16`、
`Fp8KeyNvfp4Value` 五个 profile，编码解释见
[KV cache 合同](../../include/ninfer/ops/kv_cache_append.h)。`BFloat16` 实际为 BF16 K /
FP16 V；compressed index keys 始终为 BF16，独立于主 KV profile。

每个逻辑页组覆盖 64 个原 token：主 K/V 各有 64 行，index plane 有 16 行，使用同一
block table、页引用、epoch、COW 和 offload 事务。原 token `p` 查 `table[p/64]`；
index block `g` 查 `table[g/16]`。Main 的 12 层位于 Main pool，MTP 的一层位于独立
Backend pool；两者不因几何相同共用地址空间或前沿。

在 `F` 下，某页有 `c` 个有效原 token，则 index 只有 `floor(c/4)` 个有效行，不能
另建可独立推进的 index cursor。准备可写未提交后缀，commit/rollback 改变 coverage；
已写或复制的后缀字节不自动有效。COW 在候选写入前保护全部可能覆盖的范围，迁移保留
整个页组所有 planes 的原 words，且对应 raw tail 完成后才能发布完整恢复点。具体
`tokens_per_row=1/4` 布局、Host packing 和传输规则沿用 Paged KV。

### 4.2 Tail、记录与 Graph

每层的规范 tail 保存前沿 `F` 所在未完成块的 `F%4` 个 raw BF16 keys，最多三列；
其余列为 +0，同时保存该块首 token 的三轴坐标；完整块边界的 tail/坐标为零。
Prepare 只读旧 tail，记录实际 raw projection words 和坐标，Program 维护记录的
base position、offset、有效长度及连续性。

`qsa_tail_extract` 从旧 tail 和 `[F0,F0+a)` 的实际记录精确选择新规范 tail，不重新
投影、压缩或旋转。`a=0` 仍产生相同状态，独立 destination 可接收旧值；这与 GDN
fold 的严格 no-op 不同。任意接受前沿无需对齐四 token 块或 64 token 页。
例如 `F0=6,a=1` 时，新 tail 是 raw 4/5/6，候选阶段已写的块 4..7 仍不可见。

Prepare → Indexer → Sparse GQA 同 stream 直接衔接，不将 counts/IDs 拷回 Host。
固定物理 `T/B`、views、storage 和资源上界后，positions、有效列、state selectors、
record offsets、page IDs 和 selection 都是执行时输入。同一捕获上界内允许 batch
行之间、同一行的各 query 之间有不同历史长度，包括全选和 inert 情况；host envelope
只承诺容量，不替代因果 metadata。Sparse GQA 在合同允许的 envelope 更新间保持
节点数/顺序兼容，Program 使用既有 Graph update 机制，不复制 Op 的私有路由阈值。
依照 [Indexer 合同](../../include/ninfer/ops/qsa.h)，可见完整块数 `C<=512` 时在同一
kernel 内全选并跳过评分；混合 batch 不需要 Host 读回长度或另启短上下文拓扑。

## 5. PLE 的 GPU 交接与规范状态

Token IDs 已知后可提前 hash、读取和 NF4→BF16 解码，得到 `E[2560,T,B]`。它不依赖
上一层 hidden；K/V Linear 的输出与 PLE 前物化的 residual 在 `ple_gate` 汇合：

```text
真实 raw IDs + 同前沿 raw-ID tail → PLE I/O → BF16 E → 合并 K/V Linear
K/V + 完整 HC residual → ple_gate → BF16 U、BF16 N records
U + N records + 只读旧 history + residual
  → ple_conv_add_norm → BF16 h_ple、下一组 HC normalized → hc_mix
```

[PLE 合同](../../include/ninfer/ops/ple.h)维护 Gate、ConvAdd/Norm 和 StateExtract 的
完整公式与 views。K/V 子范围保留 `[12800,M]` parent 的真实 strides；U 与 N 是不同
的公开 BF16 值，NormConv 必须消费实际 U words，N 直接写入正式 records。卷积以 N
为输入，四 tap 的时间偏移为 `-9,-6,-3,0`，随后 SiLU 与 residual/U 相加。
下一 HC norm 消费实际 BF16 `h_ple`，同时保留该 residual。Gate → Conv 之间必须
建立完成依赖，不能让不同 CTA 假定其他 CTA 已发布未来需要的 N。

Main 前沿 `F` 的 PLE 规范状态包括最近九列 BF16 N history，以及 Host 上最近两枚
**已执行原始输入** ID。新序列负位置分别为 +0/EOS；EOS 只改变 hash 历史取值，
不清空卷积 history。视觉位置仍消费 raw token ID；padding 不成为时间步。PLE 不增加
独立位置 cursor 或 KV pool。

`ple_state_extract` 按 `a` 从旧 history 和实际 N records 精确取得新九列历史；
`a=0` 会复制完整旧 history，包括独立 destination，`a>=9` 只需接受记录的最后九列。
Raw-ID tail 由同一个 `a` 从旧两枚 ID 与接受输入更新，不从已发布输出尾部猜测。
恢复不重算 Gate/norm/conv，也不重新读取 SSD。

Record offsets 是当前窗口在记录 buffer 中的列位置，不是绝对 token position。
单次提取需要从其基准前沿开始的连续记录，较长事务可用中间状态锚点与分段记录。
`E` 被 K/V Linear 消费后可归还其传输槽；N records 和最终提交所需 IDs 的寿命更长，
不能随 I/O ticket 或 staging 释放。生产 ticket、H2D/event 与 Graph A/B 的精确交接
遵循 PLE I/O 文档；文件读取和 Host 等待不放进 CUDA Graph。

## 6. 统一提交、恢复与资源寿命

同一 Main 输入前沿必须同时覆盖：GDN conv/recurrent state，QSA 主 KV/index coverage
及 raw tail，PLE conv/raw-ID tail，以及有效的 Main 末端特征。MTP Backend 状态按
bridge 确定的自身前沿纳入同一请求事务。任一层或部分 planes 完成不代表整个恢复点有效。

```text
Program 预留 state/record/page 资源，完成必要 COW 与输入准备
  → 执行完整候选 forward，保留实际 records 和末端窗口
  → 后端验收与 Frontend preview 确定每行输入接受前缀 a
  → GDN fold、QSA/PLE extract、末端 pair 选择到 candidate state
  → Program 一次采用完整 StateImage、KV coverage 与执行 provenance
  → 提交输出及预算，发布事件，释放已无消费者的候选数据
```

取消或零接受保留基准恢复点。外层 PendingBatch 仍可能截断时，内部已采用 working
state 不能使外层基准、records 或特征窗口失效；原位更新须等旧值的全部恢复消费者结束。
普通 prefill/Replay 可在 chunk 采用范围明确后提取 QSA/PLE candidate state，再复用
记录，无需保存整个 prompt 的记录。完整 checkpoint 还需要同前沿的 GDN state；当前
长窗口只产生末端 recurrent state，因此保存点对齐实际执行边界。需要中间保存点时应在
该处划分执行窗口，不能仅凭 QSA/PLE 的任意前缀提取宣布完整模型 checkpoint 可用。

Snapshot 的 Device/Host copy、clone 和 restore 保留原执行 words；所有组成部分的
身份、前沿、coverage 和 readiness 一致后才发布。Replay 则重新执行已经接受的输入，
按 Engine 合同保留请求与输出语义。不可变 PLE row cache、expert 权重缓存不属于请求
checkpoint；缓存命中、预取、搬运和 eviction 均不推进输入前沿。

| 数据 | 最后消费者 |
|---|---|
| HC residual / normalized / injection | 对应 combine / Mix / block 后的 combine |
| GDN Q/K/V、g/beta、Z | recurrence 与 record 写入；Z 另外保留到 gated norm |
| QSA prepared Q、gate、selection | Sparse GQA；raw records 单独保留到全部 tail 提取结束 |
| PLE E、K/V、U | K/V Linear、Gate、Conv；N records 单独保留到全部 history 提取结束 |
| 基准 state、候选 records、末端特征窗口 | 最后一个接受、截断、取消或恢复决定及其设备消费者 |

Planner 对顺序互斥 scratch 取峰值，对跨阶段存活数据显式保留，使用各 Op 的权威区间
容量查询，不复制私有阈值。Host 函数返回不表示 Device 已结束使用。Graph 按实际
exact-B topology 和稳定 backing 规划；[GDN fold](../../include/ninfer/ops/gdn_replay.h)
的 host row descriptors 在 capture 时按值固定，改变它们需新 capture，不能套用
QSA/PLE Device metadata 的 replay 规则。

### 6.1 Planner 的基础 payload 预算

以下是本实例的逻辑 payload；state 副本数、页取整、对齐、workspace、Graph、传输槽和
容量余量由实际 planner 另计。每份 canonical state 都对应一个完整恢复点。

| 对象 | Payload |
|---|---:|
| Main 36 层 GDN recurrent state，FP32 `36×48×128×128` | 113246208 B = 108 MiB/状态副本 |
| Main 36 层 GDN conv history，BF16 `36×10240×3` | 2211840 B = 2.109375 MiB/状态副本 |
| Main 全部 GDN canonical state | 115458048 B = 110.109375 MiB/状态副本 |
| PLE BF16 九列 history；Host raw-ID tail | 184320 B = 180 KiB/状态副本；另 8 B metadata |
| QSA raw tail 与块首三轴坐标 | 780 B/层/状态副本；Main 12 层，MTP 另 1 层 |
| Main sample hidden 与 MTP conditioning pair | 25600 B = 25 KiB/状态槽；不启用 MTP 时只保留 5 KiB sample |
| 一份 HC residual 或 normalized buffer | `20480*M` B = 20 KiB/物理列 |
| 一份 block input/output 或 sample buffer；injection | `5120*M` B；`16*M` B |
| Main 末端 pair 候选窗口 | `25600*T*B` B；`T=16,B=8` 时 3.125 MiB |

主 KV 与 index 的每层共同页组预算如下。每页覆盖 64 个原 token，表中已包含该层
4096 B 的 BF16 compressed index plane；压缩没有减少主 K/V 的历史 coverage。

| 主 KV profile | 每层页组 bytes，含 index | Main 12 层 bytes/原 token |
|---|---:|---:|
| BFloat16 | 135168 | 25344 |
| Int8Group64 | 71680 | 13440 |
| Fp8E4M3Row256 | 70144 | 13152 |
| Nvfp4Group16 | 40960 | 7680 |
| Fp8KeyNvfp4Value | 55552 | 10416 |

MTP 使用“每层页组”列按一个 Backend QSA 层另算，其物理页数依自身前沿与 provisional
lead 规划，不假定与 Main 相同。最终容量、Host reservation、迁移 bytes 和资源汇总从
同一 expanded plane schema 推导；这张表不代替运行时布局查询。

<a id="implementation-status"></a>

## 7. 已完成能力与剩余工作

本节是 Qwen4Exp 执行进度的统一入口；完成范围按当前代码和已验收结果区分。独立 Op
验收不等于完整模型或端到端性能验收。

| 已完成能力 | 范围与权威入口 |
|---|---|
| Artifact/converter 与纯 Linear | Qwen4Exp 配置、Binding/Use、完整 Text/Vision/MTP 权重产物、NF4 持久编码/分页布局及 20 项矩阵支持；详见模型文档 |
| GDN 输出 norm | `sigmoid_gated_rmsnorm`，与 SiLU gated 形式共用 RMSNorm 家族；已通过数学与 Op 性能验收 |
| GDN control | BF16 `[96,2560]` 合并 AB projection、权重准备、FP32 controls 和容量查询；已通过数学与 Op 性能验收 |
| HC 基础计算与 Mix | 五种基础形式、block/final Mix 及区间容量查询，保留 residual seam；已通过数学与 Op 性能验收 |
| GDN input 与 ReplaySSM | Q8 `[16384,2560]` 的 projection/snapshot/record 和容量查询；`(36,16,48,10240)` 全层 fold 已实现 |
| QSA | 分段输入投影、五类 KV 的 Prepare/Sparse GQA、Indexer、tail extraction 已实现 |
| Paged KV 机制 | `tokens_per_row=1/4` 的共同逻辑页及 Host/Device plane 布局、传输和相应覆盖语义已有通用实现；模型 pool/StateImage 接入仍待完成 |
| PLE GPU 局部能力 | 共享合同、`ple_conv_add`、`ple_conv_add_norm`、`ple_state_extract` 已实现；Gate 目前只有声明 |

真实剩余工作如下；已完成项不再作为重新开展独立调优活动的前置任务。

1. **PLE Gate、hash、运行时 NF4 解码与读取服务。** 实现 `ple_gate`；补齐纯 Host
   hash、按实际 divisor 构建的 NF4→BF16 LUT/行解码；接入 cold backing、bounded
   cache/queues、ticket、pinned E 和 H2D/Graph 交接。持久 NF4 codec 与独立 SSD
   实验已经存在，但不构成生产 reader 或完整 PLE 能力，接口设计见 PLE I/O。
2. **NVFP4 MoE 与 expert streaming。** 在现有 per-expert bank 表示上落实路由/shared
   计算、完整 Op 边界、A16/A4 及 activation divisor 复用、cache lease/eviction、
   计算前 readiness、搬运粒度和重叠。Grouped/fused 策略与驻留调度仍需细化，不重新
   下放成一轮纯 Linear 任务；PLE 与 expert H2D 的实际竞争也须在完整 schedule 评估。
3. **原生模型集成。** Qwen4Exp binder、驻留、Parameters、planner、StateImage、层
   编排和 Program 尚未接入。先贯通 Text prefill/ordinary/control/Replay/scoring，
   连同统一提交、checkpoint/offload、取消和 eager/Graph；再完成 Vision 的特征替换
   与位置交接，以及 MTP 的 draft/verify/bridge、双特征选择与恢复。既有 Qwen3.5
   DFlash/DFlash2 不构成 Qwen4Exp 支持声明。
4. **完整路线资格与测量。** 原生模型就绪后验证组合数值/质量、MTP 接受率、资源峰值、
   SSD/PCIe 与计算重叠及端到端延迟。当前 Op 或 I/O 证据不能替代这些结论。

后续实现沿用 [Op 开发规则](op-development.md)及 [测试](../../tests/README.md)、
[benchmark](../../bench/README.md)已有入口。数值计算依据独立数学 oracle；exact
record/fold、feature selection 和状态搬运依据对应实际执行的逐 word/bit 合同。
跨层测试关注同一接受前沿下全部状态的一致性，不以 kernel parity 或可读模型输出替代。
