# Qwen4Exp PLE：Host hash、NF4 查表与异步 I/O

本文定义待接入生产的 PLE Host 准备合同：精确 token hash、NF4→BF16 解码、不可变磁盘
backing、有界读取服务，以及交给 Program 的窗口与完成通知。已有 NF4 artifact
format/layout 不等于已实现生产 reader；独立 SSD 实验也不承担这一角色。

[模型与权重](qwen4_exp-model.md)维护实例几何与量化选择；
[数值格式](tensor-formats.md#35-nf4_g16_e4m3fn)与
[存储布局](storage-layouts.md#6-row_paged_g16_4k_v1)维护 NF4 的通用表示。
GPU Gate、ConvAdd/Norm、StateExtract 的数学与接口以
[ple.h](../../include/ninfer/ops/ple.h)为准；整层调用、统一 StateImage 和 accepted-input
frontier 见[执行与状态](qwen4_exp-execution.md)。本文不再复制这些 GPU 声明。

## 1. 准备边界与所有权

本实例只在零基第 1 层入口执行 PLE。每个 Main 输入位置按固定 head 顺序取 16 行，
每行 160 个值，准备连续 BF16 `E[2560,T,B]`，也可作为 `[2560,T*B]` 交给既有合并
K/V Linear。原生 MTP 没有独立 PLE，但其候选进入 Main 验证时仍需准备对应的 E。

首版采用 `O_DIRECT + io_uring`、行/页去重、原始行缓存和 CPU LUT 解码。NF4 表留在
V3 `.ninfer` 文件集合内，按需取行；不要求整表驻留 RAM/VRAM，不另生成运行时权重副本。
出口固定为规定的 BF16 words，后续若改用 GPU 解码，须独立验证同一出口。

| 所有者 | 责任 |
|---|---|
| Artifact / Model | 验证并持有只读文件 backing、冻结的文件区间映射、shard 描述、divisors、hash controls 和派生 LUT |
| Core Host 数值辅助 | 从明确的 divisor 构建精确 LUT，从紧凑行字节解码 BF16 words；不持有请求或文件状态 |
| 模型 Host hash | 从原始 IDs、两枚 raw tail 和控制值产生有序全局行号 |
| Program | 拥有读取服务、可变 cache/queues/buffers、CUDA 槽、ticket、预取和完整 unit 的提交边界 |
| I/O worker | 推进其 Host 读取、解码、缓存和完成队列，不提交 CUDA，不修改请求 ledger、frontier 或 StateImage |

文件 I/O 不进入数学 Op 或 CUDA Graph。GPU Op 只收到就绪的 E 或 K/V，不消费文件路径、
fd 或 artifact 对象 ID。E 是 lookup 的激活输出；CPU 解码不改变持久权重的数值格式或布局。

## 2. token hash 的精确语义

输入必须是原始 token IDs。视觉特征替换普通 embedding 后仍使用相同的原始 IDs，不能从
hidden、视觉 embedding 或输出字符串反推。Model 从 `ple_hash_controls` 的 280-byte
Resource 无损绑定 `multipliers[3]`、`head_vocab_sizes[16]`、`head_offsets[16]` 共 35 个
INT64 words，EOS 从 config 取得；本实例 EOS 为 248044。资源存储合同见
[模型与权重](qwen4_exp-model.md)。不经 FP32 转换，也不按 seed 重新生成控制数组。

令当前位置的 raw ID 为 x0，前一、前二个 raw ID 为 x1、x2；不存在的位置补 EOS。
若 x1=EOS，则只对本次查询令 x2=EOS。其他情况下保留原始 x2，包括它本身为 EOS 的情况。
当前位置 x0=EOS 时仍使用此前历史，分段效果从 EOS 后的位置开始。

```text
u2 = word64(x0 * multiplier[0]) XOR word64(x1 * multiplier[1])
u3 = u2 XOR word64(x2 * multiplier[2])
g[h] = positive_mod(signed64(h < 8 ? u2 : u3), head_vocab_size[h])
       + head_offset[h]                         // h=0..15
```

乘法是 unsigned modulo-2^64 word 运算。`signed64(w)` 在数学上为 w（w<2^63）或
w-2^64，必须显式保留该解释，不依赖 C++ signed overflow。`positive_mod(z,m)` 是
`[0,m)` 内的余数；例如先计算有符号余数 r，r<0 时再加正整数 m。
Head vocab sizes 必须为正，offset 与 size 确定的行区间必须落在已绑定的整张表内。
控制乘法/XOR 保留完整 64 bits；本实例最终 g 小于 320001536，可存为 U32。

Host hash helper 的输入为一个序列的连续已知 IDs 和紧邻起点的 raw tail
`[id(F-2),id(F-1)]`。缺失位置由调用者填 EOS，不提前改写 EOS 之前的 raw tail。
输出恰为 `16*input_count` 个 U32，`row_ids[16*t+h]` 表示位置 t、head h 的全局行号，
次序是 2-gram heads 0..7，再接 3-gram heads 0..7；空输入对应空输出。
函数同步完成，只读输入/参数，写调用者提供且与输入不重叠的输出，不分配输出存储或保留历史。

窗口内后续位置使用本窗口更早的 raw IDs，不能反复套用窗口起点 tail。分段调用和一次
调用同一串 IDs 必须给出相同结果。追加候选时，服务使用追加起点之前的两枚 raw IDs；
去重及页映射在生成完整、有序的 hash 输出之后进行。

F 始终是实际 materialize 的 Main 输入前缀长度，与预取进度或已发布输出数不同。
最终 raw tail 从 Program 实际采用的输入前缀确定；它与 PLE 卷积状态共同属于同一个
prefix identity，详见[执行与状态](qwen4_exp-execution.md)。EOS 的上述规则只改变 hash，
不清空 GPU 卷积历史。

## 3. 精确 NF4→BF16 LUT 与行解码

持久格式为 `nf4_g16_e4m3fn`，固定 codebook 的 binary32 words 由
[数值格式](tensor-formats.md#35-nf4_g16_e4m3fn)和
[core/nf4.h](../../src/core/nf4.h)定义。每个 shard 绑定其实际存储的正有限 FP32 divisor
d；lookup 的出口是该精确表示值直接按 round-to-nearest-even 舍入得到的 BF16 words：

```text
lut[shard][scale_word][code] =
    BF16_RNE(codebook[code] * E4M3FN(scale_word) / divisor[shard])
```

LUT 必须以独立精确/高精度 reference 构建并逐项验证。未经证明的“先 FP32，再 BF16”
双舍入不能替代此结果；将来 CPU/GPU 两条 lookup 路线也必须逐 word 一致。
所有 16 个 code 有效，scale word 只允许 `0x00..0x7e`，包括正零；拒绝负数、负零和 NaN。
结果为零时保留 codebook 值的符号：code 7 得到 +0，负 codebook 值乘正零 scale 得到 -0。
本实例所有有效 LUT 项须为有限 BF16。

每张 LUT 保留 128×16 个 uint16 words，占 4096 B；128 shards 合计 **512 KiB**。
Scale slot 127 只是预留空间，不是有效编码，解码时不得查用。LUT 在加载阶段构建后由
Model 作为不可变派生数据持有，Program 的服务借用它。

通用行解码的 K 由输出元素数确定，要求 K>0、K%16=0、`K/2+K/16<=4096`。
输入长度必须恰为 `K/2+K/16`：前 K/2 bytes 是 low-even packed codes，后 K/16 bytes
是按 K group 顺序排列的 scales。输入允许任意字节对齐；输出连续，且与输入、LUT
不重叠。函数只读输入和 LUT、完整写出该行，非法 scale 报错；它不创建线程、提交 CUDA
或保留跨调用可变状态，SIMD 和分块方式由实现决定。

本实例为 90 B → 160 个 BF16 words。服务先按全局行号选择 shard/LUT，再直接写入
`E[(b*T+t)*2560+h*160 : ...+160]`。拼接 head 次序与 hash 输出相同。
无效尾列不读取权重，其 E 区域精确写 BF16 +0；H2D 一次发送完整、连续的 batch E。

## 4. 磁盘 backing、行寻址与错误

### 4.1 启动绑定

表由 128 个原始 shard 组成，每个 `[2500012,160]`。Shard 是全局表的连续行段，
不是一个 hash head 对应一个 shard。每 shard 保留自己的 divisor，不重新合成 global
divisor。对于全局行 g，`shard=g//2500012`、`row=g%2500012`；对象内行地址使用
[`row_paged_g16_4k_v1`](storage-layouts.md#6-row_paged_g16_4k_v1) 的几何计算。
本实例每行 90 B，每 4 KiB 数据页容纳 45 行，末页为 37 行，divisor 在完整数据页之后。

Artifact/Model 需要补充长期 disk placement，验证对象几何、layout、encoded size、
文件归属和范围，并冻结 V3 逻辑区间到真实文件区间的映射。启动只读取所需目录、280 B
controls 和 128 个 divisors，不扫描或 materialize **27.126954 GiB** 的整表。
当前通用 Materializer 的完整 Host/Device placement 不能直接代表这种 cold backing。

Model 持有长期打开的只读文件及不可变 shard 描述；Program 的服务借用这个 backing。
不得借用加载后将销毁的 Reader，也不能用未 materialize 数据的 Host WeightParent
指针代替磁盘表。启动核对 direct-I/O 支持与对齐要求，每次 miss 不再打开文件、解析
JSON 或扫描对象目录。运行期只有不可变权重读，没有 SSD 写回。

### 4.2 从逻辑页到实际读取

PLE 行的对象内地址先加到 V3 逻辑 payload 区间，再经冻结的区间表映射到真实文件，
不能把逻辑 object offset 直接当作文件偏移。当前 recipe 的对象起点、payload 起点和
非末卷边界均按 4 KiB 对齐，因此常规布局的一行只需要一个实际文件页；shard 跨续卷
也不需要重新打包。

其他合法 V3 分卷位置仍须处理：一个逻辑数据页可拆成多个真实 file segments，每段
扩展到符合 direct I/O 的物理范围，再收集需要的有效字节。非对齐分卷可能涉及多个
物理页；全部对齐开销计入真实读取统计。EOF 短读只有在覆盖全部所需有效字节时才可
接受，不能用补零掩盖缺失数据。

读取范围、短读及 scale words 在消费时检查。无效 divisor/scale 或读取失败交给
Program 的 unit 错误路径，不返回伪造的零 embedding，不发布部分 HostReady，也不
静默改为整表 mmap/materialize。页 padding 不参与数值解码，不能混同逻辑零行。

## 5. 有界读取服务

### 5.1 窗口与 cache

```text
预留固定 T/B 窗口，复制 identities、epoch、F0、raw tail 和已知 IDs
提供或追加 IDs：hash → 行去重 → 原始行缓存
    → miss 映射到物理页 → 合并同页及已有在途读 → io_uring 提交/完成
    → 收集紧凑行 → 按 head 顺序 CPU LUT 解码
seal 冻结有效长度；全部有效 E 完成且无效尾列置零 → 发布 HostReady
```

服务属于 Program 的私有执行资源，不是公共 Engine API。所有队列、buffer、token/row
maps 按启动容量分配。首版起始配置为 **64 MiB 原始行缓存、128 个在途读、两个完整
窗口槽**，允许在预算内调整；这些不是模型数学常数。

物理在途读按 `(file, aligned_offset, length)` 合并，可服务多个窗口。行缓存按已绑定
模型内的全局 g 索引，g 唯一决定该行及 divisor；复用 cache 或读到错误草稿的行不推进
模型状态。64 MiB 包含行数据、索引及替换 metadata，可设为零。页 buffer 主要承担
短期 I/O，长期缓存保留实际使用的 90-byte 行。命中行先复制进本窗口收集区，不能保留
可能随逐出失效的裸 cache 指针。

首版保持小块并发读，不为凑顺序大块访问而读取远处未请求的数据。真正相邻的必要范围
以后可以合并，但须记录读放大并重新测量；queue depth、合并阈值和 cache associativity
是服务内部策略，不属于 NF4 格式或数学 Op。

无可用窗口时 prepare 返回“暂不可准备”，不推进请求状态。Program 推进已有完成或
让出可选 lookahead；当前必需窗口优先，可选预取不能占满所有槽并阻塞当前执行窗口。

### 5.2 控制操作的语义

| 操作 | 合同 |
|---|---|
| prepare | 固定物理 T/B 和每行 identity，B=1..8、T>0；初始 IDs 可以只有 anchor 或一段前缀。成功后立即拥有必要输入副本；无槽返回不可立即准备，不表示 I/O 失败 |
| append | 只作用于尚未 seal 的 Preparing ticket；每次调用的 batch rows 互异，每行从已知前缀后连续追加且不超过 T。返回前复制新 IDs，不能改写既有 IDs |
| seal | 接受 B 个最终有效长度，满足 `0<=n[b]<=known_columns[b]<=T`；之后不可追加。被截去候选表示不再参与本窗口验证，不表示最终接受前缀 |
| poll / embedding | poll 不同步 GPU；只有 seal 且全部有效 E 完成后才可获得 HostReady 的 borrowed pinned embedding。具体失败原因交给 Program |
| cancel / drained | 丢弃未提交任务并排空该 ticket 的 Host writers；drained 只说明没有 I/O/CPU writer 引用，不证明 H2D 或 GPU 消费已经结束 |
| release | 正常或失败/取消路径均须由 Program 证明该代次全部 Host/CUDA 引用退役后释放；Preparing 即使暂时没有在途 I/O，也须先 cancel，不能作为已完成窗口释放 |

每次 append 即可发起相应行读取，不必等全部草稿产生。seal 截去的尾列必须置零，晚到
读结果不得再写这些 E 区域；HostReady 之前不发起 H2D。普通已知窗口可以 prepare 后
立即 seal。对同一 ticket 的控制调用只来自 Program worker。

### 5.3 ticket、generation 与取消

每行记录稳定 sequence identity、候选 epoch、起始输入 frontier、有效长度和实际 IDs。
Ticket 由物理 slot 与 generation 组成，lane、compact batch row 或 token position
不能独立充当身份。准备成功后不借用随后可能变化的 caller spans；stale generation
是 caller 错误，不能返回新代次数据。

```text
Free → Preparing（append / seal）→ HostReady → Uploading → DeviceReady → Consumed → Free
              ↘ Failed / Cancelled → drain 所有引用 → Free
```

Preparing/HostReady 由服务推进；从 Uploading 开始的 CUDA 状态由 Program 管理。

| 存储 | 最早可复用条件 |
|---|---|
| I/O slab | 对应磁盘操作完成，所有所需片段已收集 |
| pinned E | H2D 完成；取消也不能覆盖仍被 DMA 读取的源 |
| Device E | 最后消费者完成，通常为 K/V Linear，无需等到采样 |
| slot generation | 旧 I/O、CPU notification、H2D 及消费者引用全部退役 |

首版整槽 release 同时满足以上条件，不要求拆分 Host/Device ownership。prepare 固定
T/B、row identities 和 batch stride，seal 只改变有效长度。若调度改变 compact batch
成员或物理 shape，重新建立 ticket，复用的是行缓存/在途读，不能直接 reshape 旧 E。

取消时共享在途读仍可服务其他窗口，已提交读可排空再回收。旧 epoch 的晚到完成可以
留下正确的不可变行缓存，但不得向重用后的 lane/ticket 发布旧 E。销毁先停止新工作，
drain I/O/CUDA 引用，再释放 buffers、线程，最后才允许 Model file backing 结束寿命。

## 6. H2D、预取与 Graph 交接

I/O worker 用 release/acquire 完成通知发布不可变 HostReady。Program worker 验证
ticket 身份后，在 transfer stream 排入 H2D，再 record 当前 slot/generation 的 event。
消费者 stream 的 wait 必须引用**已排入 record 的当前代次**；未 record 的 event
不代表未来 SSD 完成，不能先等空 event 再由 Host 补依赖。
Copy/event 入队后即可排入 GPU wait，生产路径不必像微测量一样在 Host 等 H2D 完成。

Program 等待当前 GPU unit 时仍需推进 HostReady 和 transfer 提交，不能一直阻塞在
整个 unit 的 `device.synchronize()` 上，让已就绪的下一窗口无法上传。推进可位于
Program 的 unit 等待循环，Engine 的提交边界不变。

首版 Main forward 的交接位置为：

```text
prepare current E
Graph A：普通 embedding、第 0 层、PLE 前的 hc_combine
当前 HostReady → H2D → event record → consumer stream wait
Graph B：K/V Linear、PLE GPU Ops、其余 Main 层及末端特征
```

Graph A 可在等待 HostReady 时执行；E 已提前 DeviceReady 时直接衔接 B。
A/B 只拆 Main forward，后端 draft/verify/bridge 和约束采样 Finish 仍服从
[完整 unit 协议](qwen4_exp-execution.md)。Ticket 失败时不启动依赖它的 B，先排空
A 的引用，再走现有错误结算路径，不能复用上一窗口的 E。

Graph 不包含文件读取或 CPU 阻塞 callback。Device E 各槽地址稳定，首版为各槽建立
固定 consumer graph 绑定；不修改仍在运行的 graph 输入指针。Graph 数量、容量和
拆图增加的提交开销均计入 Program 规划与完整路径测量。K/V Linear 消费完 E 后可用
已有 external-record event 机制发布 Host 可观察的 completion，以便提前回收槽。

| 执行路径 | 发起时机 |
|---|---|
| Prefill / Replay | IDs 已知，当前 chunk 计算时准备下一 chunk，从一个 chunk ahead 开始 |
| 首个 prefill chunk | 输入及必要 metadata 就绪后尽早准备，可与视觉/绑定等独立工作重叠，单独测 TTFT |
| Ordinary decode | 输入 ID 可用即发起，主要利用 embedding 与第 0 层窗口；下一未知 token 不可假定 |
| MTP target verify | 先准备已知 anchor；draft IDs 逐步可见后追加，可与后续 draft 步骤重叠 |
| Forced control / scoring | 使用真实输入 IDs 与对应 frontier，经过同一 reader 和 PLE 路线 |

设备候选需要明确的小型 pinned IDs handoff 和完成事件，按步或成组发布，不能为每个
候选同步整个 GPU。它们只是可供预取的候选，不是已提交 token。若以后接入并行 masked
draft，须在最终候选 IDs 发布后成组准备，不能把 mask 或未确定中间值作为 Main 输入；
历史实验中的 DFlash 标记仅表示验证输入形状，不代表本架构已有该后端。

起始两槽覆盖当前/下一窗口；chunk 大小由完整 Program 容量和性能决定，不因实验最大
T=4096 而固定，也不预读整个长 prompt 或为历史 token 永久保存 E。首版提前的是 I/O、
CPU 解码及 H2D；K/V GEMM 是否前移须测量实际 GPU/传输竞争，不能凭另开 stream 推断
收益。Key norm 仍属于 `ple_gate`，拆分它需要重新明确 Op 和表示边界。

PLE H2D 必须在消费者之前有明确排队位置，不能让大量无依赖的远期 expert copies
挡住当前 E。Stream priority 不保证抢占在途 DMA，传输块大小和排队时机由实际 Program
schedule 协调；独立 I/O 实验未验证这种竞争。

E/ticket 的寿命与提交 records 分开。N 直接写入供 Conv 和 StateExtract 消费的 record
buffer，不额外保存同内容 N；它可能比 E 活得更久。外层 PendingBatch 仍可回退时，
基准 history、所需 N records 及最终提交用的原始 IDs 必须保留，即使内部 working
history 已前进。Program 的 ledger/PendingBatch 独立保有这些 IDs，不能把 I/O ticket
的副本当作提交时唯一来源。早回收 E 不释放 records 或 canonical state；分段 records、
任意接受前缀及全部模型状态的共同采用由[执行与状态](qwen4_exp-execution.md)规定。
Cache、ticket 和 staging 不进入 checkpoint；丢弃错误草稿也不要求清空正确行缓存。

## 7. 启动预算

令 `Mmax=max(计划 prefill chunk, verify width×并发数, ordinary 并发数)`，按实际最大
T*B 规划；当前 prefill 一次一条请求。它是 Program 调用形状的容量，不是 Op 的 T 上限。

| 项目 | 起始容量或公式 |
|---|---:|
| NF4 表 payload | 27.126954 GiB，磁盘 |
| Divisors / hash controls | 512 B / 280 B，Host metadata |
| CPU LUT | 512 KiB，Model 不可变派生数据 |
| 原始行缓存 | 64 MiB，包含 metadata，可配零 |
| 128 个 4 KiB I/O buffers | 512 KiB，注册的 Host I/O 存储 |
| 两槽 pinned E | `2×5120×Mmax` B |
| 两槽 Device E | `2×5120×Mmax` B |
| 单个合并 K/V parent | `25600×T×B` B |
| 单个 gated U | `20480×T×B` B |
| N records | `20480×R×B` B |
| 每份规范 PLE history / raw-token metadata | 184320 B / 8 B，另计通用 identity、frontier 和对齐 |

Mmax=4096 时两槽 E 分别为 40 MiB pinned Host 与 40 MiB Device；B=1、T=R=4096
时 K/V 为 100 MiB，U 或 N 各 80 MiB；B=8、R=16 的 N records 为 2.5 MiB。
此外必须计入最多 `16×Mmax` 行的收集区、hash/去重/在途引用 metadata、实际对齐与
跨卷读取存储、norm 输出和已有 HC workspace。每 token 的有效编码虽为 1440 B，
它既不是整个服务的容量，也不是实际磁盘流量。Prefill 不要求保存整个 prompt 的 N。

## 8. 独立 SSD 实验：结果与适用范围

保留的证据为 [2026-10-07 完整报告](../../profiles/bench/ple-io-20261007/REPORT.md)，
复跑方法和原始汇总从该报告进入。实验运行于 Ryzen 9 9950X3D、RTX 5090、驱动
617.14、CUDA Runtime 13.4.92，WSL2 Linux 6.6.87.1/ext4 虚拟磁盘；未确认物理 SSD
型号。单 I/O worker 使用 liburing 2.5、O_DIRECT、固定文件/buffer 注册和最多 128 个
在途 4 KiB 读。

数据为真实写满的合法合成 NF4 表，128×2500012 行，两个布局保存相同逻辑内容；
不是正式模型 NF4 artifact。Hash 使用真实 tokenizer、35 个原始 INT64 controls，以及
从仓库中英文文档和 C++/CUDA 文件取得的 1,836,692 个连续 token。每 trial 从空应用
cache 开始；64 MiB 配置为固定 4-way 原始行缓存、轮转替换，未做策略或容量调优。

以下是 `45 行/4 KiB + io_uring + CPU LUT + pinned H2D` 的完整加载时间，单位 ms，
顺序为 **P50 / P95 / 最大值**，没有过滤最大值。计时从 IDs 可用后的 caller dispatch
开始，包含线程唤醒、hash/计划、读取/收集、cache 填充/解码、H2D 和 Host 完成确认；
不包含初始化、校验、日志及人为间隔。

| 窗口 | 无应用行缓存 | 64 MiB 行缓存 |
|---|---:|---:|
| B=1,T=1 | 0.617 / 1.813 / 64.462 | 0.504 / 0.714 / 4.090 |
| B=8,T=1 | 1.318 / 1.558 / 5.135 | 0.931 / 1.253 / 4.654 |
| B=1,T=5 | 0.995 / 1.183 / 3.027 | 0.726 / 1.068 / 1.845 |
| B=8,T=5 | 5.181 / 5.544 / 9.259 | 2.301 / 3.411 / 5.987 |
| B=8,T=16 | 14.683 / 15.435 / 18.857 | 5.719 / 8.416 / 15.687 |
| Prefill T=256 | 19.726 / 23.216 / 26.242 | 9.782 / 16.281 / 25.360 |
| Prefill T=1024 | 59.359 / 72.168 / 78.687 | 33.080 / 49.501 / 73.417 |
| Prefill T=2048 | 123.151 / 193.621 / 201.846 | 84.912 / 169.567 / 187.633 |
| Prefill T=4096 | 203.105 / 239.618 / 254.813 | 136.129 / 201.219 / 240.823 |

T 是每请求的 Main 输入位置数，不是最终接受/生成数；B=8,T=16 是 128 个输入位置、
去重前 2048 行。共三轮、111 个完整试验、53,784 个计时批次。小窗口通常每轮 1000
批、批间留 20 ms，prefill 连续处理；串行参照在确认毫秒至秒级延迟后减采样，原样本
保留。大 prefill 只有 96–384 个样本，尾部分位为描述结果；不同 T 使用不同固定文本
片段及总长度，不能作为同一 prompt 只改变 chunk 大小的严格对比。

并发读取相对串行参照是主要改进：原始完整小点串行单路/八路 decode P50 为
6.533/47.842 ms，串行 B=8,T=16 各轮为 411–436 ms，T=4096 为 5.90–6.24 s。
这是读取路线比较，不是对已有优化引擎的加速比。

页内布局另与同样使用 io_uring 的 90 B 连续行布局作配对比较。多数窗口的合并样本 P50
较低，但**单路 T=1 没有稳定收益**：P50 从 0.603 增至 0.617 ms，三轮中两轮变慢、
一轮变快。页内文件多 312.53 MiB。两种路线只合并同页，不合并相邻不同页；文件
实际存储位置不同，因此不能把时间差全部归因于请求数，也不证明页内布局胜过所有
可能的连续行 reader。退化保留，不以“噪声”排除。

无缓存 T=4096 的平均总加载为 197.115 ms，其中读取/收集 189.198 ms、CPU 解码
4.725 ms、Host 观察 H2D 0.553 ms。平均读取 108.861 MiB，低于未去重的 256 MiB
上界；“无缓存”仍有批内去重，也未排除宿主机/控制器缓存。T=2048 三轮平均读取
60.21/93.60/64.90 MiB、平均加载 109.53/161.99/115.80 ms，不能只按 T 设固定延迟。
CPU LUT 未成为这次主要瓶颈，后续优先用有界预取覆盖实际随机读取等待。

运行节奏和异常同样保留：连续提交的单路无缓存 P50/P95/P99 为
0.372/0.462/0.533 ms，20 ms 间隔下为 0.617/1.813/3.260 ms。差异包含整条 Host/存储
链路，未证明全部来自 SSD 电源状态。一次 **64.462 ms** 完整加载中，磁盘+CPU 准备
仅 **0.512 ms**（读取/收集 0.475 ms），Host CUDA H2D 调用/完成确认区间为
**63.950 ms**，CUDA event 区间为 0.012096 ms。这是未进一步定位的 Host CUDA
交接异常，**不是 64 ms SSD 读取**；另有一次真实读取/收集 11.609 ms 的慢样本，也未删除。

`max(0,L-W)` 的提前量推算说明预取需求，不是实测 GPU/SSD 重叠：T=4096、W=250 ms
时，无 cache 的 96 批中仍有 1 批等待 4.813 ms，有 cache 样本全部覆盖；B=8,T=16、
有 cache、W=10 ms 时覆盖 97.8%，无 cache 平均仍等 4.654 ms。首 chunk 缺少上一
chunk 的完整提前窗口，须单独测 TTFT。T=4096 按累计服务时间计算的供给率为无 cache
20780、有 cache 30603 输入位置/s，不是模型吞吐；长期消费速度超过供给时，单纯增加
预取距离不能解决不足。

实验有 1028 组 Python/C++ hash oracle、4336 组 Fraction BF16 舍入检查、1040 个
独立行内容检查及逐 trial 的 GPU 回读核对。这些验证实验地址/数据路径，不替代正式
converter、生产 reader、PLE 数学或整模质量资格化。固定资源包括 512 KiB I/O buffers、
512 KiB LUT、最多各 20 MiB pinned/Device 输出及可选 64 MiB row cache，另有 corpus
与收集 scratch；未测进程 RSS 峰值。

实验未运行实际 PLE 投影/Conv、MoE、attention 或 draft 模型，未实现 V3 cold backing、
生产预取调度或 GPU 解码，也没有 resident 模型的内存压力和并发 expert H2D。
O_DIRECT 只绕过 guest Linux table page cache，未排除 Windows/VHD/控制器缓存，
不能称为裸 NVMe 全冷读。缓存收益依赖本次文本轨迹；完整模型延迟、质量及传输竞争
仍需在实际集成后测量。

## 9. 生产资格化边界

- Hash：完整 64-bit overflow/XOR/正模、序列开始、EOS 自身及后续两位置、跨 chunk
  与 append、视觉 raw IDs、多请求隔离；分段结果直接对独立整数 reference。
- NF4：有效 LUT 组合逐项对精确 reference，含零符号与 RNE 边界；非法 scales/divisor
  明确失败，行解码对独立 codes/scales/divisor oracle。
- 寻址：首/末行、45 行页界、128 shard 边界、末页 37 行、divisor/padding、续卷及
  非常规分卷、实际 direct-I/O 对齐与短读，不把逻辑零行与 padding 混淆。
- 异步：miss/命中/全命中、跨 ticket 共享读、背压、append/seal 截断、late completion、
  取消中 H2D、slot generation 重用、必需/预取失败及 shutdown，验证无旧窗口误发布。
- 集成：真实 artifact、实际语料、首 chunk/稳态、Graph/eager、真实内存压力与 expert
  H2D 竞争；分别报告 CPU-ready、GPU-ready、暴露等待、尾延迟、cache/读放大与完整
  资源占用。GPU 数学和统一接受前缀资格化沿用 [ple.h](../../include/ninfer/ops/ple.h)、
  [Op 开发](op-development.md)及[执行与状态](qwen4_exp-execution.md)，不能只用 kernel
  parity、独立磁盘带宽或实验校验和宣称完成。
