# Qwen3.5-35B-A3B expert 权重容器与 offload 设计

状态：设计，尚未实现。日期：2026-10-08。

本文落实 Qwen3.5-35B-A3B 的权重容器、加载分支、GPU 副本管理和执行交接。
首个实物对象是本地 `out/qwen3_6_35b_a3b.ninfer`；它是
`Qwen3_5MoeForCausalLM` 架构的训练实例，不增加 checkpoint 专用执行入口。
数学公式继续由 [Qwen3.5 模型合同](qwen3_5-model.md)和具体 Op 拥有；
[Engine 架构](engine-architecture.md)拥有请求、提交及执行资源边界。

本文是该容器方案的设计依据，不表示当前 CLI 已有显存限制或 CPU MoE 能力。

## 1. 范围与已确定的选择

1. **先作容量判定，再选择加载位置。** 显存足够时直接沿用 GPU materialization，
   不建立完整 Host expert 副本。只有选定 offload 模式后，才建立完整 RAM backing。
   已有 loader 的有界 I/O/pinned staging 不属于完整 Host backing。
2. offload 模式中，Host 保存全部已启用 routed experts 的原生编码，地址在 Model
   生命周期内稳定；GPU 只保存可丢弃的副本。运行期不从磁盘补 expert，不回写 GPU 权重。
3. prefill/Replay 计算某个 MoE bank 之前，**该 bank 全部 expert 必须在 GPU 上可消费**。
   允许提前上传下一层的部分 expert，不执行层内 expert 分批计算和部分和拼接。
4. decode 的 GPU 目标集合 D 在启动时确定，阶段内不变；其余选中 expert 由 CPU 计算。
   GPU/CPU 使用同一次路由的系数，完整覆盖 assignment，不各自重新归一化。
5. GPU expert 预算固定，内容可以随阶段变化；不永久划分不能回收的 decode 区。
6. 阶段切换允许等待。先完成旧使用者、补齐新目标，再开始下一类计算。
   同一阶段内的后续 chunk/round 保留已有内容，不自动复位。
7. 第一阶段沿用现有 decode round 与 prefill chunk 穿插方式。以后延长某一阶段，
   只改变准备请求的时机，不改权重表示、容器或数学 Op。
8. 保留当前规则整 bank 的全 GPU 融合路径。CPU 线程、GPU 间接权重计算及分段执行是
   offload 的必要执行工作，不能只完成字节缓存就宣称支持 offload 推理。

第一条优先于任何“统一先创建完整 Host 权重”的实现便利。不能先把整个模型载入 RAM，
再判断显存是否足够；也不能先上传全部 expert，再用剩余显存倒推 offload。

## 2. 实物几何与准确容量

### 2.1 模型及来源核对

仅读取本地 v3 directory/Bindings 得到：

| 项目 | 当前实物 |
|---|---|
| Main 层数 / H / E / top-k / I | 40 / 2048 / 256 / 8 / 512 |
| Main routed gate/up | 每层 Q4，逻辑每 expert `[1024,2048]` |
| Main routed down | 37 层 Q5；层 34、38、39 为 Q6；每 expert `[2048,512]` |
| 可选 MTP routed bank | Q8+Q8，256 experts，同一 expert 几何 |
| Main 原始物理 parents | 80 个：40 个 gate/up bank、40 个 down bank |
| MTP 原始物理 parents | 2 个；仅启用 MTP 时需求化 |
| Router / shared | 按实际 Binding 准备现有 GPU 权重，不加入 routed cache |
| Main 非 routed 张量 payload | 2,482,861,056 B，约 2.312345 GiB |
| Main routed payload | 18,555,600,896 B，17.281250 GiB |
| MTP routed payload | 855,638,016 B，0.796875 GiB |
| MTP 新增非 routed payload | 42,296,320 B，约 0.039392 GiB |
| Main+MTP routed payload | 19,411,238,912 B，18.078125 GiB |

Main 固定权重与 routed 权重没有共用物理 parent。Main 三个非层级 Binding 是
token embedding、output head、final norm。上表不包含未启用的 Vision/DFlash，
也不把这些 payload 数字当作实际 device reservation、进程 RSS 或 artifact 文件大小。

加载和执行依据 config/Bindings/Use。上述 layer ID 和 profile 数量是实物核对结果，
不能硬编码为容器的合法输入条件。

### 2.2 单 expert GPU image

动态缓存以一个完整 expert 为分配单位，包含两个原生编码矩阵：

```text
image base
  gate/up codes、可选 high、scales
  256-byte alignment
  down codes、可选 high、scales
```

RowSplit 每行的编码、分组和 padding 与现有格式相同。相对偏移如下，单位均为 B：

| Profile | GU codes | GU scales offset / bytes | down base | down codes | down high offset / bytes | down scales offset / bytes | image bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4+Q5 | 1,048,576 | 1,048,576 / 65,536 | 1,114,112 | 524,288 | 1,638,400 / 131,072 | 1,769,472 / 32,768 | 1,802,240 |
| Q4+Q6 | 1,048,576 | 1,048,576 / 65,536 | 1,114,112 | 524,288 | 1,638,400 / 262,144 | 1,900,544 / 32,768 | 1,933,312 |
| Q8+Q8 | 2,097,152 | 2,097,152 / 131,072 | 2,228,224 | 1,048,576 | 无 | 3,276,800 / 65,536 | 3,342,336 |

high/scales offset 都相对 image base；GU 无 high。每个矩阵自身的 Weight.payload 指向
各自矩阵起点，payload_bytes/high_plane_bytes 使用该矩阵尺寸，不沿用源大 parent 的容量。

以上三种 image 的所有 plane 边界已经满足 256-byte 对齐，无额外尾部 padding：

| Profile | MiB / expert | MiB / 256-expert bank |
|---|---:|---:|
| Q4+Q5 | 1.71875 | 440 |
| Q4+Q6 | 1.84375 | 472 |
| Q8+Q8 | 3.18750 | 816 |

不能用 3.1875 MiB 的最大槽容纳所有条目：Q4+Q5 会浪费约 46.08% 的槽容量。

### 2.3 从原始 bank 提取

源 GU parent 是 `[262144,2048]`，down parent 是 `[524288,512]`。
expert e 的 GU 行为 `[1024e,1024(e+1))`，down 行为 `[2048e,2048(e+1))`。
gate/up 两个数学 Binding 顺序合并成该 GU 行块，不复制两遍 GU 数据。

对 RowSplit plane p：

```text
source = parent.data + parent_plane_offset[p] + first_row * plane_row_bytes[p]
length = expert_rows * plane_row_bytes[p]
destination = image_base + image_plane_offset[p]
```

每 expert 有 5 个（Q4+Q5/Q6）或 4 个（Q8+Q8）源片段。
这是原生 plane 字节搬运；不调用 encoder、decoder，不改行内布局和数值。
Host 保留源 parent，不另建完整的 per-expert packed RAM 镜像。

[Weight row planes](../../src/core/weight_view.h)已经提供源行区间的基础表达。
GPU image 必须由新的 image layout 描述建立本地矩阵视图，不能把源 parent 的偏移
和 payload size 原样套到较小的目标 allocation。

## 3. 启动分支：在 payload materialization 之前确定

### 3.1 当前需要改变的位置

目前：

- [Bindings::parameter](../../src/models/qwen3_5/load/prepare.cpp)默认请求 Device。
- [Binder](../../src/artifact/binder.cpp)遇到 Host Binding 会调用 host_object 并实际读入数据。
- [construct_model](../../src/runtime/engine/model_instance.cpp)先 materialize，
  再根据剩余显存解析 KV/runtime 容量。
- [Parameters](../../src/models/qwen3_5/execution/parameters.cpp)直接准备完整 SparseMoeWeights。

所以不能仅把 routed 参数改成 `Residency::Host`，然后再判断是否 offload：
那仍然会在显存充足的路径提前读入完整 Host 权重。

### 3.2 新的冷准备顺序

```text
读取 directory、config、所选组件、Bindings、Use 和必要小标量
  → 得到不含权重数据地址的执行 profiles
  → 查询 workspace/state/KV/Graph 需求和实际编码字节数
  → 在 device 预算下选择 Resident 或 HostBacked
  → 确定 routed 参数的真实 Residency
  → materialize 所选 payload
  → 绑定实际 Parameters，复核 profile 与容量
  → 构造 Program、warmup、验证预算
  → 接受请求
```

增加只做语义验证和需求收集的绑定阶段。routed 引用在 placement 决策前不触发
require_device，也不触发 host_object。物理 Residency 必须在最终 materialization
前解析完毕，不把“未解析参数”交给 Model/Op。

metadata profile 只保存配置、shape、format/layout、Use 许可、融合 parent/range 关系和
容量查询所需事实。不能伪造非空地址让 native_weight 通过检查。现有 workspace 构造改为
消费这些 profile；实际参数绑定与预规划复用相同的格式/几何检查，避免维护两套容量公式。

### 3.3 两条加载路线

| 项目 | Resident | HostBacked |
|---|---|---|
| routed payload 来源 | artifact → 有界 I/O staging → GPU | artifact → 完整 RAM backing |
| 完整 Host expert 副本 | 不建立 | 建立并保留 |
| GPU routed 表示 | 原规则 bank，稳定地址 | Program expert image 副本池 |
| CPU expert | 不参与 | 读取完整 Host backing |
| 现有全融合 SparseMoe | 保留 | 不把零散 images 伪装成原规则 bank |
| 运行期权重读取文件 | 无 | 无 |
| 运行期 expert D2H 回写 | 无 | 无 |

固定权重仍由 Model 持有 GPU backing；动态缓存和上传资源由 Program 持有。
Resident 沿用现有 materializer 的直接 device 路线及 Model backing，不额外走 HostExpertBacking。

### 3.4 显存预算与 KV

新增的总 device 预算是计划中的 Engine 选项，建议名为
`device_memory_budget_bytes`；本文不宣称已有 CLI 参数。
12/16 的实验值统一写为 12/16 GiB，使用准确字节数。

在分配权重前获取实际可用量，并以用户预算与实际可用量中的较小值规划；
同一 Engine 启动后已计费的 allocation 不能在后续 cudaMemGetInfo 中再重复扣除。
保持一份 startup 账本，不把“当前 free”当作人工预算的替代。

定义：

```text
B = 本次 Engine 可用 device 预算
F = 所选固定 GPU 权重的真实 allocation bytes
R(mode, kv) = state + KV + workspace + I/O + 所选 Graph/执行路径 allowance
M = expert GPU descriptor/mask 等固定 device 元数据
C = expert image 数据池容量
H = 已有 automatic KV headroom（若启用），不拿来装 expert
```

Resident 先检查：
`F + routed_regular_bytes + R(Resident, kv) + H <= B`。

HostBacked 检查：
`F + R(HostBacked, kv) + M + C + H <= B`，
并要求 `C >= max_bank_image_bytes`。C 包括全部已预留的 image，而非只计算 Ready 条目。

第一组 12/16 GiB 验收固定 explicit kv_capacity、max_context、prefill_chunk、
max_concurrency、KV codec 和启用组件，排除 Auto-KV 改变实验工作量。
当前 Engine 默认也是 explicit KV。

Auto-KV 的设计规则：先以 minimum KV 判断是否可全驻留；若可，保留完整 GPU 权重后
按已有规则扩展 KV。若只能 offload，则默认保留满足既有 minimum KV 合同的容量，
将其余预算用于 expert，用户可用 explicit KV 扩大上下文池。该优先级必须报告在容量摘要中；
不能用旧 Auto-KV 先吃完余量，再把剩下一点空间分给 expert。

R 必须包含新 CPU/GPU 交接所需 device scratch，以及实际采用的分段 Graph 或 eager 路径
开销。现有全 GPU graph_allowance 不能未经验证直接充当新路径数字。
CUDA/context/module/Graph 的已有保留和 warmup 实际占用也要检查；预算不是 OS 级硬件隔离。

对当前 Main-only 实物：

```text
F_payload = 2.312344551 GiB
C < 472 MiB：不支持整层 Main prefill
C = 512 MiB：支持单层及部分下一层预取
C = 1 GiB：能同时容纳最坏的两层 Main（944 MiB）
```

启用该 artifact 的 MTP 时最大 bank 为 816 MiB；Main 最后一层 + MTP 为 1288 MiB。
这些数值是 expert 池的条件，不能据此声称某个总预算、context 和并发配置已经测得可运行。

## 4. 容器与所有权

### 4.1 只新增两个核心实体

`HostExpertBacking` 是 HostBacked 模式的 Model 只读资源：
完整源 parents、expert 索引、native Host views、image specs 和 copy spans。
现有 MaterializedArtifact 拥有实际 parent 字节，该索引不重复拥有一份 payload。

`ExpertGpuCache` 是 Program 的动态副本池：
固定 device buffer、按 ExpertKey 的目录、空闲区间、描述表、上传 staging 和事件。
它借用 HostExpertBacking，生命周期先于 Model 结束。

Program 另外保存普通值类型的固定 decode selection、当前执行类别和短期放置计划；
不另建通用 ExpertCatalog、MoeWork 状态机或请求级 expert checkpoint。

Main bank 编号按模型绑定映射到层 0..39；所选 MTP 是另一个 bank。
编号只用于本实例内查找，不成为 Op 的模型身份。

CPU 线程只读 Host backing，不修改 GPU cache。Engine worker 是缓存元数据唯一写者。
不同 resident lanes 共用一份 Program cache，不按请求复制权重。

### 4.2 基础数据声明

以下声明是拟新增的内部合同，当前仓库尚无这些类型。

```cpp
#include "core/weight.h"
#include "core/weight_view.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer {

struct ExpertKey {
    std::uint32_t bank = 0;
    std::uint32_t expert = 0;
    friend bool operator==(ExpertKey, ExpertKey) = default;
};

struct ExpertCopySpan {
    const std::byte* source = nullptr; // stable Host backing
    std::uint64_t image_offset = 0;
    std::uint64_t bytes = 0;
};

struct ExpertMatrixImage {
    WeightGeometry geometry;         // geometry of this local matrix
    std::uint64_t image_offset = 0;
    float weight_scale_divisor = 0;
};

struct ExpertImageSource {
    ExpertKey key;
    std::uint64_t image_bytes = 0;    // includes image alignment
    std::span<const ExpertCopySpan> copies;
    std::span<const ExpertMatrixImage> matrices;
};

struct HostExpertBankView {
    std::uint32_t expert_count = 0;
    std::uint32_t matrices_per_expert = 0;
    const Weight* matrices = nullptr; // Host [E, matrices_per_expert]
};

struct HostExpertBankSource {
    HostExpertBankView host;
    std::span<const ExpertImageSource> images;
};

} // namespace ninfer
```

当前模型按 `matrices[e*2+0]=gate_up`、`matrices[e*2+1]=down` 适配。
CopySpan 的 source 均指向已选中的完整 Host parents；loader 提前验证源/目的边界、
整行/合法 scale 区间以及目的 image 的覆盖。所有 spans/geometry 数组在启动后稳定。

LinearPolicy 和 activation Use 保留在模型参数；缓存不因搬运改变许可。
未来 NVFP4 独立 parents 可继续生成本合同的 image/copies，保留每矩阵 divisor；
容器能够保存该表示不等于已经实现 CPU NVFP4 算术。

## 5. GPU 物理组织和分配

### 5.1 一个共享字节池

`DeviceBuffer(C)` 启动时分配。所有 image 起点及大小按 256 B 对齐。
既不按 layer 永久分区，也不使用“最大 expert 大小”的统一槽。

使用一个小型、单 host writer 的有序 extent allocator：

- 空闲表按 offset 排序，相邻区间立即合并。
- 目录保存 key → offset/bytes/state；地址索引用于检查重叠与回收。
- 普通分配从已有 free extents 找连续空间；批量计划完成后再提交改变。
- metadata 最大条目数按所选 expert 总数和 C/min_image_bytes 在启动时界定并预分配。
- 不需要运行期 cudaMalloc，不需要 CUDA VMM，不把矩阵拆成 kernel 可见的分页权重。

可以提取 [HostContextArena](../../src/core/host_context_arena.h)中的区间分裂/合并算法，
但不复用其 pinned Host backing、context owner 或资源事务语义。代码所有权落在
expert 容器自己的 allocator，不顺带重构 KV/context 容器。

状态只有 `Absent / Uploading / Ready`。有完整 Host backing，所以没有 dirty、
唯一副本、D2H migration 或 Host destination reservation 状态。
Ready 只描述数据完成，不表示旧消费者一定结束。

### 5.2 放置计划

容器不决定请求调度或 online 热度。Program 的 expert 权重执行辅助逻辑按固定 D、
当前/下一 bank 和已有缓存生成计划。

正常路径：

1. 保留本次 required 集合中已有副本；Uploading 的同一 key 不重复上传。
2. 保护尚未结束的 consumer 和本次 required 条目。
3. 合并空闲空间；按已知使用顺序回收非 required、非保护条目。
4. 使用实际 image 大小分配缺失项，生成 copy 计划。
5. 一次性检查 plan 的 key 唯一性、范围、对齐、预算和覆盖依赖，再开始改目录和提交 DMA。

prefetch 只申请当前可以合法写入的空间，可以只得到下一 bank 的部分 expert。
它不调用 CPU expert，也不向模型返回“部分 prefill 层可计算”的结果。

第一版回收顺序固定、可复现：已知近期要使用的 bank 优先保留；
同等使用距离下保留 D 中条目；再按 key 决定顺序。
这是一项确定性的缓存策略，不根据线上路由更新 D。

### 5.3 碎片与保证前进

`free_bytes >= missing_bytes` 不保证每个连续 image 都能分配。

若回收非 required 条目后仍无法完成 required 计划，则：

1. 结束当前可变 bank 的使用并 quiesce 已提交的 expert 上传/计算。
2. 撤销动态池的所有副本映射，重置为一个空闲区间，不需要清零整个 device buffer。
3. 将 required 集合按稳定顺序紧凑排列，从 Host 重新上传。
4. 更新 descriptor 后，再向调用者提供 complete view。

这是有界的紧凑重建路径，保证 `sum(aligned image bytes of required) <= C` 时可以完成。
被清理的仅是动态 expert 副本，不影响固定 GPU 权重、KV/SSM、Host backing 或模型状态。

重建造成的已有条目重传必须单独统计；不得将其隐去或宣称免费。
预取放置应尽量对齐下一 required 集合的可完成布局，减少重建频率。
第一版不实现运行中任意 GPU 对象的在线 D2D compactor。

## 6. 传输与事件协议

### 6.1 有界 staging

Program 提供专用 expert upload stream；不通过全局 device synchronize 协调。
使用两个启动分配的 pinned staging 槽，初始容量点可取每槽 8 MiB；
它足以容纳本实物最大的单 expert，合计 16 MiB 属于 Host 预算。
大小是可调传输参数，不是 codec/数学合同。

Host 线程按 copy spans 填入 staging，将多段原生 plane 组装到各自原生 image 偏移，
再 H2D 到最终目的区间。无额外 device weight gather/repack buffer。
未复制的 alignment bytes 确定性置零；当前三个 image profile 没有内部空洞。

每个 staging 槽记录 H2D completion。CPU 重写该槽前等待对应 DMA 已读完。
大型 batch 按槽容量切传输，不要求一次 DMA 放下整层；计算仍等待完整 bank。

这里不能把 submit 实现为“循环装填整层并不断等待两个 staging 槽”：
那会在 Host 上占住调用链，使后续可以独立入队的 GPU mixer 工作无法及时提交。
submit 只验证并安装有界的待上传计划；progress_uploads 只使用当前可用 staging 槽，
每次最多装填该组槽的容量，不等待尚未完成的 DMA。必需的 view_on/wait_uploads 则会
继续推进待上传计划，在确实缺可用槽时等待，直到 required 数据已排队或已经完成。
没有隐藏上传线程，Host 侧进展由同一个 Program 调用链驱动。

首版只预取下一个实际 bank，不排入多层远期工作，因此 FIFO 的上传顺序与近期需求一致。
GPU 当前 MoE 入队后可以推进一次预取；随后先入队下一层的 attention/GDN 等独立工作，
再推进并等待下一 MoE 所需上传。这样两个小 staging 槽不会造成全层 Host 提交屏障。
仅合并目的连续且都属于本次上传的区间，不能跨保留条目或仍在读取的区间填充。

copy 准备/等待要计入传输观测。首版不默认 cudaHostRegister 整个十几 GiB 的 backing；
以后直传或更大的 pin 范围须依据测量替换，而不保留第二份完整 RAM image。

### 6.2 Prefill consumer

限定一个注册的 expert compute stream，以及一个 host 驱动者。
若 Op 内部使用其他 stream，必须先按其合同汇合到公开执行 stream。

每个 bank 有可复用的 ready 和 consumed 事件：

```text
上传所有必要 image planes
    → 上传这个 bank 的 device descriptor
    → record bank_ready

compute stream wait bank_ready
    → 完整 MoE 消费该 bank
    → record bank_consumed

未来覆盖这些区间：
    upload stream wait bank_consumed
    → 才允许写入新内容
```

同一 host 调用链保证：消费者已经提交，consumed record 已经入队，之后才提交依赖它的覆盖。
不能等待“以后才会 record”的事件；也不能在返回 view 与提交消费者之间重布置该 view。

可选 prefetch 优先提交不需要等待当前 bank 的上传，不把会阻塞队头的可选工作排在必需上传前。
阶段改变时停止增加旧阶段预取；已提交 DMA 正常完成，不假装可以撤销它。

### 6.3 Ready 与可排队消费

Uploading 条目已占用完整目的空间。只有全部 plane 的完成事件达到，Host 查询才报告 Ready。

若整个 required 集合的上传及最终 ready event 已提交，
`view_on(stream)` 可以在 Host 尚未观察到完成时返回：
它先在该 stream 排入等待，再允许 Op 入队。地址已稳定，但不能绕开所声明的 stream 使用。

Device descriptor 的发布在相应 payload 上传之后，表更新也不能与旧 kernel 读取重叠。
失败或未完成的拷贝不能发布为完整 bank。

### 6.4 Decode 与阶段边界

decode 进入后冻结 cache 布局。可重放多个 GPU segment/round，期间不逐 expert 做引用计数，
也不每个 round 重新上传 D。

退出该稳定使用区间时，Program 在计算 stream 记录本段最后使用的完成点，
等待它及已提交上传完成，再允许容器改布局。这样可以覆盖 Graph replay 中的实际读取，
不能仅依赖 capture 时记录过的 bank 使用事件。

这些规则足以管理首版生命周期，不需要跨请求 lease、generation tree 或一般异步事务。
源 Host backing 在 CPU worker 全部结束和 Program 销毁后才随 Model 释放。

## 7. 具体接口与视图

### 7.1 Core 容器合同

下列是内部接口声明草案。类型本身不包括路由计算，也不拥有请求阶段策略。

```cpp
#include "core/device.h"

#include <memory>
#include <vector>

namespace ninfer {

struct ExpertPlacement {
    ExpertKey key;
    std::uint64_t offset = 0;
};

struct ExpertUploadPlan {
    std::span<const ExpertKey> discard;
    std::span<const ExpertPlacement> upload;
};

struct ExpertCacheConfig {
    std::uint64_t data_capacity_bytes = 0;
    std::size_t staging_slot_bytes = 8ULL * 1024 * 1024;
    std::uint32_t staging_slots = 2;
};

enum class ExpertReplicaState : std::uint8_t { Uploading, Ready };

struct CachedExpertInfo {
    ExpertKey key;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
    ExpertReplicaState state = ExpertReplicaState::Uploading;
    bool read_pending = false;
};

struct ExpertFreeExtent {
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};

struct ExpertCacheSnapshot {
    std::span<const CachedExpertInfo> entries;
    std::span<const ExpertFreeExtent> free_extents;
    bool frozen = false;
};

struct DeviceExpertBankView {
    std::uint32_t expert_count = 0;
    const std::byte* const* image_bases = nullptr; // Device [E]
};

class HostExpertBacking {
public:
    explicit HostExpertBacking(std::span<const HostExpertBankSource> banks);
    std::uint32_t bank_count() const noexcept;
    HostExpertBankView bank(std::uint32_t id) const;
    const ExpertImageSource& image(ExpertKey key) const;

private:
    std::vector<HostExpertBankSource> banks_; // descriptors only; Model owns their arrays/bytes
};

std::uint64_t expert_cache_device_bytes(std::span<const std::uint32_t> bank_expert_counts,
                                       ExpertCacheConfig config);
std::uint64_t expert_cache_host_staging_bytes(ExpertCacheConfig config);

class ExpertGpuCache {
public:
    ExpertGpuCache(const HostExpertBacking& source,
                   ExpertCacheConfig config,
                   DeviceExecutionView compute,
                   cudaStream_t upload_stream);
    ~ExpertGpuCache();
    ExpertGpuCache(const ExpertGpuCache&) = delete;
    ExpertGpuCache& operator=(const ExpertGpuCache&) = delete;

    void submit(const ExpertUploadPlan& plan);
    void progress_uploads();
    void poll_completions();
    ExpertCacheSnapshot snapshot() const noexcept;
    void wait_uploads();
    DeviceExpertBankView layer_view_on(std::uint32_t bank,
                                      cudaStream_t consumer);
    DeviceExpertBankView subset_view_on(std::uint32_t bank,
                                       std::span<const std::uint32_t> required,
                                       cudaStream_t consumer);
    void record_layer_consumed(std::uint32_t bank, cudaStream_t consumer);

    void freeze_layout();
    void quiesce(); // drains registered consumers/uploads, then unfreezes

    std::uint64_t occupied_bytes() const noexcept;
    std::uint64_t data_capacity_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
```

device 容量查询包含数据池 C 和所有 bank 基址表的对齐后大小，只需要 metadata 中的
各 bank expert 数量，不依赖 HostExpertBacking 的构造或 payload 读取。实际构造 cache 时
复核 source 的 bank 数量与几何和预规划一致。Host staging 查询包含所有 pinned 槽。
Program 的 GPU ownership masks 和 CPU/GPU bridge 另由其布局查询计费，
同一 buffer 不重复相加。构造器只在启动分配，submit/view 等运行期操作不得增长 device
allocation。上面的声明省略与项目 MemorySummary 具体字段的适配。

submit 的 plan 由调用方 storage planner 产生；cache 必须拒绝重叠 kept allocation、
超出 C、重复 key、非法源片段、写入 frozen 布局或未建立覆盖依赖的计划。
验证在修改目录前完成。发生 CUDA/拷贝失败时，不发布部分 view；按 Engine 现有执行失败路径处理。

snapshot 借用预分配的 Host 目录和 free extent 表，直到下一次缓存修改才失效。
poll_completions 更新已完成上传及读取的观测，不等待未完成工作。
read_pending 帮助 optional prefetch 避免选择会阻塞的受害者；必需装载可以在 consumed
record 已入队后，通过 stream wait 覆盖该空间。submit 不负责另选 victim。
submit 返回时 image 可以仍在 Host 待上传队列；layer/subset_view_on 必须先推进 required
的所有 payload 和 table 写入，再记录并等待 ready。不能对尚未 record 的事件排入等待。
wait_uploads 会推进并等待所有已接收的上传计划；quiesce 同样处理队列中尚未入队的工作。

freeze_layout 要求其所需上传已经完成，冻结期间 submit 被拒绝。
quiesce 等待已注册 compute/upload 完成并解除冻结，不自动删除 Ready 副本。
layer_view_on 检查所有 E 个 expert；subset_view_on 检查明确 required 集合；
required 之外的 descriptor 即使存在也不授予本次计算所有权。

非 frozen 状态只允许一个尚未 record_layer_consumed 的 bank view 使用区间；
view_on 与 record_layer_consumed 成对，后者在 kernel 提交后记录事件并闭合 Host 使用区间。
frozen 状态的 views 可跨 rounds 借用，不为每次读取建立记录，由 quiesce 的整个 stream
完成点结束其有效期。禁止在非 frozen 使用区间尚未闭合时提交重布置。

传输 plan、查询快照和临时列表的最大规模在启动时有界，使用预留的 host 存储。
以首版单 worker/单 compute stream 的调用纪律保证 view 生命周期，不提供可任意长期保存的裸缓存引用。

### 7.2 GPU 描述表

对当前实物，每 bank 的 expert 具有相同 image layout，因此 GPU 只需一张地址固定的
`image_base[E]` 表和一份 Host mirror，不上传每 expert 两个完整 Weight 结构体。
逻辑 expert ID 直接索引 image_base；GU/down 的相对 offsets、format、shape 等由
模型准备的 bank profile 提供。表中地址可以随 cache 位置改变，表自身地址固定。

Main 基址表为 `40*256*8=81920 B`；若 ownership mask 用 I32[E]，另需 40960 B，
合计 120 KiB。启用 MTP 后合计为 123 KiB。两者计入 M，且与容量查询实际布局一致。
这些数字不包括 GPU 路由与输出 scratch，也不包括 Host metadata。

Host mirror 不能在仍作为异步 DMA source 时被改写；表更新同样通过受控 staging，
在 payload 完成后发布。数学适配必须按本地 image 建立正确的矩阵 offsets，
不能拿源大 parent 的 payload size 或 scale origin 做 GPU 寻址。
后续 NVFP4 可以从对应 image 的 divisor 字段取得各 expert 的 d_w；activation Use
由模型明确提供，不由缓存推测。bank 内若出现不同 image layout，须由相应 consumer
显式支持其 layout 元数据，不能套用当前 uniform-bank 优化。

数学适配层另外提供明确的 GPU ownership mask/集合：

- prefill：所有 expert 参与可选路由消费，整层 table 完整。
- decode：mask 是固定 D_l，CPU 使用其补集。
- 不能用“descriptor 为空”隐式决定漏算某个 expert。

当前两个模型矩阵数为 2，但 core source 的存储描述不解释 gate/up、down 的数学角色。
Op 只取得它需要的数学视图；不取得 artifact 名称、cache allocator、ExpertUploadPlan 或 phase。

### 7.3 模型参数与调用

Resident 的 FfnParameters 继续持有实际 GPU SparseMoeWeights，
并调用已有公开 sparse_moe 入口。

HostBacked 的 FfnParameters 增加一个具体的 MoE 参数变体：
router/shared GPU 参数、ExpertBankId、Host expert views/Use 和真实几何。
不把 Host 指针写进旧 SparseMoeWeights 后传给 GPU wrapper。

ExecutionCore/TextContext 增加对 Program 权重执行辅助逻辑的非 owning 引用，
由它取得当次完整层或固定子集 view。模型调用语义 Op，不能直接调原 sparse_moe 私有 launcher。

## 8. Program 的确定性行为

### 8.1 固定 D

D 是每 bank 的固定 expert ID 集合，先验提供并在启动时验证无重复、范围正确和总容量。
首个实验可以使用确定性的均衡生成器：按 expert 优先级轮次在各 bank 间轮转，
只加入剩余字节能容纳的 image；未提供离线排序时按 ID 升序。
它是可复现实验集合，不宣称已经识别热 expert。

D 的输入优先级可由离线数据替换；本阶段不采集线上热度以改变 D。
将 D 具体暴露为哪种 CLI 配置不属于数学容器接口；总显存限制必须经公共 Engine 传入。

### 8.2 进入 prefill / Replay

真实需要下一 prefill unit 时：

1. 若原来是 decode，quiesce 后解除稳定布局。
2. 请求首个实际会执行的 bank 完整可用；保留已有同 key 副本。
3. 将 ready 等待放在实际 MoE 消费之前；先入队能够独立执行的 embedding/mixer 工作，
   使它们与首层上传重叠。没有必要先装所有层，也不在整个 block 入口无条件同步权重。
4. 每层提交 MoE 后 record_layer_consumed，再预取下一个实际会调用的 bank。
5. 下一层若仍缺权重，按 consumed 依赖补齐；全 E 就绪后才能执行。

next bank 来自模型调用顺序，不以 `bank+1` 猜测所有 Main/MTP/控制路径。
同一阶段的下一个 chunk 保留 cache 内容，不隐式 enter_decode。
Replay 使用同样的大块整层路径；context 资源事务不决定 expert 回写。

### 8.3 进入和执行 decode

真实需要下一 decode unit 时：

1. 完成当前 unit；quiesce 旧消费者和已提交 expert uploads。
2. 请求 D 的缺失副本，必要时紧凑重建 D。
3. wait_uploads，准备各层 ownership mask 和 GPU descriptors，然后 freeze_layout。
4. 每层 GPU 做路由；GPU expert 分支与 CPU expert 分支并行；最终在 GPU 汇合。
5. 连续 decode rounds 直接复用 D，不重复执行 1..3。

CPU 使用 Program 提供的固定线程执行资源；一个 CPU Op 调用返回时其工作已完成。
CPU activations/routes/partial buffers 预先分配，weight 上传 staging 与激活交接的
生命周期分别计费，不能把未完成 H2D 的 host partial 覆盖给下一层。

prefill 已产生的输出按现有提交顺序发布，不等待恢复 D 才发布已经完成的首 token。
权重副本布局不进入 StateImage，不因 speculative rejection 恢复过去的布局。

### 8.4 所选后端、Graph 和大 T

最先闭环验证 Main、SpeculativeBackend::None、Vision disabled。
这不意味着容器完成后可以静默拒绝已启用的其他执行路径。

MTP 启用时注册其 bank，按实际调用点取得完整/子集视图；其 Q8+Q8 和 816 MiB 容量必须计费。
DFlash/DFlash2 的 Main verification 也经过同一 FFN；draft 自身 dense 权重保持现有路径。
Verify 的物理列数可以大于 ordinary decode 的 B，CPU/GPU bridge 和 scratch 必须按真实范围查询。
CPU Op 不把数学域限制为 T=1。

首版 HostBacked 的大块 prefill/Replay 走整层 GPU；decode/verification 固定 D。
离线 CausalScoring 的窗口也走整层 GPU；不能只根据 TextContext 的 Phase 枚举名字推断
权重策略，调用方必须明确传入本次执行类别。forced control 沿用当前小列数 decode 集合。
不在一个 speculative round 中自动来回切换两套驻留计划。针对大 verification T 的
整层策略属于以后基于测量的执行优化。

Resident 保留已有完整 CUDA Graph。
HostBacked 需要 GPU 分段或 eager 提交与 CPU 汇合，不允许把 CPU 等待塞进旧完整 GPU graph
后宣称兼容。采用哪一种 GPU 提交方式必须在构造时固定、纳入 R、明确报告；
开启 use_cuda_graph 时，只可使用已经资格化的 offload 分段实现，不能静默忽略选项。
首个普通 offload 垂直实验明确设置 use_cuda_graph=false；Resident 回归比较使用相同
Graph 配置，同时另外保留原有完整 Graph 产品路径的基线。分段 Graph 未完成时，
要求 offload+Graph 的启动必须在 payload materialization 前明确拒绝。

### 8.5 最低容量和多层容量

- C 恰好满足最大 bank：必要时等旧 bank 完成，重建/装入整层后继续。
- 一层与两层之间：prefetch 若干下一 bank 的完整 image，剩余部分随后补齐。
- 两层及以上：compute 与下一层上传重叠；更多空间可留住仍有用的副本。
- prefetch 只转移数据，不调用 routed Op 处理部分 prefill experts。
- 阶段切换允许等待；缓存必须跨单位保留，以便未来 Scheduler 连续执行多个单位。
- 全显存模式从启动起选 Resident，不为统一容器强制安装动态表或完整 RAM backing。

## 9. 容器之外必须完成的计算接入

容器只提供权重，不负责改变 MoE 数学。完整 offload 需要以下公开语义边界：

| 计算 | 必需交接 |
|---|---|
| GPU router + top-k/normalization + shared scalar gate | I32 ids[K,T]、FP32 coefficients[K,T]、shared gate |
| GPU routed SwiGLU reduce | 明确 GPU 子集，完整 GU/activation/down/加权归约，FP32 partial[H,T] |
| CPU routed SwiGLU reduce | 同一公式、明确 CPU 子集，直接读原生 Q4+Q5/Q6/Q8，FP32 partial[H,T] |
| GPU shared 与最终合并 | shared 分支、两侧 partial、原 residual → BF16 结果 |

这些是需要落实的数学合同，不能将“现有全融合 kernel 拆成 launcher 给模型调”作为接口。
数学资格化遵守 [Op development](op-development.md)，CPU/GPU/fused 各自对独立 oracle，
不以旧 GPU 输出的逐位一致作为唯一正确性标准。

CPU 带宽、SIMD 解码和 GPU indirect/grouped 路径的性能尚未验证。
特别是容器采用 per-expert image 后，当前大 bank 的 kernel 寻址不能原样消费；
新增的 GPU consumer 是必需实现工作，不能只改一个指针就假设性能成立。

## 10. 文件落点与实施顺序

拟新增文件仅在相应实现任务获授权后创建：

| 位置 | 责任 |
|---|---|
| `src/core/expert_weight_store.h/.cpp` | 原生 image/copy 描述、字节区间分配、GPU 副本目录、受控传输和事件 |
| `src/models/qwen3_5/expert_weights.h/.cpp` | 由实际 Bindings 构造 bank、CPU views、Use、copy spans 与固定 D 输入 |
| `src/models/qwen3_5/program/expert_execution.h/.cpp` | 阶段、短期放置计划、CPU/GPU 交接及 cache 生命周期 |
| `src/models/qwen3_5/load/prepare.cpp`、`load/text.cpp`、`load/mtp.cpp` | 延迟 residency 决策；按所选路线收集实际 materialization demands |
| `src/artifact/binder.*` | 将 metadata 验证与可能读入完整 Host 的 demand 分开 |
| `src/models/qwen3_5/execution/parameters.*` | pointer-free profiles 与实际 native preparation 的共同检查 |
| `src/models/qwen3_5/program/planning/*`、`src/runtime/engine/model_instance.cpp` | materialization 前选择模式；总预算、KV、cache 和实际 workspace 联合规划 |
| `execution/ffn.*`、`execution/text.*`、`execution/mtp.*` | Resident 原融合调用；HostBacked 显式视图与公开 Op 组合 |
| `include/ninfer/ops/` 与各 owning sources.cmake | 新计算合同、CPU/GPU 实现、原 fused kernel 的合法复用 |
| `include/ninfer/types.h`、CLI/bench 调用面 | 显式 device 预算与观测，不增加 checkpoint 特例 |

实施顺序：

1. metadata profiles 与双加载分支；验证 Resident 不读入完整 Host expert。
2. Host bank/copy plans 和字节池，完成 exact-copy、事件及容量测试。
3. GPU indirect 整层 consumer、CPU expert 数学及 bridge；闭环普通 Engine 推理。
4. 有限显存下 prefill/固定 D decode 切换，完成 16/12 GiB 的真实资源及性能验证。
5. 接通所选 MTP/verification/Graph 路径，资格化真实列数与生命周期。
6. 以后再调持续调度、离线 D、传输参数。online D 与互补驻留不在本阶段。

每一步保留必要的垂直接口与验收，步骤 1..2 的完成不标记整项 offload 已完成。
并发维护中的 GDN/Linear/QSA 代码不属于本设计文档的改动范围。

## 11. 验证与完成条件

### 11.1 存储和传输

- 核对所有选中 parent/Bindings，不使用对象名称猜 expert 角色。
- 两条加载路线分别验证：Resident 没有完整 Host backing；HostBacked 没有全量 eager GPU experts。
- 对每种真实 profile 的首/末 expert 和中间 expert 检查 copy span/source bounds、image coverage、
  rebased Weight 的 planes/shape/bytes；container 的搬运用独立 exact-byte/reference 验证。
- 不同 profile 共池，验证真实容量而非最大槽数量；包括 Q6 层和所选 MTP。
- C 小于一整层正确拒绝；C 等于所需实际尺寸、略大、一至两层之间及多层均能前进。
- 构造总 free 足够但连续块不足的布局，触发紧凑重建并核对上传增量、目录和未参与数据。
- 用不同 expert 的独立编码图样和延迟消费者验证 upload-ready/read-done 依赖，
  检出过早覆盖、staging 过早复用、descriptor 提前发布和 phase 后旧地址继续使用。
- 多次 P→D→P、P→P、D→D；没有实际阶段变化时不发生自动重建。
- Graph replay 后再换位，确保 completion 约束真实 replay，而不是 capture 时的使用。
- 关闭/取消等待已提交使用结束；缓存布局本身无需模型状态 rollback。

### 11.2 数学与 Engine

- CPU 三种实物 profile、GPU indirect 和原 fused 通过各自公开入口对独立数学 oracle。
- 路由拆分互斥且完备，保持原系数；空 CPU/GPU 子集有定义；每个物理 token 列得到完整结果。
- 小 T、批 decode、真实 verify 列数和整层 prefill 均覆盖；所有必需 workspace 计入预算。
- 公共 Engine 完成无人工限制、16 GiB、12 GiB 三类运行；配置与启用组件明确记录。
- 记录实际峰值、完整 Host payload、固定权重、cache、KV/state、workspace、staging、
  descriptors 和 Graph/driver allowance，区分 reservation 与实际观测。

### 11.3 性能解释

最少分别报告：

- 原优化全驻留路径与重构后的 Resident 路径，包含加载时间和 Host 峰值；
- prefill 计算、逐层 H2D、可见等待；
- 稳定 decode 的 CPU/GPU 分支与交接；
- P→D/D→P 等待及上传字节；
- 保留命中节省的上传与紧凑重建额外上传；
- 固定 D、请求/列数、codec、KV/并发/Graph 设置。

GPU microbenchmark 不代替 Engine 证据。阶段切换减少不等于 CPU kernel 变快。
本文件的容量是 directory 和编码公式计算结果，不是已经实现或测得的 12/16 GiB 性能。

## 12. 对 Flash Next 的边界

本实现先覆盖 Qwen3.5-35B-A3B 的真实矩阵和码型。Flash Next 后续可以复用：

- 先选 Resident/HostBacked，再 materialize 的流程；
- 完整 Host backing + 无回写 GPU 副本池；
- 不等长 image、逐 expert 矩阵元数据、完整 bank 就绪语义；
- 固定 D 与阶段等待；
- metadata profiles、GPU tables、预算和观测。

其 NVFP4 per-expert divisors、activation Use、512/top-10、不同 H/I、BF16 shared 和 HC
仍需各自的数学 consumer；不把它们塞进当前 Qwen3.5 精确几何 wrapper。
参见 [Qwen4Exp 权重方案](qwen4_exp-model.md)。
