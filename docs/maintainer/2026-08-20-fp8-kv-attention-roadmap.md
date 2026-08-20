# FP8 KV-cache Attention 分阶段实施路线图（2026-08-20）

## 1. 文档状态与目标

本文是当前 FP8 KV-cache Attention 工作的临时实施路线图，只拥有阶段顺序、阶段准入条件和
尚未落入稳定 contract 的数值决策。它不替代以下长期 authority：

- [`op-development.md`](op-development.md) 的 Op 边界、ownership、qualification 和性能规则；
- `include/ninfer/ops/attention_geometry.h`、`softmax_attention.h`、
  `sliding_window_attention.h` 和 `kv_cache_append.h` 的当前公共 contract；
- [`paged-kv-cache.md`](paged-kv-cache.md) 的物理 page、plane、block table、frontier 和 prefix
  ownership；
- 迁移完成后的 `include/ninfer/ops/` contract header，以及对应模型数学 reference。

工作必须严格按以下顺序推进，不在前一阶段留下新旧双路径时开始后一阶段：

1. 完成 Softmax Attention 公共语义和源码目录的一次性整理；
2. 在现有 INT8-G64 KV-cache profile 上加入固定 Hadamard rotation；
3. 复用已经验证的语义和 primitive，实现 FP8 E4M3FN KV-cache Attention。

本路线图的交付边界是 Op 层支持。Engine/CLI 的 cache storage 选择、target pool 配置和实际模型
接线在第三阶段 Op 完成后另行实施，不得反向改变这里已经 qualification 的 Op 语义。

第三阶段完成且稳定要求已经进入 contract header、活跃 maintainer reference、测试和 benchmark
后，删除本文及 `docs/README.md` 中的入口，不保留历史 roadmap。

## 2. 跨阶段固定事实

### 2.1 公共 Op 边界

目录迁移后，Text/MTP causal cache 只保留以下三个语义效果：

```text
kv_cache_append                    只写 cache
causal_softmax_attention           写入当前 K/V，并计算 causal Attention
causal_softmax_attention_cached    只读已有 cache，并计算 causal Attention
```

`prefill`、`decode`、`small_t`、`prompt`、`split_kv`、Hadamard、Q 量化和 V 解码都是私有实现
阶段，不创建公共 Op。workspace capacity query 是资源查询，不是第四个 Op。

### 2.2 固定 Hadamard 变换

INT8 和 FP8 使用同一个 D256 正交变换：

```text
R = H256 / 16
```

其中：

- `H256[i,j] = (-1)^popcount(i & j)`，即标准 Sylvester Hadamard matrix；
- device primitive 以 8-stage FP32 FWHT 实现，最后乘精确的 `2^-4`；
- 不存在 seed、sign table、request/layer/position 相关状态或 cache metadata；
- Q/K 必须在 Q/K RMSNorm 和 MRoPE 之后、量化之前施加同一个 `R`；
- V 默认不旋转。

固定变换使 prefix-reuse page 可以直接共享，不需要保存或转换 request-specific rotation 状态。
数学上 `R^T R = I`；生产 kernel 可以在旋转坐标中计算 dot product。Hadamard 是私有计算策略，
不成为公共 Op、中间 Tensor 或独立 oracle 结果；Attention 仍由现有公共输入和独立理想 oracle 定义。

## 3. 第一阶段：完成 Attention 目录和语义整理（已完成）

### 3.1 范围

2026-08-20 已完成一次一致 cutover：

- 建立 `include/ninfer/ops/attention_geometry.h`、`softmax_attention.h`、
  `sliding_window_attention.h` 和 `kv_cache_append.h`；
- 把 causal cache、plain/packed、context 和 sliding-window 实现迁入
  `src/ops/softmax_attention/` 的目标目录；
- 把 standalone append、device-count prefix append 和唯一 cache codec 迁入
  `src/ops/kv_cache/`；
- 将 target caller、workspace capacity、显式 CMake source、测试和 benchmark 全部切到新
  semantic entry；
- 删除 `gqa_*`、`vision_attention`、`bidirectional_gqa_attention`、`swa` 的旧 header、符号、
  source path、测试 target 和 forwarding path；
- stable formula、supported domain、state mutation、alias 和 workspace 规则进入新的 contract
  header。

本阶段只改变了 ownership、命名和目录，没有改变 BF16/INT8 codec、数值 profile、可见域、cache
状态效果或生产 route。迁移使用独立 FP64/exact oracle、page mapping、masked-row、batch 和 Graph
用例验证；目标侧无 artifact 的 real/load-plan 用例按既有 skip contract 跳过。稳定态 representative
benchmark 未发现超过 3% 的迁移回退，context T=16/L=131072 的 64-key route 前后约为
326.1/326.8 us。

### 3.2 完成门槛

- 所有生产 target 只 include 新 semantic header；
- 同一个 launcher/kernel 只有一个 build owner，旧路径已删除；
- 当前 BF16 和 INT8-G64 causal append、cached-only、standalone append qualification 通过；
- packed、context 和 sliding-window 的现有 observable behavior 通过各自独立 oracle；
- B=1..8、CUDA Graph、masked rows、identity/offset/fragmented page mapping 继续有效；
- 相关 target Text、MTP、Vision 和 DFlash focused execution 通过；
- 已完成迁移的临时说明按文档生命周期删除或收敛，不留下第二套 implementation map。

上述条件已经满足；逐文件临时计划和完成使命的目录目标说明已删除。第二阶段可以开始。

## 4. 第二阶段：为 INT8-G64 KV 加入 Hadamard rotation（已完成）

第二阶段已经在现有 INT8 K/Q 量化中接入后续 FP8 可以复用的 H256 primitive，同时保持公共
Attention 数学、cache page layout 和 V/PV 路径不变。完成使命的逐 kernel 临时计划已经删除；
本节只保留第三阶段仍需依赖的稳定事实。

### 4.1 INT8 私有计算 profile

现有 INT8-G64 scale、rounding、clamp 和 decode 公式保持不变。唯一变化是 K codec 的输入：

```text
k_rot       = R * BF16(k_after_mrope)
k_code      = existing_INT8_G64_encode(k_rot)
K_logical   = R^T * existing_INT8_G64_decode(k_code)

v_code      = existing_INT8_G64_encode(BF16(v))
V_logical   = existing_INT8_G64_decode(v_code)
```

因此：

- K cache 的私有 code/scale 表示有意改变；
- V 不旋转；
- BF16 cache profile 完全不变；
- 当前调用中新写入的 K/V 必须以 cache codec 后的逻辑值参与 Attention，不能旁路使用未量化
  BF16 K/V。

生产 Q route 执行：

```text
q_rot  = R * BF16(q_after_mrope)
q_code = existing_Q8_G64_encode(q_rot)
```

Q/K rotation 和量化均是私有 compute profile，不成为公共输入语义或 oracle 的显式中间值。
包含该 profile 的 append-and-attend 和 cached-only 结果继续直接对现有独立 Attention oracle。

### 4.2 实现 ownership

- `src/ops/kv_cache/hadamard_d256.cuh` 唯一拥有 `R` 的完整 warp-register primitive 及其等价的
  H64/H4 factorized device helpers；standalone append、prompt Q 和 small-T Q 共享这些定义；
- standalone K writer 由一个 warp 完成整行 H256。small-T fused writer 为保持原有
  `(token,group)` 并行度，在同一 CTA 内并行形成四个 FP32 H64 fragment，经一次 shared-memory
  exchange 完成 H4 后进入不变的 G64 codec；该分解不增加 cast，也不复制 transform 数学；
- small-T append 的 page ID 从已有 CTA shared page table 广播，四个 group 不重复读取 global
  block table；
- 不创建 standalone Hadamard Op，不在 target 内实现 rotation；
- prompt 和 small-T 都在现有 kernel 内使用 warp-register FWHT；本阶段不增加 prepared-Q
  workspace、额外 staging kernel 或 route switch。

本阶段不借机改变现有 INT8 PV dtype、softmax profile、scale granularity 或 cache physical layout。

### 4.3 Qualification 结果

- 已复用现有两个 D256 geometry、small-T、prompt、mapping、masked-row 和 Graph 用例，没有为
  Hadamard 建立新的笛卡尔测试矩阵；
- append-and-attend 及 standalone append 后的 cached-only Attention 已直接通过同一个独立
  Attention oracle；
- 没有建立 exact Hadamard、basis/impulse、seed、raw rotated-K bytes 或 activation sweep；
- 实施前冻结的 public Attention 代表点复测没有显著退化，因而不保留候选调度、route forcing 或
  额外性能 gate。

### 4.4 完成结果

- 固定 H256 只有一个 implementation owner；
- INT8 append-and-attend 和 append-then-cached-only 已通过现有独立 Attention oracle；
- 没有可选 `use_hadamard` flag、unrotated compatibility path 或旧 codec alias。第三阶段可以开始。

## 5. 第三阶段：实现 FP8 E4M3FN KV-cache Attention

第三阶段复用第二阶段已经 qualification 的 `R`、append state semantics、logical-cache oracle
结构和 Q preparation 边界。FP8 是现有三个公共 Op 的新 cache profile，不增加新 entry。

### 5.1 已选择的持久格式

基准生产格式固定为：

```text
K code   [256, 64, Hkv, Nphysical]  FP8 E4M3FN
V code   [256, 64, Hkv, Nphysical]  FP8 E4M3FN
K scale  [  1, 64, Hkv, Nphysical]  FP16
V scale  [  1, 64, Hkv, Nphysical]  FP16
quant_group = 256
```

即每个 `(token, kv_head)` 的 D256 K row 和 V row 各有一个 FP16 scale。每 token/KV head 的
K+V payload 为 `256 + 256 + 2 + 2 = 516` bytes。

对 K，codec 输入是 `R*k`；对 V，codec 输入是未旋转 V。row encode 定义为：

```text
a = max_d abs(FP32(x[d]))

a == 0:
    scale_bits = FP16(+0)
    code[d]    = E4M3FN(+0)

a != 0:
    raw_scale  = a / 448
    bounded    = clamp(raw_scale, min_positive_FP16, max_finite_FP16)
    scale_bits = FP16_RNE(bounded)
    s          = FP32(scale_bits)
    code[d]    = E4M3FN_RNE_SATFINITE(FP32(x[d]) / s)

decode[d] = FP32(code[d]) * FP32(scale_bits)
```

对应原坐标逻辑值为：

```text
K_logical = R^T * decode(K_code, K_scale)
V_logical =       decode(V_code, V_scale)
```

合法输入域是注册目标产生的有限 BF16 activation；不增加逐元素 device validity scan。

### 5.2 已选择的计算 profile

```text
Q preparation:
    BF16 Q -> FP32 Hadamard -> rowwise E4M3FN
    transient Q scale 保持 FP32

QK:
    E4M3FN x E4M3FN Tensor Core
    FP32 accumulator
    Q/K scales、attention scale、mask 均在 FP32 应用

Softmax:
    max、exp、sum、online rescale 和 split statistics 全部 FP32

PV operand:
    FP32 P tile -> FP16_RNE
    E4M3FN V code -> exact FP16 -> 与 FP16 V scale 做一次 FP16_RNE multiplication

PV/result:
    FP16 x FP16 Tensor Core，FP32 accumulator
    split partial output、merge 和 normalize 保持 FP32
    只在公共 Out store 时转换为 BF16
```

E4M3FN 的全部有限 code 都能精确表示为 FP16，因此 `code -> FP16` 不引入舍入；任意
`code * scale` 不保证仍可被 FP16 精确表示，生产路线只允许上述一次 FP16 multiplication，不经
BF16 中转，也不把 scale 预先折入 P。不得为了乘法精确而默认使用 power-of-two scale，因为它会
改变原始 FP8 量化误差。

FP16 是唯一首选 PV operand profile。只有真实 activation trace 和专门 softmax-tail 用例证明
FP16 下溢造成不可接受回退时，才重新打开 BF16 比较；不同时保留两条没有独立 supported domain
的生产路线。

### 5.3 Prefill 私有路线

- append-and-attend 必须先产生与 standalone append 相同的 FP8 code/scale，再让当前 Query 使用
  这些量化后逻辑值；
- Q tile 在 register/shared memory 中完成 Hadamard、FP32 amax 和 E4M3FN 量化；
- K pages 以原生 FP8 operand 进入 FP8 QK Tensor Core，不生成 global dequantized K；
- V pages 只在片上转换和缩放为 FP16；
- P 不写 global tensor，只在进入 PV 前从 FP32 tile cast 为 FP16；
- online softmax、PV accumulator 和最终 normalization 保持 FP32。

### 5.4 Decode/small-T 私有路线

性能目标是对每个 Query group 单次流式读取可见 K/V cache：

- CTA 以 `(sequence, kv_head, split)` 组织，同一个 K/V tile 复用于该 KV head 对应的全部 6 或 8
  个 Q heads；
- key tile 对齐 page size 64，每 tile 只进行一次 block-table lookup；
- K/V code 和 scale 使用合并、向量化 load，不先 gather 到 contiguous buffer；
- K 直接进入 FP8 MMA，V 只在 register/shared memory 中转 FP16；
- split partial `m/l/o` 全部 FP32，reducer 以 FP32 合并并归一化；
- 不允许第二次完整 cache pass、GQA-shared K/V 的乘数级重复加载或 global FP16/BF16 KV 副本。

Q Hadamard/quantization 是 fused 重算还是一次性物化，由完整公共 Op latency 决定。workspace
capacity 必须覆盖选中的 graph-safe 路线，地址在 CUDA Graph capture/replay 期间稳定。

### 5.5 允许重新打开的数值 gate

以下不是并存 format，而是基准 profile 失败时按顺序重新评估的单次设计 gate：

1. 默认 `V row256 + no rotation`；若真实 trace 显示 V quantization 是主要误差，再评估 V-G64；
2. 只有 V-G64 仍不能满足质量门槛，才评估 `V rotation + output inverse rotation`；
3. FP16 PV 只有在长尾下溢造成实际回退时才与 BF16 重比；
4. route 选定后删除失败候选及 forcing 参数，并将最终唯一格式更新到 contract。

K/Q 的固定 Hadamard、FP8 QK 的 FP32 accumulator、FP32 softmax、FP32 PV accumulator、最终 BF16
公共输出不是候选项。

### 5.6 Qualification 与性能门槛

数值和状态至少覆盖：

- 枚举全部 254 个有限 E4M3FN code，验证 `FP8 -> FP16` exact；
- exact FP8 codec、zero、rounding ties、scale bounds 和 saturation；
- A1 与 standalone append 的 cache code/scale bit identity；
- A1/A3 分别直接对使用 `K_logical/V_logical` 的独立 FP64 oracle；
- 两个 D256 geometry、B=1/2/4/8、W=1..16 route seams、代表性 prompt widths；
- context 0/1/63/64/65/127/128/2K/8K，以及长 context spot；
- identity/offset/fragmented page mapping、invalid rows、A3 no-mutation、Graph replay；
- 真实目标 activation trace，以及专门覆盖 FP16 softmax-tail underflow 的构造输入。

Prefill 必须确认生产 kernel 实际发出原生 E4M3FN QK Tensor Core 指令，且不存在 global K/V
dequantization。

Decode 的 VRAM 门槛使用相同 FP8 page layout、block-table access 和 fragmentation 的任务期
read-only control 测量可实现上限。大 context、cache-dominant workload 的生产 Attention 必须达到
该 control 至少 95%，并同时报告相对 RTX 5090 `1674.5 GB/s` 已测 pure-read ceiling 和
`1792 GB/s` 规格带宽的比例。若未达到，先确认实际 DRAM traffic、重复读取和 stall attribution，
不得仅用 logical bytes/s 宣称完成。

### 5.7 第三阶段完成门槛

- FP8 persistent format 与公共 cache/Attention 语义已写入稳定 contract；Hadamard 仍是私有
  compute profile；
- standalone append、append-and-attend、cached-only 全部通过直接 qualification；
- prefill 使用原生 FP8 QK，PV 使用选定的 FP16/FP32 路线；
- decode 满足单次 cache stream 和匹配布局的 VRAM throughput 门槛；
- BF16、INT8 和 FP8 各自只有一个已 qualification 的生产 profile；
- 所有临时候选、route forcing、对照实现和本路线图均已删除。

## 6. 阶段状态

| 阶段 | 状态 | 后续阶段准入条件 |
|---|---|---|
| 1. Attention 目录和语义整理 | 已完成 | 准入条件已满足，第二阶段可以开始 |
| 2. INT8-G64 Hadamard | 已完成 | 准入条件已满足，第三阶段可以开始 |
| 3. FP8 E4M3FN KV Attention | 未开始 | 第二阶段共享 H256 primitive 与 paired Q/K 路径已经稳定 |

实施中只更新本表的当前状态和仍会影响后续工作的实质决策。详细命令、日志、候选 sweep、profile
报告和已完成事项不回填为历史记录。
