# Qwen3.8-Flash-Next：子工作区任务清单

更新日期：2026-10-09。**A1/A2/A3 已实现并通过独立数学资格验证；A4 与第二批待实现。**

子工作区负责下表七项独立计算。主工作区负责共享存储、状态、读取服务和模型执行接入。
本文件维护任务范围与协作边界；数学、数值表示及接口分别由
[HC](qwen3_8-flash-next-hc.md)、[GDN](qwen3_8-flash-next-gdn.md)、
[PLE](qwen3_8-flash-next-ple.md)专题维护，实现后以对应生产头文件为准。

## 1. 共同基础

- Converter、矩阵合并、卷积离线转置、NF4 持久格式/分页 codec 和 20 项纯 Linear 已完成。
  本清单在这套表示上实现复合计算与运行时解码。
- 本机完整权重入口为 `/home/neroued/ninfer/out/qwen3_8_flash_next.ninfer`，同目录包含
  三个续卷。子工作区可以按显式路径只读使用这些权重。
- Tensor 第 0 维连续；M=T*B、column=t+T*b。GPU Op 使用现有 Tensor、Weight、
  DeviceExecutionView，以及合同规定的 caller-owned WorkspaceArena。
- 每项任务包含对应声明、实现、容量查询（若有）、构建注册和 Op 验证/调优；方法沿用
  [Op 开发](op-development.md)及仓库已有调优流程。

## 2. 完整清单

“影响面”描述需要共同修改的代码与合同；“难度”描述计算实现和优化工作量。

| ID | 批次 | 任务 | 影响面 | 难度 | 实现与数学资格 |
|---|---|---|---|---|---|
| A1 | 第一批 | Sigmoid-gated RMSNorm | RMSNorm 家族内部 | 低 | 已完成 |
| A2 | 第一批 | GDN 合并 AB 控制投影的新几何 | 控制投影 wrapper、plan、kernel、容量查询 | 中 | 已完成 |
| A3 | 第一批 | HC expand、grouped norm、combine、combine-norm | 独立 HC Op 家族，消费/产生 forward 激活 | 中 | 已完成 |
| A4 | 第一批 | HC block/final Mix | HC 家族内部，复用 Linear 与私有 workspace | 中至高 | 待实现 |
| B1 | 第二批 | PLE Gate | 固定 projection/record views 下的独立 GPU 计算 | 中 | 待实现 |
| B2 | 第二批 | PLE token hash | 模型内的纯 Host 数学函数 | 低至中 | 待实现 |
| B3 | 第二批 | NF4→BF16 LUT 与行解码 | Core 数值辅助，不依赖文件或请求对象 | 中 | 待实现 |

批次是建议推进顺序。B2/B3 已有完整输入/输出定义，可按主工作区需要提前合入。
完整模型接入的状态由[支持总览 §11.1](qwen3_8-flash-next-model.md#111-当前状态)维护。

## 3. 第一批

### A1. Sigmoid-gated RMSNorm

已在 [RMSNorm 合同](../../include/ninfer/ops/rmsnorm.h)实现 `sigmoid_gated_rmsnorm`，计算
`RMSNorm(x, gamma) * sigmoid(z)`。本实例 x/z/out 为 BF16 `[128,48,T,B]`，
gamma 为 BF16 `[128]`，沿 128 维归约、使用直接乘数 gamma。出口为 BF16。

完整 norm 与 gate 是一个数学 Op，中间不增加公开 BF16 舍入点；入口无 workspace。
既有 SiLU-gated 入口已对称命名为 `silu_gated_rmsnorm`，两种 gate 都使用直接乘数 gamma。
实现统一归 `src/ops/rmsnorm/`；两种 gate 的资格验证保留在
[同一个测试文件](../../tests/ops/rmsnorm/test_gated.cpp)，bench 统一由 `ninfer_rmsnorm_bench` 选择。
详见 [GDN §3.2](qwen3_8-flash-next-gdn.md#32-sigmoid-gated-rmsnorm)。

### A2. GDN 合并 AB 控制投影

已扩展 `gdn_control_proj` 的合并 parent 形式，并通过完整投影与非线性的独立数学验证：

| 输入/输出 | 表示 |
|---|---|
| x | BF16 `[2560,M]` |
| AB parent | BF16 `[96,2560]`，前 48 行 A、后 48 行 B |
| A_log、dt_bias | 各 FP32 `[48]` |
| g、beta | 各 FP32 `[48,M]` |

完整计算包括 AB 投影与 `g=-exp(A_log)*softplus(a+dt_bias)`、`beta=sigmoid(b)`。
入口直接消费 HC 产生的 x，采用不带 input norm 的形式。heads=48、input_rows=2560
的权重准备、容量查询及执行规划已补齐；该新几何不属于带 input norm 的入口支持域。

公共声明统一在 [GDN control 合同](../../include/ninfer/ops/gdn_control.h)，执行与容量查询
消费同一份 prepared `ProjectionWeights`；`GdnControlOutputs` 组织 g/beta 输出。
合并 parent 使用零拷贝 row views，权重准备不做运行时重排。
实现、测试与 bench 分别归 `src/ops/gdn_control/`、`tests/ops/gdn_control/`、
`bench/ops/gdn_control/`，bench 入口为 `ninfer_gdn_control_bench`。

### A3. HC 基础计算

已在 [gated_residual.h](../../include/ninfer/ops/gated_residual.h) 与 `src/ops/gated_residual/` 实现：

| 入口 | 完整效果 |
|---|---|
| `hc_expand_norm` | embedding 逐 word 展开为四条 residual streams，同时产生 grouped norm |
| `hc_grouped_rmsnorm` | 四组独立的 H=2560 offset RMSNorm |
| `hc_combine` | residual + FP32 injection × block output，输出 BF16 residual |
| `hc_combine_norm`，带 injection | 完成 combine，再对实际 BF16 residual 做下一组 HC norm |
| `hc_combine_norm`，无 injection | MTP 使用的 unit-weight broadcast-add，再做 grouped norm |

Residual/normalized 为 BF16 `[2560,4,M]`，block output 为 BF16 `[2560,M]`，
injection 为 FP32 `[4,M]`，已包含 `2*sigmoid(logit/4)`。Norm 参数是各 stream 独立的
BF16 offset。保留 HC 设计中的 residual exact-alias 规则与 BF16 residual→norm 边界。

这些入口处理本次 forward 的激活，不依赖 Main/MTP checkpoint，全部零 global workspace。
五种计算形式均通过完整独立 FP64 数学验证；两种 CombineNorm 另外验证实际 BF16 residual seam。
实现、测试与 bench 分别归 `src/ops/gated_residual/`、`tests/ops/gated_residual/`、
`bench/ops/gated_residual/`；公共 bench 为 `ninfer_gated_residual_bench`，包含明确的分阶段组合对照。
数学与交接见 [HC §6](qwen3_8-flash-next-hc.md#6-op-接口)，已实现接口以生产头文件为准。

### A4. HC Mix

实现普通与 final 两个 `hc_mix` 重载，以及对应 workspace capacity queries。

| 形式 | BF16 parents `[N,K]` | 公共输出 |
|---|---|---|
| Block Mix | down/injection `[324,10240]`、up `[10240,320]` | BF16 block input `[2560,M]`、FP32 injection `[4,M]` |
| Final Mix | down `[320,10240]`、up `[10240,320]` | BF16 sample hidden `[2560,M]` |

输入都是 normalized BF16 `[2560,4,M]`。完整 Op 包含 down、SiLU(a/4)、up、sigmoid
mixing 与四 stream 均值；普通形式另外产生 `2*sigmoid(b/4)`。私有中间量和内部 Linear
scratch 由同一 workspace 规划覆盖，query 覆盖声明的完整 M 区间。

可以在一个 Op 内组合已有 Linear 与私有 kernel，也可以继续融合优化；保持
[HC §6.3](qwen3_8-flash-next-hc.md#63-hc-mix-与-final-重载)的完整公共边界。
A3/A4 共用 HC 家族，由子工作区统一维护。

## 4. 第二批

### B1. PLE Gate

实现 [PLE §9](qwen3_8-flash-next-ple.md#9-gpu-接口) 已声明的 `ple_gate`，完整包含：

```text
K/Q offset norm → dot / sqrt(H) → sign/sqrt/sigmoid
→ shared V 广播门控 → BF16 U → NormConv → BF16 N
```

接口固定使用 `PleProjectionView`、`PleGateParameters`、`PleBatchView` 和
`PleConvRecordWindowView`：

- K/V 保留 `[12800,M]` parent 的真实 token/batch strides；residual 与 U 连续，
  形状为 BF16 `[2560,4,T,B]`。
- U 是公开 BF16 输出，NormConv 必须消费它的实际 words；N 的正式输出位置是
  `records.values[:, offsets[b]+t, b]`，feature extent 为 10240。
- 有效列产生 U/N；无效列 U 精确为 +0，records 对应位置保持原样。
  Op 只读取 Device valid_columns/offsets；记录预留、有效长度和前沿由主侧维护。
- 支持 B=1..8、合同允许的任意正 T，以及来自 pool 切片的 record strides。
  本入口不需要 caller-owned global workspace。

`include/ninfer/ops/ple.h` 的共享 views/声明由主工作区按设计先落地，两侧共用。
子工作区负责 `src/ops/ple/gate.*` 及必要的私有辅助文件；主工作区负责 conv、norm
重载和 state extraction 的实现文件。Gate 可直接用显式 Tensor/record buffers 调用。

### B2. PLE token hash

实现 [PLE §11.1](qwen3_8-flash-next-ple.md#111-纯-token-hash) 的
`PleHashParameters` 与 `ple_hash_rows`，位于 `src/models/qwen4_exp/ple_hash.h/.cpp`，
namespace 为 `ninfer::models::qwen4_exp`，实现编入 `ninfer_model_runtime`。

输入为已绑定的控制值、EOS、紧邻本段的两枚 raw 历史 ID，以及一段连续 input IDs。
输出由调用者提供，长度为 `16*N`，`row_ids[16*t+h]` 保存全局行号。
顺序固定为 2-gram 8 heads、3-gram 8 heads，所有整数运算遵循源 INT64 words 的语义。

函数按 §3 的 EOS 规则读取历史，段内后续位置消费本段更早的 raw IDs；分段与完整调用
等价。调用同步完成，输入和历史保持不变。该文件只实现 hash 数学，不依赖 Model、Program
或读取服务类。主侧提供追加起点的历史，并负责全局行号的去重、shard/page 映射与状态提交。

### B3. NF4 LUT 与行解码

实现 [PLE §6.2.1](qwen3_8-flash-next-ple.md#621-host-lut-与行解码接口) 的
`Nf4Bf16Lut`、`build_nf4_bf16_lut`、`decode_nf4_row_bf16`。
声明扩展已有 `src/core/nf4.h`，实现位于 `src/core/nf4.cpp`，归 `ninfer_core`。

每个实际 FP32 divisor 构造一张 4 KiB LUT：

```text
lut[scale*16+code] = BF16_RNE(codebook[code] * E4M3FN(scale) / divisor)
```

按精确表示值直接舍入到 BF16，保留零符号。每个 shard 一张，128 张共 512 KiB，
由 Model 持有。运行时行解码消费已有的紧凑行 bytes 与对应 LUT，直接写调用者的 BF16
输出 span。本实例为 90 B → 160 words，可直接写入 pinned E 的某个 head 区间。

行起点允许任意字节对齐，公开入口遵守完整 span 范围；SIMD 等优化留在实现内部。
全局行号→shard/page/file、读取缓存及输出 buffer 寿命由主侧读取服务负责。

## 5. 两个工作区的交接

主侧完整推进顺序见[主工作区任务清单](qwen3_8-flash-next-main-tasks.md)。其中 M1 优先
落地共享 `ple.h`，作为 B1 的直接前置；A1—A4、B2/B3 可按本清单独立推进。

| 边界 | 子工作区交付 | 主工作区负责 |
|---|---|---|
| HC | A3 基础计算已完成；A4 Mix 与其容量查询待实现 | 层间调用、激活寿命、Main/MTP 末端特征保存与恢复 |
| GDN | 已完成 A1 输出 norm、A2 控制投影的实现与数学资格 | 输入投影/snapshot/record 家族、36 层 fold、原生 GDN 层和状态接入 |
| PLE Gate | 从 K/V 与 residual 产生 U/N | shared views、record 预留、ConvAdd/Norm、state extraction |
| PLE hash | 有序全局行号 | 资源绑定、原始 ID 历史、候选窗口与提交、行/页调度 |
| NF4 decode | 不可变 LUT 和 Host 行解码 | divisor/backing 绑定、LUT 所有权、SSD/cache、pinned E、H2D |

子工作区完整维护各任务的实现、声明和对应验证；`ple.h` 共享接口由主侧维护。
构建与测试注册中的必要增量随任务提交。主工作区维护全局支持进度，合入已经完成的独立任务。

建议顺序为 A1 → A2 → A3 → A4 → B1/B2/B3；B2/B3 可以提前支持主侧读取流水。
B1 只需同步共享声明，B2/B3 只需本文件引用的纯数据接口，均不等待完整 Program 或 SSD 服务。

QSA/共享页组、GDN 状态路径、PLE I/O/卷积状态、MoE/expert 流式执行，以及 Loader、
Program、Vision/MTP 集成继续由主工作区推进。两个工作区各自构建；同卡 GPU 性能测量错开。
