# Qwen4Exp MoE GPU 计算设计

本文定义 Qwen3.8-Flash-Next 的 GPU MoE 计算。Main NVFP4 A4、MTP block-FP8 A8/G128、
BF16 router/shared 已在现有 MoE 入口实现。Expert streaming 与完整 Program 接入另行推进。

权重、binding 和驻留由[模型与权重](qwen4_exp-model.md)定义；表示值和地址分别沿用
[数值格式](tensor-formats.md)、[存储布局](storage-layouts.md)。本设计扩展现有
[MoE 合同](../../include/ninfer/ops/moe.h)，负责数学、GPU 接口、计算分解和调优目标。
Expert 上传、CPU 计算分工及 Program 编排继续由
[expert cache](expert-weight-cache.md)和[模型执行](qwen4_exp-execution.md)负责。

## 1. 本阶段的决定

1. 使用现有 `topk_softmax_route`、`routed_swiglu`、`topk_softmax_swiglu_moe` 三个语义入口，
   新增格式和几何支持；不增加模型专用 MoE 入口。
2. 直接消费已存储的 bank planes、每 expert divisor 和现有 affine/indexed 地址视图。
   持久权重布局不变；MTP recipe 与产物的 Use 许可已设为 `AllowA8`。
3. **Main 以 W4A4 NVFP4 Tensor Core 为主路径；MTP 以 W8A8 FP8 Tensor Core 为主路径。**
   A16 只作为小 M 的性能候选，数学验证通过且完整 Op 实测更快时保留。
4. 先建立低精度 Tensor Core 主路径，再比较小 M 的直接计算、窄 token tile 与较宽
   grouped tile。计算机制、精度和 token 区间分别设计，不把小 M 预先绑定到 A16/SIMT。
5. **Host 不根据实际路由选择实现。** 固定格式、几何、T、policy、输出形式和设备事实后，
   kernel 序列、实例、容量固定。路由只改变 GPU 内部的任务数据和有效范围。
6. 将 gate/up、SiLU/product、A4 或 A8 down 输入量化组织在同一生产 kernel；shared 与最终
   routed 归约在 FP32 中汇合，最终写 BF16。先保留清楚的 GU → down 依赖。

Main 现可独立调用完整 GPU MoE，也可用既有 routes/mask 计算一个 GPU 子集，
为后续混合执行提供 FP32 contribution。它不包含 CPU NVFP4/FP8 数学、动态上传调度，
也不代表完整 Qwen4Exp 模型已经接入。

## 2. 几何和数学

用 `H=2560`、`I=640`、`E=512`、`K=10` 表示 hidden、expert intermediate、expert 数及
top-k。Shared intermediate 同为 640。本文用 `M` 表示一次 Op 的总列数；上层
`[H,T,B]` 按 `m=t+T*b` 展平为 `[H,M]`，`M=T*B`。现有 MoE API 中的 `tokens/T` 对应这里的 M。
HC 的四个 streams 不属于 MoE batch，也不把 H 扩大为 10240。

### 2.1 闭合公式

对每列 `x_m`：

```text
l_e       = W_router[e,:] · x_m
ids_j     = topK(l)[j]                         // 降序，同分取较小 logical expert ID
a_j       = exp(l_ids_j - max_selected_l) / sum_j exp(l_ids_j - max_selected_l)

g_e       = W_gate_e · x_m
u_e       = W_up_e · x_m
h_e       = SiLU(g_e) * u_e
v_e       = W_down_e · h_e
routed_m  = sum_j a_j * v_ids_j

beta_m    = sigmoid(W_shared_score · x_m)
h_shared  = SiLU(W_shared_gate · x_m) * (W_shared_up · x_m)
shared_m  = beta_m * (W_shared_down · h_shared)
y_m       = routed_m + shared_m
```

`norm_topk_prob=true`，因此选中 logits 的 softmax 与“对全部 E 项 softmax，再选 K 项并
归一化”的实数公式相同。无需物化 `[E,M]` 概率。Shared 不参加 top-k 或归一化。
没有 bias、expert 输出 RMSNorm、clamp、额外 routed scaling factor 或 sigmoid expert 激活。

核对固定版本的
[Transformers 实现](https://github.com/huggingface/transformers/blob/8073e6dcaea9aa4b42acf2d92ec72148786bbaeb/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L812-L899)
后，MoE 主体仍是归一化 top-k SwiGLU 加 sigmoid-gated shared SwiGLU，与现有 35B-A3B
属于同一数学家族。差异集中在几何、表示、标定及外层 HC：

| 项目 | 现有 35B-A3B GPU MoE | 本实例 |
|---|---|---|
| H / I / E / K | 2048 / 512 / 256 / 8 | 2560 / 640 / 512 / 10 |
| Routed weights | Q4+Q5、Q4+Q6、Q8+Q8 | Main NVFP4；MTP block-FP8 |
| Shared weights | Q8 | BF16 |
| 缩放 | 整数分组 scale | Main G16 scale + 每矩阵 divisor；MTP 二维块 scale |
| 输出交接 | 普通 FFN/residual 调用 | 2560 维 block output，由后续 HC 回注四流 residual |

所以沿用语义接口，但不能只修改旧 kernel 的几个 shape 常量。

### 2.2 数值边界

数学合同沿用现有 MoE：输入是实际 BF16 x，权重是持久格式定义的表示值；router dot、
softmax、SiLU、加权归约用 FP32 实现，数学 oracle 用高精度计算。Routes 对外提供 I32 IDs
和 FP32 coefficients。最终输出为 BF16，子集 contribution 也允许 FP32。

上游 BF16 Linear、routing coefficient cast、逐 expert BF16 `index_add` 是源实现的
中间舍入，不新增为本 Op 的数学边界。尤其不为迁就源代码或另一条 kernel 路线，先将
FP32 logits、gate/up、product、expert 输出舍入到 BF16 再计算。近 top-k 边界和同分情形
单独验证，不能用普通输出误差掩盖选错 expert。

A4/A8 描述两次 expert 收缩的激活 operand 精度，输出、非线性和归约不会随之降到 4/8 bit。
本实例的计算约定为：

| 部分 | 主要激活表示 | 累积与后处理 | 性能候选 |
|---|---|---|---|
| Main GU/down | E2M1，沿 K 的 G16 E4M3 scale + 已有 tensor divisor | FP32 dot、缩放、SiLU/product 和加权和 | 小 M A16 |
| MTP GU/down | E4M3FN，每 token 沿 K 的 G128 动态 FP32 scale | 每 K128 的 FP32 dot、缩放、SiLU/product 和加权和 | 小 M A16 |
| Router/shared | BF16 权重和 A16 输入；内部 product 精度由融合实现决定 | FP32 累积、softmax/sigmoid 和汇合 | BF16 的直接/MMA 调度 |

`AllowA4/AllowA8` 允许小 M 采用 A16，不要求量化后再解码去模拟 A4/A8。不同计算路径
不要求逐位相同，均对同一高精度数学 oracle 验证，并使用对应 arithmetic profile 的数值预算。
A4 的 calibration 和 A8 的动态 scale 不进入理想公式；量化仿真用于解释误差，不能替代
完整 routed/shared 结果的数学验收。

Qwen4Exp 使用 `MoeEpilogue::Write` 输出 y，再由 HC combine 处理 10240 维 residual。
现有 `AddResidual` 仍只表示同形 `[H,M]` residual，不在 MoE 内实现 HC injection。

## 3. 已存储的权重与 GPU 消费

本次读取了实际产物 `/home/neroued/ninfer/out/qwen3_8_flash_next.ninfer` 的目录、Binding、
Uses 和小型 divisor 元数据，核对结果如下。

### 3.1 Main NVFP4

| Bank | Shape（持久 C-order） | 每 expert codes | 每 expert scales | Bank payload |
|---|---|---:|---:|---:|
| gate/up | `[512,1280,2560]` | 1,638,400 B | 204,800 B | 943,720,448 B |
| down | `[512,2560,640]` | 819,200 B | 102,400 B | 471,861,248 B |

Layout 为 `batched_block_scale_k16_m128x4_v1`。全部 experts 的 codes、全部 swizzled
scales、FP32 divisors `[512]` 分 plane 存放；不是 512 个带 trailer 的 rank-2 payload。
每 expert 内 gate 行 `[0,640)`、up 行 `[640,1280)`。640 是 128 的倍数，gate/up 边界
不切断 scale tile。

```text
W_e[n,k] = E2M1(code_e[n,k]) * E4M3FN(scale_e[n,k/16]) / d_w[e]
```

Divisor 按 logical expert ID 读取。实际第 0 层 gate/up 的 512 个 divisor 有 218 个不同值，
down 有 281 个不同值；把整个 bank 共用一个 divisor 会直接算错。
Scale swizzle 使用 expert 内局部 n/k。不能沿用整数 bank 的“flatten expert 到 row 后
按自然行序索引 scale”的假设。

Gate/up 共用该 expert 的一个 weight divisor。Gate/up、down 各有一个层内共享的
activation divisor，来自 FP32 `[48,2]` 辅助对象；实际第 0 层分别约为 719.799、313.927。
这两个数只用于 A4 的输入标定，与 `d_w[e]` 分开。GU 的输入量化可跨该 token 的 K 个
experts 复用；down 的输入 h 依赖 expert，只能逐 assignment 量化。

`ExpertMatrixDeviceView` 已能表达原 planes、affine strides、GPU offsets 和 divisor plane。
新增 kernel 同时支持全 resident 与 indexed cache view。访问顺序为：检查 subset 归属，
再取该 expert 的 offset/planes/divisor。未选中的 entry 即使未发布也不能被访问。

### 3.2 MTP block-FP8

几何相同，layout 为 `block_scale_m128_k128_v1`：自然 C-order E4M3FN codes 后接 BF16
scales；GU scale shape 为 `[512,10,20]`，down 为 `[512,20,5]`。

```text
W_e[n,k] = E4M3FN(code_e[n,k]) * BF16(scale_e[n/128,k/128])
```

GU/down 两个 bank 分别为 1,677,926,400 B 和 838,963,200 B；每 expert 合计 4,915,800 B。
没有 matrix divisor 或 A4 calibration。Gate/up 各自保留原 scale，不要求两半相等。

现有 Linear FP8 kernel 消费的是 row-scale，不能直接拿来处理这种沿 K 变化的 scale。
后续 FP8 MoE kernel 直接执行 W8A8 MMA，在每个 K128 block 内计算 raw-code dot，
将该 block 的 FP32 部分和乘 activation scale 与 weight scale，再累计。不能等整个 K
归约结束才乘一次 scale。动态 G128 激活编码和融合边界见 §6.6。

### 3.3 Router 与 shared

BF16 parent `[1793,2560]` 的行顺序是：

| 行区间 | 含义 |
|---|---|
| `[0,512)` | router |
| `[512,513)` | shared score |
| `[513,1153)` | shared gate |
| `[1153,1793)` | shared up |

`MoeWeights.control` 绑定前 513 行；`shared.gate_up` 绑定后 1280 行，另有 BF16
`shared.down [2560,640]`。全部是零拷贝 row views，保留原地址。
接口不要求 control 与 shared GU 物理相邻，完整 Op 的合并投影可通过两个视图读取。
不将 shared 填成第 513 个 routed expert，也不把 1793 行持久 padding 到整 tile。

### 3.4 来源数值方案与 Use

已核对本地 NVIDIA `config.json` 的 `quantization_config.config_groups`：Main experts
声明 4-bit/G16 激活，MTP experts 声明动态 8-bit/G128 激活。MTP 没有静态 input-scale
张量，是因为 scale 在执行时生成，不表示只能用 A16。

| 对象 | 当前 recipe/产物 | 本方案要求 |
|---|---|---|
| Main GU/down Uses | `AllowA4`，有 GU/down activation divisors | 沿用 |
| MTP GU/down Uses | `AllowA8` | 动态 G128，无新增 calibration 张量 |
| Router/shared Uses | `A16Only` | 沿用 |

`official_recipes.py` 已为 MTP expert 的逻辑 gate、up、down 三个角色显式设置
`AllowA8`，现有产物的对应 Use 已同步。原 codes/scales、Binding 和 payload 布局保留；
这完成了计算许可准备，A8 GPU 数学仍按后续步骤实现。

## 4. 接口与所有权

### 4.1 保留的接口形态

以下为现有接口的目标用法，省略未变的重载：

```cpp
void topk_softmax_route(
    const Tensor& x, const RouteProfile& profile, const Weight& control,
    ExpertRoutes& routes, Tensor& shared_coefficients,
    MemoryArena& workspace, CudaExecutionView execution);

void routed_swiglu(
    const Tensor& x, const RoutedSwiGLUProfile& profile,
    const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
    ExpertSubset subset, MoeEpilogue epilogue, Tensor& destination,
    MemoryArena& workspace, CudaExecutionView execution);

void topk_softmax_swiglu_moe(
    const Tensor& x, const MoeProfile& profile, const MoeWeights& weights,
    MoeEpilogue epilogue, Tensor& destination, MemoryArena& workspace,
    CudaExecutionView execution, const MoeExecutionHints& hints = {});
```

同时扩展已有“routed + `ScaledSwiGLUTerm`”重载，使 GPU 子集能够携带恰好一次 BF16 shared。
`ExpertRoutes.ids/coefficients` 分别为连续 I32/FP32 `[10,M]`；shared coefficient 为 FP32 `[M]`。
`x/destination` 为连续 `[2560,M]`，遵守原有对齐、位置与 alias 约束。

`ExpertSubsetKind::{All,None,MaskOnes,MaskZeros}` 保留原语义：mask 为 GPU I32 `[512]`，子集
直接使用原 coefficients，不重新归一化。`None` 不读 routed weights/routes，有 shared 时
仍计算 shared。FP32 destination 只用 Write；BF16 可 Write 或 AddResidual。

所需的是扩展 `RouteProfile`、`SwiGLUProfile` 的准入及实现，不增加新字段：

| Profile | 注册内容 |
|---|---|
| Route | H2560 / E512 / K10，BF16 control |
| Main routed | H2560 / I640 / E512 / K10，NVFP4 GU/down，`BlockScaleK16M128x4`，允许 A4 |
| MTP routed | 相同几何，block-FP8 GU/down，`BlockScaleM128K128`，允许 A8 |
| Shared | H2560 / I640，BF16 GU/down |

Rank-3 bank 的矩阵 layout 已由 `routed_swiglu_profile` 规范化。Main 新 profile 要求
GU/down 均允许 A4，MTP 要求均允许 A8；相关 gate/up Uses 融合前取交集。各主路径覆盖
全部正 M，小 M 可以在这些许可内选择 A16。这里不因有 A16 私有候选，就额外注册一个
要求全 M 支持的 `A16Only` profile。未准入的 policy 组合由 query 和执行入口一致拒绝。

NVFP4 的 `ExpertActivationDivisors` 提供 GU/down 两个正有限标定值；block-FP8 不设置
它们，动态 scales 由 Op 写入 workspace。不增加代表 A8 scale 的 Host scalar 字段，
也不从 NVFP4 weight divisor 推导 activation divisor。

### 4.2 Workspace、状态与 Graph

继续由三组 `*_workspace_capacity_bytes` 查询完整区间容量；query 和执行使用同一私有
Plan、相同分配规则。容量不依赖实际 route values、活跃 experts、mask 命中数或 cache 命中数。
SM 数影响提交的 CTA 数，不影响 scratch 容量，现有查询无需新增 execution 参数。
大于内部工作窗口的 M 允许切片，但不收紧公共正 M 语义；切片大小属于性能设计，不能照搬
旧模型的常量而无视权重复用损失。

本 Op 没有持久状态。Model 持有不可变权重；Program 持有 destination、routes、scratch、
cache leases 和 Graph。权重 readiness 在调用前建立，Op 内没有 Host 同步、H2D 或隐藏设备分配。
所有依赖在同一个传入 execution stream 上成立。

固定一次 capture 的 profile、M、views 和 backing 后：

- x、route IDs/coefficients、mask、已发布的 offset contents 可以在 replay 前更新；
- GPU count/scan/scatter 每次重新建立当前 assignment 描述，不复用上次有效长度；
- Host 不读取 `n_e/U`，不据此改换 kernel、精度、grid 上界或增删节点；
- grid 按静态容量或固定 CTA 数提交，GPU 用实际 job count 限定循环；
- 空 expert、空 mask、相同 expert 集合或极端偏斜只是任务数据变化；
- backing/offset 更新必须在旧消费者结束后进行，既有 cache acquire/release 负责寿命。

既有 35B 实现在固定节点内部读取 `direct/spread` 标志、让某些 kernel 跳过工作，也能兼容
Graph。本轮采用更直接的约束：**新路径不按路由切换实现族或 A16/A8/A4；分布只用于组织工作。**
不同 M 的调优计划由 Program 的实际 Graph 变体承载，不承诺不同 M 自动共享同一捕获。

## 5. 负载模型：M 与 expert 的有效列数

令 `R=K*M` 为全部 assignments，`n_e` 为 expert e 在当前子集中的 assignment 数，
`U=count(n_e>0)`。同 token 的 IDs 不重复，因此单 expert 的紧上界为 `n_e<=M`。
实际 contraction 是 `W_e @ X_e`，其 token 维度是 `n_e`。

下面只是假定各 token 独立、均匀选 10 个 expert 的分析值，不是真实路由统计：

| M | R | 全 512 experts 的平均 n_e | 预期 U |
|---:|---:|---:|---:|
| 1 | 10 | 0.020 | 10 |
| 8 | 80 | 0.156 | 74.7 |
| 64 | 640 | 1.25 | 367.1 |
| 128 | 1280 | 2.5 | 471.0 |
| 512 | 5120 | 10 | 512.0 |
| 1024 | 10240 | 20 | 512.0 |
| 2048 | 20480 | 40 | 512.0 |
| 4096 | 40960 | 80 | 512.0 |

因此 M=1024 并不是 512 个各有 1024 列的 GEMM。若直接给每个 expert 一个 128-token tile，
大部分工作落在 padding。相反，全部 token 选择同十个 experts 时，只有十个矩阵，但每个
有 M 列，权重复用高，需要把每个 expert 拆成多个 token/output-row tiles 充分占满 SM。

调优覆盖三类真实使用范围：decode 的 M=1..8；verify/control 的 M=8..128，包含八路
短窗口；prefill/replay 的 M=256..4096 及尾块。B 只影响 M 和输入内容，没有独立 batch
路径。性能比较必须在同一 M 下覆盖均匀分散、热点集中、热点加长尾，不能用一种分布定全部阈值。

## 6. GPU 数据流与 kernel 分工

### 6.1 Router/shared 前段

独立 route 计算 BF16 `[513,2560]` 投影及 FP32 top-10/shared sigmoid。小 M 比较直接 dot
与小 tile MMA；大 M 用 BF16 MMA。Top-k 可以复用现有排序/归一化机制并实例化 E512/K10，
score stride、分块 K 和 workspace 都按新几何计算，不能保留 2048/256/8 的隐含常量。

完整 Op 允许在一个投影 launch 中同时算 control 和 shared GU，分别写 route scratch 与
融合 SiLU/product 后的 shared activation；无需公开 `[1793,M]` 中间 Tensor。合并投影
仍须正确处理 513 行边界，以及 gate/up 相距 640 行的配对。

独立 route 入口只算 control，尽早发布 routes，便于以后 CPU/GPU 分工或准备 expert。
完整 Op 则依据其自身测量决定合并投影，不要求两种入口共用调优记录。

### 6.2 两种格式的主流水

```text
x BF16 [H,M] → A4 G16 / A8 G128 量化一次 → Xq
routes + subset → GPU assignment maps              ↓ 按 token gather
                          quantized W_gate/up × Xq
                            → FP32 部分和按格式缩放/累积 → gate/up → SiLU/product
                            → 同 kernel 编码 Hq，按 assignment 保存
                          quantized W_down × Hq
                            → FP32 部分和按格式缩放/累积 → 系数加权 + shared → destination
```

Xq 的量化不依赖 expert，供所有 GU jobs 复用；Hq 则属于 `(token,expert)` assignment，
由对应 GU epilogue 生成。GU 必须先完成其全部 K reduction 和缩放，才能进入 SiLU。
Down 之后再乘 route coefficient，不先将它乘入 product。

选择窄 tile 时仍执行相同的 A4/A8 数值方案。对特别小 M，评估省去 grouping、按 route
slot 直接生成工作的版本；独立量化与片上量化也按完整流水成本比较。A16 的直接计算
候选放在主路径建立后实现，见 §6.7。

### 6.3 Grouping：只生成任务，不决定实现

对需要复用权重的计划，GPU 内建立紧凑 expert-major assignment 列表：

```text
每次 replay 清零计数与队列游标
  → routes + subset
  → count[e]
  → prefix offsets[E+1]、固定 tile 尺寸下的 job ranges
  → packed_assignment[R_selected]、inverse[K*M]
  → GU jobs → down jobs → 按 token 收集并归约
```

`assignment=m*K+j`，packed 条目能恢复 token、route slot 和系数。未选 assignment 的
inverse 为 -1。对每个 expert，job 描述其局部 token 起点和有效列数；输出特征 tile
由 contraction kernel 的工作编号展开。

首版在固定的清零节点后执行 count → scan → scatter 三阶段。Count 使用 CTA 局部
直方图合并全局计数；scan 同时初始化 scatter cursors；scatter
按 CTA/expert 取得连续区间后写各 assignment，避免热点 expert 的每个元素都争用一个
全局 atomic。Atomic 仅用于编组地址，不改变 routes 的系数或最终求和槽位。
完整 Op 可将 top-k 发布与 count 合并，消除重复读取。

Maps 为 O(R+E)，不建立 `[E,M,H]` 激活，也不把每个 expert 的 token axis padding 到 M。
固定 token tile 宽 b 时，job 上界可用 `ceil(R/b)+E`；有效 jobs 由 GPU 写出。核函数用
固定 grid 的 grid-stride 工作循环消费它，按 expert 相邻安排多个 tiles 争取 L2 复用。
需要动态 work queue 时，计数器也位于预分配 scratch，仍是相同 kernel 的 GPU 内部工作分配。

### 6.4 窄 tile 与较宽 grouped Tensor Core

使用两类调优区域，但复用主循环：

| 路径 | token tile 候选 | 主要问题 |
|---|---|---|
| 窄 grouped | 8、16，必要时 32 | 避免大量单列/少列 expert 导致无效 MMA，摊薄编组开销 |
| 较宽 grouped | 32、64 | 增加权重复用，平衡 stages、寄存器、SM 波次及 output traffic |

这些是候选集合，不是全部长期保留的实例。Host 依据 M 选择固定计划，GU/down 可使用
不同 tile，因为 K 分别为 2560/640，输出宽度与瓶颈不同。`n_e` 只限制有效列、生成更多
或更少 jobs；同一 capture 不根据它切换 tile 模板。

GU 每 CTA 负责一个 expert、一个 token tile 和一组配对 gate/up 行。完整 K reduction
后立即计算 SiLU/product。A4 可按 64/128 个 product features 分块；A8 的融合 epilogue
需要完整 128 个 product features，对应 128 行 gate + 128 行 up。两者各自选择行 tile、
寄存器和 shared-memory 预算，不强行共用一种 Schedule。量化 scale group 不跨 CTA。
Down 的 K=640 对 K64/K128 都完整；选择 K256 的候选必须显式处理最后 128，不能截断。

最初采用 `cp.async` 配合 shared-memory staging。它能直接 gather token，且能消费 replay
时变化的 GPU expert offsets。现有 Linear 的 TMA wrapper 在 Host 构造矩阵描述符，不能
直接套到动态 indexed expert 地址；只有具体 profiling 证明复制机制限制吞吐时，再设计
同时满足寻址和 Graph 的 TMA 路线。

### 6.5 Main NVFP4 A4

对每个 K16 激活组 z 和显式 divisor d_a，使用同一个标定 profile：

```text
s     = RN_satfinite_E4M3(d_a * max(abs(z)) / 6)
q_i   = RN_satfinite_E2M1(z_i * d_a / decode(s))   // s=0 时 q 全零
z_hat = decode(q_i) * decode(s) / d_a
```

不使用 KV cache 的“最小非零 scale clamp”量化 profile。公共量化工具新增能直接接收
FP32 fragment 的同 profile 入口，现有 BF16 输入入口借用它，避免为 h 强加 BF16 中间舍入。

GU/down 分别在完整 dot 后乘 `1/(d_a*d_w[e])`。GU 恢复后的 g/u 才进入非线性。
Product 直接从 FP32 fragment 量化为 down 输入，不增加 BF16 staging。其量化结果按
packed assignment 保存，GU/down 共享同一映射。

Activation codes/scales 属于私有 scratch，不复用持久 weight scale swizzle 的地址公式。
初版用逐列连续 codes、逐列自然 G16 scales，装入 shared 时形成 MMA 需要的 tile 排列。
X 每列 1280 B codes + 160 B scales；H 每 assignment 320 B codes + 40 B scales。
Down scale 行步长 40 B，加载不能假设每行都 16-byte 对齐。

使用已有 `common::mma_nvfp4_e4m3` 的 `m16n8k64` block-scaled MMA。M 很小时，独立
量化 launch 与片上量化是完整 Op 的两种候选；比较时保持 scale/rounding profile 相同。
GU 的预量化 X 可以跨 experts 复用；片上量化则可能在不同 expert/output tiles 重做，
两种成本都计入结果。

Main 当前实现以权重作为 MMA 的 m16 轴、token 作为 n8 轴；GU 的同一 warp 配对
处理 gate/up，FP32 product 按 16 features 做 warp 内 amax、编码和打包。Codes 使用
XOR shared swizzle，weight scales 按存储 quartile 向量读取，K128 双缓冲覆盖复制与 MMA。
整块无效的 n8 子 tile 在 kernel 内跳过；部分有效 tile 零填充，CTA 同步仍由所有线程参与。

M=1 的直接 A16 路径保留 FP32 product，并在 down 内归约十个 routes。其他 M 使用
GPU grouping 和 A4：小批量以一个 CTA 完成计数/scan/scatter，大批量分阶段完成。
私有 shape policy 分别选择 GU/down 的 token tile。
较宽的 GU 与较窄的 down 共用一份 job 表，down 在 kernel 内拆分 job，不重复分组。
具体阈值由各 Op 的 shape policy 维护，不由路由直方图或 Host readback 决定。

完整入口融合 BF16 control/shared GU，并在大批量 top-k 中产生 grouping 所需的计数。
Shared product 以 FP32 或 BF16 high/low 两个平面交接；shared down 合并 routed FP32 和后
再执行最终输出舍入。独立 routed/shared 重载沿用相同数学和存储约定。

### 6.6 MTP block-FP8 A8

每个 token 沿输入 K 轴分 G128 动态量化，scale 存 FP32：

```text
s_a   = RN_FP32(max(abs(z[0:128])) / 448)
q_i   = RN_satfinite_E4M3(z_i / s_a)
z_hat = decode(q_i) * s_a
```

`s_a=0` 时 codes 全零。Scale 不舍入到 BF16，不强制变成二次幂；MTP 没有额外 tensor
divisor。现有 row-FP8 A8 的整 token 一个 scale 不满足这个 profile，需要 G128 的量化
工具；X 的 BF16 读取和 H 的 FP32 fragment 编码共享其数学规则。

对输出行 n、token m 和 K128 block b：

```text
p_b[n,m] = sum_{k=128*b}^{128*b+127} decode(q_w[n,k]) * decode(q_a[m,k])
y[n,m]   = sum_b p_b[n,m] * s_w[n/128,b] * s_a[m,b]
```

使用 `common::mma_fp8_e4m3` 的原生 E4M3 MMA。当前 primitive 使用单位硬件
scale；每个 K128 block 以四个 K32 MMA step 形成 FP32 partial，再在 FP32 中乘真实
`s_w*s_a` 并累计到最终 accumulator。每块重置 partial，不能先跨 K128 累加 raw dot
再统一缩放，也不能将任意 BF16/FP32 scale 替换成 UE8M0 scale。

当前内核使用 K128 双缓冲和绕过 L1 的 `cp.async`，每个 n8 fragment 的 partial 在
当前 K128 内完成并缩放，不保留整张 tile 的第二套 accumulator。GU/down 的列宽由
各 Op 的 shape policy 独立选择，共享 job 表，较窄的 down 消费其子 tile。

GU 的 K=2560 对应 20 个 blocks，down 的 K=640 对应 5 个 blocks；gate/up 两半的
weight scales 可以不同，各自恢复后再计算 product。GU epilogue 必须在同一 CTA 内得到
连续 128 维 FP32 product，归约 amax 并写出 A8 codes/scale。允许 CTA 内分子 tile 计算，
不允许两个独立 CTA 各量化 64 维却宣称仍是 G128。若为了某个小 M 调度需要独立量化
阶段，保留同样的 G128 数学，并将额外 launch/流量计入候选比较。

X 每列有 2560 B codes + 20 个 FP32 scales，共 2640 B；H 每 assignment 有 640 B
codes + 5 个 FP32 scales，共 660 B。Scales 按 token/assignment、再按 K128 block 连续
保存。H scale 行步长为 20 B，只保证 4-byte 行对齐，不能按每行 16-byte 对齐加载。

### 6.7 小 M 的 A16 候选

A16 在主路径可测量后加入小 M sweep，比较完整 route/quantize/GU/down/reduce 成本。
它可使用直接 SIMT，或解码后的小 tile BF16 MMA；只保留有明确获益的区间，不开展大 M
A16 实现和调优，也不将其作为 A4/A8 数学验证的 oracle。

当前 Main 在 M=1、MTP 在 M≤2 使用直接 A16；其余走 A4/A8。
直接路径从 routes 取 expert，GU 完成 dot、缩放和 SiLU/product；down 按 token/output-row
tile 合并 K 项贡献，避免 grouped 路径的大型 expert-output scratch。SIMT 可直接消费
FP32 product。若候选采用 BF16 MMA，product 的私有表示要通过同一数学标准；不预设
需要全套 high/low 两平面机制。

Main 的 `E2M1*E4M3` 可精确扩展为 BF16，`d_w[e]` 留在 FP32 dot 出口。MTP A16
精确扩展 E4M3 codes，每个 K128 partial 分别乘该块 weight scale。两者均保持原权重
表示；使用 A16 时跳过激活量化，不量化后再还原。Shared 始终按其 BF16/A16 profile 计算。

## 7. 融合出口与临时内存

固定的核心融合为 GU + SiLU/product，以及 A4/A8 路径的 product quantization。
GU → down 首版保留 kernel 边界：down 需要完整 640 维 product，一般 GU tile 只产生其中
一部分，不能用跨 CTA 自旋模拟完成屏障。为“单 kernel”把整个 expert 放进一个 CTA
会限制并行度、扩大 shared/register 工作集，只有后续证据支持才增加这种机制。

直接路径的 down 在一个输出 tile 内归约 K 项。Main/MTP 小批量 grouped 路径由 down 写
FP32 `V[H,R_selected]`，再根据 inverse 和原系数按 token 求和；较大批量在 down epilogue
乘 route coefficient，直接 atomic-add 到 FP32 `[H,M]`。该 buffer 的清零并入输入量化，
不需要 inverse map。Shared 在最终 FP32 汇合点加入，最后写 destination。

完整 partial 方案在 M=4096 时需要 **400 MiB**，一次写加一次读为 **800 MiB**；Main/MTP
已选择的 atomic 出口只需 **40 MiB** accumulator。两种主路径的紧凑激活 payload 如下，
实际 query 另计对齐、shared、路由、
编组和已选择的 tile 所需 padding：

| M=4096，R=40960 | X codes + scales | H codes + scales |
|---|---:|---:|
| Main A4 G16 | 5.625 MiB | 14.0625 MiB |
| MTP A8 G128 | 10.3125 MiB | 25.78125 MiB |

Xq 在全部 GU jobs 后可复用，Hq 在全部 down jobs 后可复用；query 与执行按这些寿命规划。
一次调用只分配选定 profile 的激活表示，不同时预留 A16、A4、A8 buffers。

两种格式的出口保留以下取舍：

- FP32 partial + gather 是最初完整实现，归约顺序清楚，容易验证 mask/shared 行为。
- 若 partial 流量显著限制完整 Op，比较 down 的 coefficient epilogue 直接 atomic-add
  到 FP32 `[H,M]` accumulation buffer，再一次加入 shared/舍入的固定流水。
  它将 accumulator 降至 M=4096 时的 40 MiB，但需先清零，且有 atomic 竞争和求和顺序变化。
  Main 已对完整输出误差和各分布性能验证后采用；求和顺序不承诺逐位确定。
- 仅为降低峰值容量，可按 H 面板串行 down/reduce，保持全部 M 的 GU 激活；它不减少总
  partial traffic，并增加 launch，不能当作免费的性能优化。

以上出口候选也只能按 Host 已知的 profile/M 选择。不能看到热点路由后临时启用另一套
归约实现。不要通过把大 M 拆成大量小 token chunks 来规避内存账目，那会反复读取同层权重。

## 8. 性能目标与 roofline

本任务是增加新格式/新几何能力，没有已经优化好的同形 MoE 基线。临时首版只用于定位
各阶段成本；最终报告给出绝对延迟、有效吞吐和 roofline，不用百分比提升代替性能判断。

每个 routed assignment 的三个投影共：

```text
F_assignment = 6*H*I = 9,830,400 FLOP
F_routed     = 98,304,000*M FLOP                 // All，10 个 experts
W_one_expert = 2,764,800 B                       // Main codes + scales，不含 8 B divisor
B_unique     = U*W_one_expert
```

完整 Op 另计 router、shared、激活准备和归约。All 时 router + shared 的线性 FLOP 为
`12,456,960*M`，总线性有效计算为 `110,760,960*M`。SiLU、softmax 和索引成本单独统计。

假设每个 expert 的权重只从 DRAM 读一次，Main routed 的理想强度约为
`3.556*(R/U) FLOP/B`。实际 kernel 会按 token tile 重读权重，估算式为：

```text
B_tiled ≈ 1,843,200 * sum_e ceil(n_e / b_GU)
        +   921,600 * sum_e ceil(n_e / b_down)
```

MTP 的计算量相同，单 expert 权重为 4,915,800 B；其 GU/down tile 权重系数分别改为
3,277,200 / 1,638,600 B，理想权重强度约为 `2.000*(R/U) FLOP/B`。
这是 tile 权重需求，不能直接称为实测 DRAM traffic；跨 CTA 的 L2 复用、输出和其他输入
会改变 counters。Main A4、MTP A8 和小 M A16 分别使用 NVFP4、FP8、SIMT/BF16 MMA
的算力 roofline，完整 Op 的量化和 scale 应用成本也计入延迟。

仅举传输模型：若达到 1.6 TB/s 的持续 DRAM 带宽，十个冷 expert 的 27.648 MB 权重需要
约 17.3 μs，全 512 个的 1.416 GB 需要约 884.7 μs。这不是 Op 延迟承诺，还没有计入
shared/control、launch、解码、量化和 reduction；热 L2 也不适用这个 DRAM 下限。

最终按以下问题验收性能：

| 场景 | 优化目标及解释依据 |
|---|---|
| M=1..8 | launch 与稀疏权重读取主导；看完整 Op 延迟、实际读取 bytes、SIMT/窄 MMA 的成本 |
| Verify 小窗口 | 编组成本不能吞掉复用收益；同 M 下比较 independent/hot/mixed 分布，检查阈值断点 |
| M=1024/2048/4096 | NVFP4/FP8 MMA 的有效吞吐、tile 利用、DRAM/L2、量化、K128 scale 与 partial 流量，GU/down 分阶段归因 |
| 集中分布 | 少 expert 仍有足够 token/output tiles；避免仅按 expert 数分配 CTA 导致空闲 |
| 分散/长尾 | 空专家不访问权重；少列专家不导致成片大 tile 浪费；队列、tail 和 gather 成本可解释 |
| Indexed/subset | 同数学下正确跳过缺席 entry，寻址与 mask 成本可量化，FP32 contribution 可直接衔接混合执行 |

完整 Op 记录 median/P95、workspace 和 cold/warm 条件。同一组 routes 反复 replay 可能
把全部热点 weights 留在 L2，必须与冷权重条件分开。Nsight 仅用于解释具体瓶颈，报告
实际 DRAM bytes/BW、L2 traffic、Tensor Core 指令吞吐及寄存器/spill。
同时列有效 FLOP/s 和 padded MMA 工作量，不能把尾 tile 的无效运算算成模型有效吞吐。

## 9. 实现组织与复用边界

继续使用 `src/ops/moe/` 的三个语义目录，各自拥有 selector、shape policy 和容量规则。
新增 `h2560_i640_e512_k10` 的 NVFP4/block-FP8 policies，以及 H2560/E512/K10 router policy。
完整 Op 与独立 routed Op 共享计算模板，不共享私有阈值。

CUDA kernel 库保留现有整数实现。`block_scaled/` 统一维护紧凑分组、scratch、BF16
control/shared 与 FP32 输出汇合；`nvfp4/`、`fp8/` 分别维护编码、原生 MMA 和小 M A16。
两种浮点格式的 scratch 不进入整数 kernel 的描述符，没有新增公开 Op 或模型专用入口。

`operands.cpp` 统一负责格式字节数、divisor、profile 和 shared 的校验；新 profile 显式
进入自己的 shape policy。独立 routed 与完整入口分别维护调优选择，共用 kernel 模板。

复用 `ops/common` 的 codec、向量转换、MMA、copy/reduction 原语；NVFP4 tile staging 与
配对 epilogue 可从 Linear kernel 库抽取合适的计算体，MoE 自己拥有 assignment 寻址和
工作调度。不要在 MoE 内调 Linear 的 private shape dispatch，也不要在 Host 上逐 expert
调用它。Block-FP8 scale addressing 是新计算能力，不能冒充已有 row-FP8。
新增 FP32-fragment NVFP4 编码与 G128 FP8 编码放在 `ops/common/quant/` 既有格式文件中；
MMA scale 应用、assignment 寻址和 GU epilogue 属于 MoE kernel 库。
两种原生低精度 MMA 的编译单元按现有 `ninfer_nvfp4_non_rdc` 边界注册。

## 10. 实施顺序

| 步骤 | 交付与重点 |
|---|---|
| G1：接口与数值准备（已实现） | 固定新 profile 的 policy、bank/divisor 校验和 workspace 规则；准备独立解码/完整数学 oracle 和 BF16 route/shared。MTP recipe/Use 许可已完成；A4/G16 与 A8/G128 数值工具随其首个消费路径完成，不先实现全套 A16。 |
| G2：Main A4 主路径（已实现） | 紧凑 grouping、一次 X 量化、NVFP4 MMA GU + SiLU/product + H 量化、NVFP4 down 和最终归约，贯通三个公开入口、subset 与 Graph。A4 路线功能覆盖全部正 M；先在 M=1024/4096 验证数学和完整 Op 的 roofline。 |
| G3：Main 各负载优化（已实现） | 保留 decode A16、按 M 选择的窄/宽 A4，以及不同宽度的 GU/down；control/shared、top-k/count 融合和大批量 FP32 atomic 出口已实现。调优覆盖分散、热点、混合分布及 cold/warm，生产分派不读取实际路由。 |
| G4：MTP A8 主路径（已实现） | 在同一接口/maps/shared 基础上实现动态 G128 X、原生 FP8 MMA 的 K128 partial 缩放、128 维 product epilogue 编码及 A8 down。独立资格验证，覆盖小 M、prompt 初始化和 Graph。 |
| G5：MTP 各负载优化（已实现） | 测量 A8 窄/宽 tile、128 维融合 epilogue 的资源成本和完整出口；小 M 比较 A16，按收益保留。共用适合的计算体，独立确定 MTP 阈值，不照搬 Main 的 winner。 |
| G6：收尾（已完成） | 通过公共入口检查实际生产 routes、Graph 新内容 replay、精度、容量和性能曲线；全量编译，整理源码/测试/bench，更新完成状态。 |

G1 是必要准备；G2/G4 各自产出可独立调用的低精度主路径，G3/G5 完成其负载优化。
数学验证通过才比较性能；A16 未获胜就不进入生产 dispatch。各阶段不要求临时 stage
成为独立长期 Op，也不把普通 Linear 调优或完整模型加载作为前置任务。

## 11. 必要的正确性与测量覆盖

沿用现有 MoE test/bench 入口，扩展真实 profile，避免再建一套模型专用框架。

- 独立 oracle 按持久 words 解码，计算高精度路由和完整 SwiGLU/shared/subset；A4/A8 也与该
  oracle 比。独立 activation quantization 仿真用于分辨表示误差和实现错误，不替代验收。
- 重点覆盖不同 expert divisor、GU/down 不同 activation divisor、NVFP4 swizzle 跨 tile；
  A8 覆盖零组、同 token 各 G128 组幅度不同、K128 scale 变化，以及 gate/up 在 640 行处
  的正确配对。检查 A8 同组 scale 覆盖完整 128 维，发现误用整 token 或两个 G64 的情况。
- 覆盖 top-k 同分/清楚的近边界、路由系数归一化、shared 只计一次、mask 两侧相加、None、
  FP32 Write/BF16 Write/AddResidual 和输入保留。数值预算由 arithmetic profile 拥有，
  不能对失败样本单独放宽。
- 同一次 Graph 依次换分散/集中/mixed routes、mask 和 indexed offsets，直接对 oracle
  验证输出；不重新 capture 来绕过不同分布。未选 entry 使用不可读哨兵检验访问约束。
- 数学样本取代表性 extent、每个最终 dispatch 边界及内部切片尾部；完整 E/M 性能样本
  单独测量。无需在每个大 M 做昂贵的 CPU 全矩阵 oracle。
- 性能点以 M=1、2、4、8、16、32、64、128、256、512、1024、2048、4096 为主体，再加
  最终阈值相邻点。低 T 重点包含八路 verify；分布至少有独立均匀、相同十个 expert、
  热点加长尾。记录实际 U、n_e 分布和 tile 填充，不把它们传给 Host selector。

长期测试不加载整份百 GB 模型，不读取 PLE，不触发转换或下载。需要验证真实校准值时，
使用显式选定的少量 expert/元数据作补充样本；性能 fixture 按真实编码几何生成。
本轮设计尚无这些新路径的实测延迟、精度或模型质量结论。
