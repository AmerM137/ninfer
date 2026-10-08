# Qwen3.8-Flash-Next：主工作区任务清单

更新日期：2026-10-08。本文维护主工作区的推进顺序、依赖和任务 1 的实施方案。
子工作区的七项独立计算见[子工作区任务清单](qwen3_8-flash-next-independent-ops.md)；
整体实现进度统一维护在[支持总览 §11](qwen3_8-flash-next-model.md#11-支持进度与后续工作)。
各专题继续拥有数学、存储和状态设计，具体 Op 合同落地后由生产头文件维护。

## 1. 推进清单

| ID | 任务 | 交付范围 | 依赖与交接 |
|---|---|---|---|
| M1 | PLE 共享头文件 | projection、batch、history、record views；Gate、ConvAdd、ConvAddNorm、StateExtract 声明及完整合同 | **最高优先级：解除子侧 B1 PLE Gate 的接口依赖**；实施方案见第 3 节 |
| M2 | 共享 KV 页组扩展 | `tokens_per_row=1/4`、Device/Host 布局、容量、COW、迁移与原 token coverage | 主 K/V 与 compressed index key 共用页组；为 M5 的存储和恢复提供基础 |
| M3 | GDN 输入投影与状态路径 | Q8 `[16384,2560]` 的独立 projection、snapshot、record 及容量查询；36 层 Replay fold | 主侧完整维护 input projection 家族；子侧 A1/A2 负责输出 norm 和 AB 控制投影 |
| M4 | PLE 卷积与状态提取 | `ple_conv_add`、`ple_conv_add_norm`、`ple_state_extract`；dilation=3、9 列历史、任意接受前缀 | 使用 M1 合同；与 B1 通过 BF16 U/N records 对接，可用显式输入独立实现 |
| M5 | QSA Op 家族 | Prepare、Indexer、Sparse GQA、tail extraction；projection views、选块结果、因果尾部与 cache 读写 | 使用 M2 页组；Indexer 与 Sparse GQA 保持独立、顺序消费 |
| M6 | PLE cold backing 与读取服务 | artifact 区间、controls/divisors/LUT 绑定；去重、缓存、异步读、pinned E、ticket 与 H2D 交接 | 数值部分接入子侧 B2 hash、B3 NF4 decode；Program 接入在 M8 完成 |
| M7 | MoE / expert 流式执行 | 先细化路由、缓存槽、搬运与计算接口，再实现 grouped GEMM/fused MoE、A16/A4 与传输重叠 | NVFP4 expert bank 已具备；完整执行合同仍需细化 |
| M8 | 原生 Qwen4Exp 执行接入 | 原生参数绑定、驻留、planner、StateImage、层编排与 Program；统一提交/恢复及 eager/Graph | 组合主侧能力与子侧 HC 等 Ops；先贯通 Text prefill/decode/replay/scoring，再接 Vision 与原生 MTP |

M1 单独交付后，主侧优先推进 M2/M3。其余按依赖开展；序号不要求所有任务串行等待。
M8 的绑定和规划可以随相应基础能力推进，完整 Text 路线需要 M7 的 MoE。
Qwen4Exp 使用自己的原生层编排，复用中央 Ops 和 core 存储机制。

## 2. 两个工作区的直接依赖

| 子侧任务 | 主侧前置条件 |
|---|---|
| A1—A4：输出 norm、AB 投影、HC | 无新增前置接口，可按既定专题直接实现 |
| B1：PLE Gate | 同步 M1 的 `include/ninfer/ops/ple.h` |
| B2：PLE hash | 无新增前置接口；按 PLE §11.1 实现纯 Host 函数 |
| B3：NF4 LUT/行解码 | 无新增前置接口；按 PLE §6.2.1 扩展 core 数值辅助 |

主侧维护共享 `ple.h`。子侧拥有 `src/ops/ple/gate.*` 及必要的私有辅助，主侧拥有卷积和
状态提取实现；各自随实现注册源文件。B1 可以直接构造 Tensor/record buffers 调用，
不等待 M4、SSD 服务或完整 Program。两侧 GPU 性能测量错开。

## 3. M1：PLE 共享头文件实施方案

### 3.1 交付与代码依赖

新增 `include/ninfer/ops/ple.h`，namespace 为 `ninfer::ops`，只需包含
`core/tensor.h` 和 `core/device.h`。按 [PLE §9](qwen3_8-flash-next-ple.md#9-gpu-接口)
落地六个结构和四个函数声明，保留已定名称、参数顺序和传参方式。
本任务交付可独立包含的共享合同；函数定义随 B1/M4 实现，不添加占位函数或源文件。

| 结构 | 字段和用途 |
|---|---|
| `PleBatchView` | `valid_columns`：Device I32 `[B]`，本次每行有效输入长度 n |
| `PleProjectionView` | `key`：BF16 `[2560,4,T,B]`；`value`：BF16 `[2560,T,B]`，保留合并 parent 的真实 strides |
| `PleGateParameters` | `norm_key`、`norm_query`、`norm_conv`：各 BF16 `[2560,4]` offset weights；`rms_epsilon=1e-6F` |
| `PleConvStateView` | `history`：BF16 `[10240,9,S]`；`slots`：Device I32 `[B]`，显式选择状态槽 |
| `PleConvRecordWindowView` | `values`：BF16 `[10240,R,B]`；`offsets`：Device I32 `[B]`，用于本次 Gate 写入和 Conv 读取 |
| `PleConvRecordReadView` | `values`：BF16 `[10240,R,B]`；`valid_columns`：Device I32 `[B]`，用于按基准提取接受前缀 |

所有结构均为非 owning views。`Tensor` 本身不区分可写内存，读写权限由每个函数合同定义。
卷积入口的下一 HC norm 仍接收显式 Tensor/epsilon，共享头文件无需包含 HC 或模型头文件。

### 3.2 头文件需要完整表达的合同

按 [Op 开发](op-development.md#3-contract-headers)组织数学、形状、数值、effects、workspace
和执行约束。从 PLE §4、§9、§10.2 整理下列内容，使实现者和 oracle 作者只读头文件即可
确定一次调用的效果。

| 入口 | 必须明确的完整效果 |
|---|---|
| `ple_gate` | 每 stream 的 K/Q offset norm、dot/sqrt(H)、sign/sqrt/sigmoid、共享 V 门控，产生 BF16 U；从实际 U words 做 NormConv，直接将 BF16 N 写到 records |
| `ple_conv_add` | 读取旧 history 与当前 N，做 4-tap、dilation=3 卷积及 SiLU，与 residual/U 相加，输出 BF16 updated residual；history 保持不变 |
| `ple_conv_add_norm` | 完成同一 ConvAdd，再从实际 BF16 updated residual 做下一 HC grouped offset norm，输出 BF16 normalized；两个输出均保留 |
| `ple_state_extract` | 按接受长度 a，从旧 history 与 records 精确复制新 9 列 history；包含 a=0 时向独立 destination 复制旧状态 |

三个数值入口使用 BF16 表示的输入；norm 沿 H=2560 独立处理四个 stream，gain 为 `1+w`。
Gate 的 floor 固定为 1e-6，独立于可传入的 RMS epsilon；`sign(0)=0`，对应 gate=0.5。
U、N、updated residual 是公开 BF16 边界，后继计算使用其实际 words；其余中间量不增加
公共舍入点。StateExtract 是逐 word 索引复制，不重新计算 norm 或卷积。

### 3.3 布局、记录与有效范围

- H=2560、streams=4、D=10240、history=9 固定；B=1..8，T/R/S 为正。
  T 支持 Tensor 与存储可表达的任意正值，4096 不是接口上限。
- `Tensor::nb` 的单位为 byte。合并输出 parent `[12800,M]` 中，K 的 token stride
  为 `2*12800`，V 从 parent 的 `2*10240` byte 偏移开始；二者 batch stride 均为
  `2*12800*T`。这是典型布局，接口按实际正 strides 访问，不能将 K/V 当成紧密子矩阵。
- residual/U/updated/normalized 连续且至少 16-byte 对齐；norm/kernel 连续。
  K 的 feature/stream 连续，V 的 feature 连续；history/records 的 feature 连续，
  其余轴保留合法、不重叠的正 strides，允许 pool 切片产生的间隙。
- Device metadata 均为连续 I32。窗口满足 `0<=n[b]<=T`、`0<=offsets[b]`、
  `offsets[b]+n[b]<=R`。Gate 将 N 写入 `values[d+2560*s,offsets[b]+t,b]`；
  Conv 使用同一个窗口，不能把 offset 当成 token 的绝对位置。
- Conv 的 tap j 读取局部位置 `k=t-9+3*j`：k<0 时从旧 history 的 `9+k` 列取值，
  否则从当前 records 的 `offsets[b]+k` 列取值。History 的九列从旧到新排列。
- StateExtract 的 read view 第 0 列对应其 before 的起点，满足 `0<=a[b]<=u[b]<=R`。
  输出第 j 列令 `k=a[b]-9+j`，k<0 取 before 的 `9+k`，否则取 records 的 k。
  before/after 可使用不同 pool 和不同槽数，分别按各自 view/slots 寻址。
- Gate/Conv 的无效输出列精确置 +0，不读取对应激活；Gate 不写无效 records，Conv 不读
  无效行的 state slot。StateExtract 每行的 source/destination slot 均须有效，a=0 也适用。

头文件区分 Host 可检查的 dtype/shape/strides 与调用者保证的 Device counts/indices。
入口不把 metadata 拷回 Host 再决定计算；执行从 Device metadata 读取本次有效范围。

### 3.4 Alias、执行与所有权

| 入口 | 可写范围与 alias |
|---|---|
| Gate | U 与 records 写区间互不重叠，且不覆盖任何输入；窗口外 records 保持原样 |
| ConvAdd / ConvAddNorm | updated 可与 residual 精确同址；其余写区间与仍需读取的输入不重叠，normalized 与 residual/updated 分离；禁止部分重叠 |
| StateExtract | destination slots 互异且不覆盖其他行的 source；只读 source 可以共享；同一行允许 before/after 精确同址，其余输入和输出不重叠；未选中槽保持原样 |

四个入口均无 caller-owned global workspace，省略 arena 和容量查询。计算和状态复制提交到
`DeviceExecutionView` 的 stream，可由多个有序 kernel 完成；不限定 launch 组织。
Graph replay 从稳定地址读取本次 Device metadata，地址与资源寿命由调用者保证。

Op 只计算和读写指定范围；Program 负责 record 预留、counts/offsets、槽选择、基准历史、
buffers 寿命以及 frontier 的采用。Gate/Conv 不更新规范 history；StateExtract 只写指定
destination，不发布请求状态。SSD、token hash、I/O ticket 和 Host raw-ID tail 留在各自层。

### 3.5 落地顺序与交接

1. 新增头文件，将上述完整合同与 PLE §9 的原始声明一起落地。
2. 将 PLE §9 的重复 C++ 声明替换为头文件链接，专题保留整层流程、状态关系和生命周期；
   同步总览、文档索引及子侧 B1 的入口说明，标明“声明已落地，计算实现待完成”。
3. 使用现有 C++/CUDA 编译配置检查头文件可独立包含。数值、状态和 Graph 验证随实际
   B1/M4 实现完成；本任务不增加测试目标或 benchmark。
4. 将同一份头文件同步到子工作区，B1 后续只新增 Gate 实现和所需构建注册。

完成条件是子侧能够直接包含该头文件、构造规定 views 并实现 `ple_gate`，无需再决定
record 布局、有效列或状态读写边界。M1 不改变 B2/B3 已约定的 Host 接口。
