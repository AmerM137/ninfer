# Qwen4Exp 模型与权重

Qwen3.8-Flash-Next 是本合同的 checkpoint 实例，数学架构为 Qwen4Exp。现有
`qwen3_8_flash_next` recipe 已完成 Text、可选 Vision 和原生 MTP 的 V3 `.ninfer` 转换；
Qwen4Exp 尚未接入 Engine。Op 与模型执行的完成范围、状态组合和剩余需求统一见
[执行进度](qwen4_exp-execution.md#implementation-status)，PLE 的 Host 数学和读取流水见
[PLE I/O](qwen4_exp-ple-io.md)。

本文维护模型结构、实例 recipe、物理 parent 和驻留方案。通用表示值由
[数值格式](tensor-formats.md)定义，packing、padding 和字节地址由
[存储布局](storage-layouts.md)定义，文件集合与 Binding/Use 由
[容器合同](artifact-container.md)定义。转换命令见[权重转换](../weight-conversion.md)。
精确 config 字段、逻辑参数名和源映射由
[Qwen4Exp adapter](../../tools/convert/qwen4_exp.py)维护，编码和放置选择由
[official recipe](../../tools/convert/official_recipes.py)维护。

## 1. 架构与实例

源架构为 `Qwen4ExpForConditionalGeneration`，converter 也接受
`Qwen4ExpForCausalLM`，统一输出 `Qwen4ExpForCausalLM / qwen4_exp_text`。
Vision、MTP 是 target 为 `text` 的独立可选组件。架构接入使用原生 `qwen4_exp` 模块，
checkpoint 名称和 recipe 不承担执行注册身份。

| 参数 | Qwen3.8-Flash-Next |
|---|---|
| Text hidden / vocabulary | 2560 / 248320；embedding 与 head 不绑权重 |
| Main 层数 | 48；零基层号 `l%4==3` 为 QSA，共 12 层，其余 36 层为 GDN |
| HC streams / lowrank | 4 / 320；residual 宽度为 10240 |
| GDN Q/K heads / V heads / head dimension | 16 / 48 / 128；每组 Q/K 对应连续三个 V heads |
| GDN QKV / Z 宽度 | 10240 / 6144；4-tap、dilation=1 depthwise conv |
| QSA Q heads / KV heads / head dimension | 24 / 2 / 256 |
| QSA indexer heads / key heads / head dimension | 4 / 1 / 128 |
| QSA compression / budget | 四 token 一块；至多 512 个完整块，另加未完成块的至多三个可见 token |
| Routed experts / top-k / intermediate | 每层 512 / 10 / 640 |
| Shared expert | 每层一个，intermediate=640，带独立标量 gate |
| PLE 位置 | 仅零基第 1 层，在本层 attention HC 之前；源 `ple_layer_ids=[2]` 转为 `[1]` |
| PLE heads / 行宽 | 2-gram、3-gram 各 8 heads；16 行 × 160 维拼成 2560 维 embedding |
| PLE 表 | 128 个 `[2500012,160]` shards，共 320001536 行 |
| PLE conv | 10240 通道，4 taps、dilation=3，历史长度 9 |
| 原生 MTP | 一个 HC/QSA/MoE 层，无 PLE；共享 Main embedding/head |
| Vision | 27 blocks；hidden=1152，16 heads × 72，MLP intermediate=4304 |
| Vision patch / merger | patch `[3,2,16,16]`；2×2 merge，4608→4608→2560 |
| RoPE | theta=10000000；主 Q/K 旋转前 64 维，MRoPE sections `[11,11,10]`、interleaved |
| Context / norm epsilon / EOS | 262144 / `1e-6` / 248044 |

源层类型 `full_attention` 由 converter 规范化为 `indexed_attention`。
缺省 `norm_topk_prob=true` 写入产物，不依赖外部库将来的默认值。

Main 的数学数据流为：

```text
token embedding（在指定位置替换视觉特征）→ 展开为四条 HC residual streams
每层：
    零基第 1 层先将 PLE 加入 residual
    attention HC → GDN 或 QSA → 按 HC injection 加回 residual
    MoE HC       → routed + shared MoE → 按 HC injection 加回 residual
最后：final HC norm/mixer → 2560 维 sample hidden → LM head
```

HC 的四条 streams 是特征轴；GDN/QSA/MoE 的 block 输入输出仍为 2560 维。
Final HC mixer 后直接接 head。HC 的完整数学见
[gated_residual.h](../../include/ninfer/ops/gated_residual.h)。

GDN 接收 HC 输出，经 QKVZ 和 AB 投影、QKV 因果卷积/SiLU、Q/K L2 normalization 和
gated delta-rule，产生 48×128 维输出。输出端使用直接 gamma 的 head-wise RMSNorm
乘 `sigmoid(Z)`，再投影回 2560 维；这里的 sigmoid 与卷积后的 SiLU 分别保留。
Recurrent state 按 `S[value_dim,key_dim]` 存 FP32。完整层组合见执行文档，计算合同见
[GDN](../../include/ninfer/ops/gated_delta_net.h)、
[分段投影](../../include/ninfer/ops/linear_split.h)、
[投影与卷积融合](../../include/ninfer/ops/gdn_input_proj.h)、
[控制投影](../../include/ninfer/ops/gdn_control.h)和
[gated RMSNorm](../../include/ninfer/ops/rmsnorm.h)。

QSA 的四个 index heads 共同选出完整块，供所有 24 个 query heads 使用。主 K/V 保留完整
历史，4:1 压缩只作用于 index key。压缩顺序是 raw key 四列平均、BF16 舍入、offset norm、
块首坐标 RoPE；indexer 同样旋转前 64 维。Sparse GQA 输出乘逐元素 sigmoid gate 后再接
O projection。精确数学、选块、表示边界与 cache codec 见
[qsa.h](../../include/ninfer/ops/qsa.h)。

MoE router 从 BF16 投影结果计算 FP32 softmax，选 top-10，并重新归一化选中概率：

```text
expert_e(x) = W_down_e * (SiLU(W_gate_e*x) * (W_up_e*x))
routed(x)   = sum_selected_e(prob_e * expert_e(x))
shared(x)   = sigmoid(W_shared_gate*x) * shared_ffn(x)
moe(x)      = routed(x) + shared(x)
```

Shared FFN 使用同样的 SwiGLU 结构。MoE 的执行接口、路由和流式 expert 调度仍由
[执行需求](qwen4_exp-execution.md)细化。

MTP 从 Main 最后一层、final mixer 之前的完整 10240 维特征开始；后续 draft step 使用
前一步 MTP 的 pre-final-mixer 特征。其初始化数学为：

```text
e = W_fc_embedding * OffsetNorm_2560(token_embedding)
v = OffsetNorm_10240(multistream_hidden).reshape(4,2560)
h_init[s] = W_fc_hidden * v[s] + e
```

`pre_fc_hidden_norm` 对整个 10240 维归一化；`W_fc_hidden` 共享作用于四条 streams。
两个 FC 输入不同、权重独立。额外 QSA/MoE 层后的 final HC mixer 产生 2560 维 sample，
交给共享 head。BF16 边界、Main/MTP 双特征与恢复规则见执行文档。

Vision patch Conv3d 的 kernel=stride=`[2,16,16]`，已分块输入可按 C/T/H/W 次序展平为
1536 维。Block 使用非因果 attention、LayerNorm 和 `gelu_pytorch_tanh` MLP；merger 在
合并前做 LayerNorm，四个 patch 合成 4608 维，两次 FC 之间使用 `nn.GELU`。
本实例没有 deepstack。

## 2. 来源、精度与量化

| 用途 | 固定来源 |
|---|---|
| Config、资源、非主 routed-expert 参数及自量化源 | `/home/neroued/models/llm/qwen/Qwen3.8-Flash-Next/base-hf-bf16` |
| Main routed experts | `/home/neroued/models/llm/qwen/Qwen3.8-Flash-Next/nvidia-nvfp4-experts/experts.safetensors.index.json` |

NVIDIA 来源为 `nvidia/Qwen3.8-Flash-Next-NVFP4`，revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47`。专用 index 只覆盖本地的主 expert 分片；
转换不从上游完整 index 推断本地还具备 MTP/PLE 权重。该 recipe 导入主 expert 的原始
NVFP4 编码和校准参数，其余使用 BF16 源。具体来源与转换方法进入 `.conversion.json`。

| 部分 | 持久表示 | 计算许可/消费方式 |
|---|---|---|
| Main routed gate/up/down | NVIDIA NVFP4，G16/E4M3FN scale，每 expert 每 parent 一个 FP32 divisor | `AllowA4`；同一编码供 A16 或已资格化的 A4 路线 |
| Main GDN QKVZ/output、QSA Q/gate/output | BF16 源转 Q8 G32/FP16 scale | `A16Only` |
| HC、QSA K/V/indexer、PLE 投影、router/shared FFN | 原 BF16 words | `A16Only` |
| GDN AB、conv/norm，其他 norm/bias | 原 BF16 words | 投影 A16；其余按各自 Op 数学 |
| GDN A_log / dt_bias | 源 BF16 精确展开为 FP32，各 `[48]` | 保留原始含义，不预计算 exp |
| Token embedding / LM head | 两份独立 BF16 对象 | lookup / A16 head；MTP 共享对应对象 |
| PLE 表 | BF16 源转 `nf4_g16_e4m3fn` | lookup 解码为 BF16 |
| MTP routed experts | BF16 源自量化为 NVFP4 | `A16Only`，无 A4 校准辅助值 |
| MTP 其他独有参数、全部 Vision | 原 BF16 words | 投影 `A16Only` |
| PLE controls | 原始 35 个 INT64 | 无损资源，见 §4 |

Q8 使用现有 `grouped_absmax`，沿 K 每 32 个值分组。矩阵合并和行置换先确定最终行序，
再编码；所有 packing、scale swizzle、conv 转置均离线完成。Loader 校验并绑定已存表示。
KV cache、recurrent state 和 workspace 是执行资源，其 dtype 不由这个权重 recipe 决定。

### 2.1 Norm 参数语义

| 角色 | 持久参数 | affine 与归约轴 |
|---|---|---|
| HC、PLE grouped RMSNorm | BF16 offset `[10240]`，native `[2560,4]` | gain=`1+w`；四组各 2560 维、独立 affine |
| Main/MTP QSA Q/K、indexer Q/K | BF16 offset `[256]` / `[128]` | 每 head 归约，同角色的 heads 共用权重 |
| MTP pre-FC embedding / hidden | BF16 offset `[2560]` / `[10240]` | 分别对整个 2560 / 10240 维归约 |
| GDN output gated RMSNorm | BF16 gamma `[128]` | 直接 gain，无 unit offset；所有 V heads 共用 |
| Vision LayerNorm | BF16 weight/bias `[1152]` | 直接 affine；merger norm 在 merge 前执行 |

保留源 offset，不将 `1+w` 重新舍入成 BF16 gain，也不预先把 HC 的 `/4` 折入权重。
参数存储 dtype 与归约精度分别规定；实际激活舍入边界由 Op 合同维护。

### 2.2 NVIDIA NVFP4 导入

源投影的 `weight`、`weight_scale` 是 packed E2M1 codes 和自然行序 E4M3FN scales；
`weight_scale_2`、`input_scale` 是正有限 FP32 multipliers。导入按
[ModelOpt adapter](../../tools/convert/sources/modelopt_nvfp4.py)转换：

```text
source weight = E2M1(code) * E4M3FN(scale) * weight_scale_2
d_w = RN_FP32(1 / weight_scale_2)
d_a = RN_FP32(1 / input_scale)
stored weight = E2M1(code) * E4M3FN(scale) / d_w
```

Codes/scales words 保留，随后做目标 layout 的 packing/swizzle。Reciprocal 的 FP32
舍入是明确的元数据变换，不承诺转换前后所有解码值逐位相同。

Recipe 检查同 expert 的 gate/up 两个 multiplier words 均相等，且同层各 experts 的
gate/up 输入 multiplier 相等、down 输入 multiplier 相等；条件不满足则拒绝该来源。
Gate/up 可因此共用一个 parent 的 `d_w`，不同 expert 仍各自持有 divisor。
主模型 activation divisors 存为 FP32 `[48,2]`：C-order 元素 `2*l` 为 gate/up 输入，
`2*l+1` 为 down 输入。对应 Uses 共享该 scalar 引用，weight divisor 与 activation divisor
分别解释。

### 2.3 PLE 与 MTP 自量化

两者使用 [stored_scale_mse](tensor-formats.md#36-stored_scale_mse-conversion-method)。
PLE 每个原始 shard 独立计算完整 `amax` 与 divisor；MTP 每 expert 的合并 gate/up parent
及 down parent 分别计算。固定 codebook、完整 scale 候选域、FP64 误差和平局规则由该
通用合同维护。分块只改变工作集，不能改变分组或 divisor。

该方法不提供 activation calibration。NVIDIA 来源的 MTP/PLE 使用不同表示，本 recipe
从 BF16 自量化，MTP 不继承主 expert 的 A4 许可。主 expert 的来源校准也不能替代整个
Q8/NVFP4/NF4 组合的模型质量验证；这仍属于模型接入后的验收。

## 3. 物理 parent 与行顺序

矩阵按逻辑 `W[N,K]` 表达；N 是输出行，K 是输入特征，行区间均为半开区间。
相同输入、相同表示及兼容元数据的投影沿输出行合并，逻辑参数仍分别通过 Binding 引用。
原生 Tensor 第 0 维连续；`M=T*B`、`column=t+T*b`。普通 Linear 使用 `[K,M]→[N,M]`，
MTP fc_hidden 共享作用于四条 stream 时使用 `4M` 列。

| Parent | 数量 Main / MTP | `[N,K]` | 格式 | 行顺序 |
|---|---:|---|---|---|
| GDN QKVZ | 36 / 0 | `[16384,2560]` | Q8 | Q `[0,2048)`；K `[2048,4096)`；V `[4096,10240)`；Z `[10240,16384)` |
| GDN AB | 36 / 0 | `[96,2560]` | BF16 | a `[0,48)`；b `[48,96)` |
| GDN output | 36 / 0 | `[2560,6144]` | Q8 | 原 out_proj 行序 |
| Main QSA Q/gate | 12 / 0 | `[12288,2560]` | Q8 | Q `[0,6144)`；gate `[6144,12288)` |
| Main QSA KV/index | 12 / 0 | `[1664,2560]` | BF16 | K `[0,512)`；V `[512,1024)`；index-Q `[1024,1536)`；index-K `[1536,1664)` |
| Main QSA output | 12 / 0 | `[2560,6144]` | Q8 | 原 o_proj 行序 |
| MTP QSA input | 0 / 1 | `[13952,2560]` | BF16 | Q `[0,6144)`；K `[6144,6656)`；gate `[6656,12800)`；V `[12800,13312)`；index-Q `[13312,13824)`；index-K `[13824,13952)` |
| MTP QSA output | 0 / 1 | `[2560,6144]` | BF16 | 原 o_proj 行序 |
| Block HC down/inject | 96 / 2 | `[324,10240]` | BF16 | down `[0,320)`；injection `[320,324)` |
| Final HC down | 1 / 1 | `[320,10240]` | BF16 | 全部 down |
| HC up | 97 / 3 | `[10240,320]` | BF16 | stream-major，`row=s*2560+d` |
| PLE KV | 1 / 0 | `[12800,2560]` | BF16 | 四流 key `[0,10240)`；共享 value `[10240,12800)` |
| MoE input | 48 / 1 | `[1793,2560]` | BF16 | router `[0,512)`；shared score `[512,513)`；shared gate `[513,1153)`；shared up `[1153,1793)` |
| Shared expert down | 48 / 1 | `[2560,640]` | BF16 | 原 down_proj 行序 |
| Routed expert gate/up | 24576 / 512 | `[1280,2560]` | NVFP4 | 每 expert 独立；gate `[0,640)`；up `[640,1280)` |
| Routed expert down | 24576 / 512 | `[2560,640]` | NVFP4 | 每 expert 独立 |
| MTP fc_embedding | 0 / 1 | `[2560,2560]` | BF16 | 独立矩阵，M 列 |
| MTP fc_hidden | 0 / 1 | `[2560,2560]` | BF16 | 独立矩阵，共享作用于四流，4M 列 |
| LM head | 一份，MTP 共享 | `[248320,2560]` | BF16 | vocabulary 顺序 |

Main QSA 两个 input parents 精度不同；MTP 同类权重全部 BF16，合为一个 parent。
BF16 的 N=324、N=1793 是紧凑实际行数。MoE 的控制段 `[513,2560]` 与 shared gate/up
段 `[1280,2560]` 可作零拷贝 row views，允许路由、传输和 shared 计算分别调度。
持久合并不强制投影在同一次调用执行。

### 3.1 Q/gate 离线置换

源 QSA q_proj 每个 head 的 256 行 Q 后紧跟 256 行 gate。Main 和 MTP 均先解交错，再按
上表放入 parent：

```text
Q[h*256+d,:]    = source_q_proj[h*512+d,:]
Gate[h*256+d,:] = source_q_proj[h*512+256+d,:]   // h=0..23, d=0..255
```

行内 K 顺序不变，Q8 groups/scales 随行。Indexer 源投影已是 512 行 Q、128 行 K。
Packed activation 的子 views 必须保留真实 parent strides；具体 Op 输出可以采用独立连续
分段，持久权重仍只保存上述 parent。

### 3.2 Vision 与非矩阵参数

| 参数 | 数量 | 持久 shape / 顺序 |
|---|---:|---|
| Vision patch projection | 1 | `[1152,1536]`；源 `[1152,3,2,16,16]` 的 C/T/H/W 展平 |
| Vision QKV | 27 | `[3456,1152]`；Q/K/V 各连续 1152 行，bias 同序拼接 |
| Vision attention output | 27 | `[1152,1152]` |
| Vision FC1 / FC2 | 各 27 | `[4304,1152]` / `[1152,4304]` |
| Merger FC1 / FC2 | 各 1 | `[4608,4608]` / `[2560,4608]` |
| Vision position embedding | 1 | `[2304,1152]` |
| Token embedding | 1 | `[248320,2560]`，MTP 共享；与 LM head 是不同对象 |
| PLE 表 | 128 | 每 shard `[2500012,160]`，NF4 |
| GDN conv / PLE conv | 36 / 1 | 各 `[4,10240]`，channel 连续 |

Vision 全部为 BF16，投影 bias、norm weight/bias 保留各自逻辑角色与源 words。
Conv 与 norm 不因 shape 为二维就变成 Linear。GDN/PLE 卷积的确定映射为：

```text
source [C,1,4] → artifact/Binding [4,C] → native Tensor [C,4], C=10240
stored[j*C+c] = source[c*4+j]
```

这是 BF16 word 置换，tap 时间顺序不变；每个对象仍为 81920 B。Binder 上传原字节并建立
native view。PLE channel 连续布局已做 Op 对照，未显示与源布局有一致优劣，继续使用该
转换布局；它的 dilation=3 只改变时间索引。

## 4. Binding、expert bank 与 PLE 资源

逻辑目录保留独立的 Q/K/V、HC down/up/inject、routed gate/up/down 等角色；完整参数名和
Use 输入名以 adapter 为准。同 parent 的完整行 `[a,b)` 对应 Binding 逻辑元素区间
`[a*K,b*K)`。区间不包含 scale bytes，上传和绑定不重新排列 Q/gate 或卷积。

每 expert 有两个普通 rank-2 NVFP4 objects，Main 共 49152 个、MTP 共 1024 个。
逻辑 bank 为 gate/up 各 `[512,640,2560]`、down `[512,2560,640]`；Binding parts 按
expert ID 引用各自 parent 的 gate/up 两半或完整 down。每个 part 必须保留 parent geometry、
scale swizzle 与独立 `d_w`。Bank 是有序对象集合，不增加新的数值格式。

NVFP4 使用 `block_scale_k16_m128x4_v1`：每 expert 的 gate/up payload 为 1843204 B，
down 为 921604 B，共 2764808 B。对象起点按 256 B 对齐，相邻起点跨度合计 2765312 B；
activation divisor 经 Use 引用辅助对象。GPU expert cache 保存同一编码原字节。

PLE table Binding 按 shard ID 引用 128 个完整 NF4 parents，保留原 shard 边界和各自
divisor。使用 `row_paged_g16_4k_v1`；本实例每行 90 B，每 4 KiB 页 45 行，每 shard
payload 为 227557380 B。全局行号 g 的 shard 是 `g/2500012`，局部行为 `g%2500012`；
hash head 与 shard 不一一对应。实际文件范围必须经过 V3 逻辑区间映射。
持久编码的完整定义见[布局合同](storage-layouts.md#6-row_paged_g16_4k_v1)，
lookup 的 BF16 精确舍入与行解码见 [PLE I/O](qwen4_exp-ple-io.md)。

`text.resources.ple_hash_controls` 引用一个 280 B 的 `raw_bytes_v1` Resource：

| 字节区间 | 原始控制值 |
|---|---|
| `[0,24)` | 三个 little-endian signed INT64 layer multipliers |
| `[24,152)` | 十六个 little-endian signed INT64 head vocab sizes |
| `[152,280)` | 十六个 little-endian signed INT64 head offsets |

资源复制源 buffers，不重新生成 seed/prime，也不经浮点数往返。它由模型绑定，
hash 的 64-bit word、EOS 与历史规则由 PLE I/O 合同维护。

本 recipe 的物理放置顺序是资源/辅助值、Main 固定权重、Vision、MTP、Main experts、
PLE shards。Main experts 按 layer→expert→gate_up/down 相邻放置，PLE 按 shard ID 放置，
对象起点 4096 B 对齐。MTP experts 位于自己的常驻组件区域。共享 embedding/head 不复制，
A16/A4 共用同一份 NVFP4 编码。对象允许跨 V3 续卷，文件名不承担参数身份。

## 5. 驻留与容量

以下是原生执行的既定驻留方案，尚未由 Qwen4Exp loader/Program 实现：

| 数据 | 驻留与所有权 |
|---|---|
| Main 非 routed-expert、非 PLE 表权重 | GPU 常驻 |
| 启用的 Vision、MTP（含其 experts） | GPU 常驻；共享 embedding/head 只一份，未启用组件不 materialize |
| Main routed experts | 完整 RAM 编码 backing；Program 使用有界 GPU expert cache/传输槽 |
| PLE 表 | V3 磁盘原位 backing；Program 按需读取、有界缓存与 E staging |
| PLE controls、128 个 divisors、行几何及 LUT | Model 只读数据；LUT 512 KiB |

Model 持有不可变表示、参数和长期只读文件 backing；Program 持有可变 cache slots、
staging、请求状态及 Graph。完整 Host expert backing 包括已缓存到 GPU 的 expert，
因此 eviction 无需写回，RAM 预算仍按完整 bank 计算。启动绑定不扫描或整体加载 PLE 表。

容量按当前 recipe 的编码公式计算，包含 divisors、NF4 页内 padding、280 B controls
和 384 B activation-divisor 对象；不含对象间对齐、目录和文件 framing：

| 部分 | Payload bytes | GiB |
|---|---:|---:|
| Main 非 routed-expert、非 PLE 表 | 7418143896 | 6.908685 |
| Main routed experts | 67947921408 | 63.281433 |
| PLE NF4 表 | 29127344640 | 27.126954 |
| Vision | 897862112 | 0.836199 |
| MTP experts | 1415581696 | 1.318363 |
| MTP 其他独有权重 | 181136896 | 0.168697 |
| 全部组件 | **106987990648** | **99.640331** |

Text-only payload 约 97.317072 GiB；启用 Vision+MTP 时固定 GPU 权重 payload 约
9.231944 GiB，再加 Main expert cache。本机全组件产物入口为
`out/qwen3_8_flash_next.ninfer`，另有同名前缀的 `.part-0001`、`.part-0002`、`.part-0003`，
四文件合计 **107025649668 bytes**。文件集合总长与上述 payload 分开计算，PLE
磁盘化只改变驻留成本。RAM 还须计入 reader/cache metadata、pinned staging、进程和系统
占用；GPU 还须计入页组、固定状态、候选记录、workspace、Graph 和传输槽。
执行资源的具体容量与同时存活关系见[执行文档](qwen4_exp-execution.md)和
[PLE I/O](qwen4_exp-ple-io.md)，不以权重 payload 代替启动时可分配余量。

## 6. 数学来源

数学结构以固定版本的
[Transformers 模型](https://github.com/huggingface/transformers/blob/8073e6dcaea9aa4b42acf2d92ec72148786bbaeb/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)与
[配置](https://github.com/huggingface/transformers/blob/8073e6dcaea9aa4b42acf2d92ec72148786bbaeb/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py)为依据；
MTP 特征关系参照固定版本的
[vLLM MTP](https://github.com/vllm-project/vllm/blob/6fddc9b21e763bfc947cc01d3ce815179aac1dc6/vllm/models/qwen4_exp/nvidia/mtp.py)。
[NVIDIA 来源说明](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/blob/fc694b54fb0174e0913e6adf86691ef85a4ead47/README.md)
记录其 NVFP4 主 experts 与其他组件的表示。上游 TP、padding、私有 casts 和 kernel packing
不自动成为 NInfer 的数学或持久合同；正式数值边界以对应生产 Op 头文件为准。
