# Qwen3.8-Flash-Next：权重转换实现方案

更新日期：2026-10-08。

在现有 converter 中增加 `qwen3_8_flash_next` recipe，输出 V3 `.ninfer`。
精度、合并矩阵和字节布局以[模型权重方案](qwen3_8-flash-next-model.md)为准。

## 1. 代码改动

| 位置 | 改动 |
|---|---|
| `tools/convert/qwen4_exp.py`，新增 | 解析 config，建立逻辑参数、源映射、组件和合并分组 |
| `tools/convert/__main__.py` | 按源架构选择 Qwen3.5 或 Qwen4Exp adapter |
| `tools/convert/official_recipes.py` | 增加 `qwen3_8_flash_next` 的类型、来源与存储分配 |
| `tools/convert/sources/modelopt_nvfp4.py`，新增 | 读取 NVIDIA packed weights/scales，转换 multiplier 为 divisor |
| `tools/convert/recipe.py`、`model.py`、`methods.py` | 支持逻辑参数的分段 parent 规划、共享 auxiliary 引用 |
| `tools/convert/quantization/` | 实现 NF4 G16 与 MTP NVFP4 的 stored-scale MSE 编码 |
| `tools/artifact/formats.py`、`layouts.py`、`codecs/` | 增加 NF4 格式、分页 geometry 和 codec |
| `tools/artifact/tensor_output.py` | 流式组装 NF4 页面、padding 和 divisor |
| `tools/convert/pipeline.py` | 按指定顺序放置资源、辅助值和权重对象 |
| `src/core/weight*`、`src/artifact/` | 注册 NF4 类型/layout，增加分页行地址与 divisor 读取 |

继续复用 `SafetensorsSource`、`LogicalSource`、`PreparedMethod` 和 `ArtifactWriter`。
Qwen4Exp 的架构映射单独实现；writer 的分卷、临时文件和最终发布机制沿用现有代码。

## 2. 输入与配置

| 来源 | 本地路径 |
|---|---|
| BF16 | `/home/neroued/models/llm/qwen/Qwen3.8-Flash-Next/base-hf-bf16` |
| NVIDIA experts | `/home/neroued/models/llm/qwen/Qwen3.8-Flash-Next/nvidia-nvfp4-experts/experts.safetensors.index.json` |

主 experts 使用 NVIDIA 专用 index；其余参数从 BF16 读取，MTP experts 也从 BF16 自量化。
96 个 BF16 主 expert banks 只核对几何。全部组件使用其余 1562 个 BF16 来源条目，以及
NVIDIA index 中的 294912 个投影字段。来源 revision 沿用总览 §2。

将源架构映射为 `Qwen4ExpForCausalLM`。解析 Text/HC/GDN/QSA/MoE/PLE 的维度和固定规则，
将源 `full_attention` 规范化为 `indexed_attention`、PLE 层号转为零基，写出 resolved defaults。
MTP 与 Vision 使用各自组件配置，target=text。具体数学配置见总览 §3。

组件选择支持 text、text+vision、text+mtp 和全部组件。Tokenizer、template、generation
config 及选中的 processor 资源直接使用本地源文件。首版不增加 DFlash 或 indexed proposal。

## 3. 转换顺序

1. 读取 config、资源和 tensor headers，建立源映射。
2. 应用 recipe，确定合并 parent、Binding、Use、辅助对象和放置顺序。
3. 写出直接参数与 Q8 投影，导入 NVIDIA experts。
4. 编码 MTP experts；逐 shard 扫描并编码 PLE。
5. 使用现有 writer 完成文件集合，沿用 `.conversion.json` 记录来源和编码方法。

先实现 format/layout 和模型映射，再接入导入与自量化方法，最后运行完整转换。

## 4. 逻辑参数目录与源映射

`c` 为 text 或 mtp，`P=c/layers/{l}`。对应源前缀 `S` 为
`model.language_model.layers.{l}` 或 `mtp.layers.0`。矩阵 shape 和合并行区间见总览 §5。

| 逻辑参数 | 源映射 |
|---|---|
| `text/token_embedding`、`text/output_head` | `model.language_model.embed_tokens.weight`、`lm_head.weight` |
| `mtp/token_embedding`、`mtp/output_head` | 分别共享对应 Text Binding |
| `P/attention_hc/{norm,down,up,inject}` | `S.attn_hyper_connection` 的 `hc_norm`、`input_mix_weight_down/up`、`block_inject_weight` |
| `P/moe_hc/{norm,down,up,inject}` | 同上，源模块换为 `S.mlp_hyper_connection` |
| `c/final_hc/{norm,down,up}` | 对应 `hyper_connection_mixer`，没有 inject |
| `P/gdn/{query,key,value,z}` | `linear_attn.in_proj_qkv.weight` 三段和 `in_proj_z.weight` |
| `P/gdn/{a_projection,b_projection,output}` | `linear_attn.in_proj_a/b.weight`、`out_proj.weight` |
| `P/gdn/{convolution,a_log,dt_bias,norm}` | `linear_attn.conv1d.weight`、`A_log`、`dt_bias`、`norm.weight` |
| `P/attention/{query,output_gate,key,value,output}` | `self_attn.q_proj.weight` 去交错，以及 `k_proj/v_proj/o_proj.weight` |
| `P/attention/{index_query,index_key}` | `self_attn.indexer.index_qk_proj.weight` 的前 512 / 后 128 行 |
| `P/attention/{query_norm,key_norm,index_query_norm,index_key_norm}` | `self_attn` 的 `q_norm/k_norm`、`indexer.q_layernorm/k_layernorm` |
| `P/moe/{router,shared_score}` | `mlp.gate.weight`、`mlp.shared_expert_gate.weight` |
| `P/moe/shared/{gate,up,down}` | `mlp.shared_expert` 的对应投影 |
| `P/moe/experts/{gate,up,down}` | 三个逻辑 bank；主模型逐 expert 导入，MTP 从 BF16 banks 切片 |
| `P/ple/{key,value,norm_key,norm_query,norm_conv,convolution,table}` | `ple` 的投影/norm、`conv1d.weight` 和 128 个 embedding shards；仅 Text 第 1 层 |
| `mtp/pre_fc_embedding_norm`、`mtp/pre_fc_hidden_norm` | `mtp.pre_fc_norm_embedding.weight`、`mtp.pre_fc_norm_hidden.weight` |
| `mtp/fc_embedding`、`mtp/fc_hidden` | 两个同名独立源矩阵 |

表内 GDN/QSA/MoE/PLE 字段均相对 `S`。Vision 沿用现有 `vision/patch_embedding`、
`vision/position_embedding`、`vision/layers/{i}/attention/*`、`mlp/*`、norm/bias 和
`vision/merger/*` 命名，对应 `model.visual.*`；全部 BF16。

Use 输入名统一如下，`G` 为对应 HC 参数前缀；norm、conv、embedding/table lookup 不建立 Linear Use：

| 投影 | input |
|---|---|
| HC 组 `G` 的 down/inject、up | `G/normalized_streams`、`G/lowrank_activation` |
| GDN 与 QSA 输入投影 | `P/attention_input` |
| GDN / QSA output | `P/gdn/gated_output` / `P/attention/gated_output` |
| MoE router、shared_score、shared/routed gate/up | `P/moe_input` |
| routed / shared down | `P/moe/experts/product` / `P/moe/shared/product` |
| PLE key/value | `P/ple/embedding` |
| MTP 两个 FC | `mtp/normalized_embedding`、`mtp/normalized_hidden_streams` |
| Text / MTP head | `text/final_hidden` / `mtp/final_hidden` |

Vision 的 Use 沿用现有对应投影的输入命名。所有 Use 为 A16Only，主 routed expert banks
改为 AllowA4 并绑定对应 activation divisor。输入名表示数学用途，不增加公共中间 buffer。

## 5. 编码实现

### 5.1 直接参数与 Q8

BF16 参数按源 words 复制、切片和拼接。QSA Q/gate 去交错后按目标行序编码；Q8 复用
`grouped_absmax` 和现有 `row_split_k128_v1`。

GDN 与 PLE conv 都离线转换：

```text
source [C,1,4] → artifact [4,C] → native Tensor [C,4]
stored[j*C+c] = source[c*4+j], C=10240
```

GDN A_log/dt_bias 从 BF16 精确展开为 FP32。Norm 保留原 BF16 offset/gamma。
Vision patch 按源 C/T/H/W 展平；QKV weights/bias 保持三段原序。

### 5.2 NVIDIA NVFP4

每个投影读取 `weight`（U8 packed codes）、`weight_scale`（自然行序 E4M3FN）、
`weight_scale_2` 和 `input_scale`（FP32 multipliers）。转换为：

```text
d_w = RN_FP32(1 / weight_scale_2)
d_a = RN_FP32(1 / input_scale)
```

确认同 expert gate/up 的 d_w、d_a 相同，并确认 input scale 在同层各 experts 间一致。
Gate/up 行拼接后调用现有 NVFP4 packing；保留 codes/scales words，离线执行 scale swizzle。
按完整 128 行 tiles 写入，每个 parent 写一个 d_w。

### 5.3 PLE NF4

注册 `nf4_g16_e4m3fn / row_paged_g16_4k_v1`。码本、合法 words 与字节公式见总览 §7。
Python/C++ 增加独立的分页行 geometry：row offset、code/scale spans、页数和 divisor offset。
`default_layout`、`TensorOutput` 和 C++ 格式解析显式选择该路线。

逐 shard 两遍处理：先有界扫描完整 amax，取得 `d=RN_FP32(448/amax)`；再分块编码 G16。
每组枚举 126 个正有限 E4M3FN scales，选最近 NF4 code，按最终表示值的平方误差取最优。
使用固定码本与存储后的 d；全零对象 d=1，全零组 code=7、scale=0。平局规则沿用总览 §4.4。

生产编码在有界 batch 上遍历候选，保留每组最佳结果，避免建立完整候选距离张量。
误差比较采用 FP64 基线；可比较 `sum((w*d-code*s)^2)`，省去公共的 `1/d²` 因子。

每行写 80 B codes、10 B scales，每页组装 45 行并填充页尾。跨 chunk 的未满页接续，
最后补齐末页并写一次 divisor。Chunk 大小只影响工作集，不改变分组、divisor 或分页。

### 5.4 MTP NVFP4

逐 expert 切出 BF16 gate/up 和 down。合并 gate/up 共用完整矩阵 amax，down 独立计算；
`d=RN_FP32(6*448/amax)`。使用 E2M1 码本和 G16 stored-scale 搜索，再交给现有 NVFP4 codec。
每次处理一个 expert 的当前投影，Use 保持 A16Only。

## 6. Parent、Binding 与写出

扩展 prepare 的 emit 层，使一个 parent 可由多个逻辑参数区间组成。每个输入片段包含
`parameter`、逻辑 `[begin,end)` 和有界 `LogicalSource`；最终仍生成现有 WeightJob。
这用于 expert banks 和 PLE table，普通矩阵继续使用现有 grouping。

每个 expert 写两个 parent：gate/up `[1280,2560]` 和 down `[2560,640]`。三个 bank 的
Binding 按 expert ID 排列 parts，分别引用 gate/up 两半和完整 down；每个 part 保留自己的
parent geometry/divisor。MTP BF16 source 映射为：

```text
gate[e,i,k] = source_gate_up[e,i,k]
up[e,i,k]   = source_gate_up[e,640+i,k]
down[e,h,i] = source_down[e,h,i]
```

主模型 activation divisors 存一个 FP32 `[48,2]`：gate/up 引用元素 `2*l`，down 引用
`2*l+1`。Recipe 增加共享 auxiliary Part 引用，替代逐 Use 重复生成 scalar 对象。

PLE table Binding 按 shard ID 引用 128 个完整 NF4 parents，各自保留 divisor。
三个 INT64 buffers 按总览约定合成 280 B `ple_hash_controls`，挂在 `text.resources`。

Pipeline 的对象顺序为：资源/辅助值 → 主常驻权重 → Vision/MTP → 主 experts → PLE。
主 experts 按 layer/expert/gate_up,down 排列，PLE 起点按 4096 B 对齐。Writer 根据完整
对象计划计算 offsets 和分卷；各 job 有界读写，不整体保留大表或 expert bank。

实现 NF4 后同步更新通用数值格式与布局文档；模型总览继续维护实例 recipe 和容量。
PLE conv 当前使用 channel 连续，布局性能留待其 Op 实现时核实。
