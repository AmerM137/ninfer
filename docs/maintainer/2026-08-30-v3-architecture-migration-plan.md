# NInfer v2 到 v3 整体架构迁移计划

> 状态：临时执行计划，实施尚未开始。
>
> 本文不加入文档索引，不是已经生效的产品、容器或运行时权威。它把
> [`2026-08-29-architecture-config-weight-realization-discussion.md`](2026-08-29-architecture-config-weight-realization-discussion.md)
> 中已经收敛的方向组织成一条可执行、可验收的整体迁移路径。实施完成后，应将稳定契约回写到
> 对应 active maintainer 文档、`AGENTS.md`、公共文档和 model cards，并删除本文与前述讨论笔记。

## 1. 计划目的

本次工作不是在现有 target/`WeightsProfile` 体系中增加 v3 container，而是一次跨 artifact、
converter、loader、Architecture、Program、Engine、工具和发布流程的产品契约切换。最终交付包括：

1. 一个能够表达显式 Architecture Config、分层 Weight Realization、Primary model、内嵌
   speculative modules、共享对象和权重分片的 `.ninfer` v3 容器；
2. 一个以 Architecture Definition 和 NInfer-owned Config 为数学权威、在加载期解析全部
   realization、在执行期保持稳定直接路径的 Engine；
3. 一个将 source recipe、realization assignment 和 artifact serialization 分离的 v3 权重生成
   工作流；
4. 五个当前已发布 artifact 的 source-to-v3 生成结果；
5. 一个只面向五个精确已发布 v2 artifact 的离线 v2-to-v3 升级工具及仓库内配套资源，使已有
   用户无需重新下载或重新量化大体积权重；
6. 与新所有权一致的代码、目录、类型、诊断、测试、benchmark、文档和发布命名；
7. 对旧 v2 Engine 路径、checkpoint-named execution packages、`weights_id`、完整
   `WeightsProfile` 和硬编码 speculative backend 表达的删除。

本文规定工作依赖、阶段边界、验证门和完成定义。它有意不提前规定 v3 JSON spelling、prefix、
shard filename、最终 C++ 类型名、最终目录名或逐个 physical-form ABI；这些属于后续设计阶段的
正式产物。

## 2. 保持不变的产品边界

除本计划明确改变的 artifact、architecture 和 weight realization 合同外，当前产品模型继续成立：

- 一张 RTX 5090、一份常驻模型实例、启动时固定的一至八个 active requests；
- 有界 FIFO ingress、无 active request preemption、每轮形成一个 compact decode batch；
- Text、Vision、speculative decode、prefix reuse、CLI、OpenAI/Anthropic serving 和 measurement
  继续通过同一个公共 `.ninfer` Engine 路径；
- 运行时仍是直接 C++/CUDA architecture implementation，不引入通用 compute-graph IR、graph
  optimizer、未知节点解释或任意模型插件；
- artifact、converter 和发布流程仍由项目单一 owner 管理并视为可信；
- converter 完成量化、fusion/split、packing、layout 转换和 persistent auxiliary values 生成；
  loader 不修改权重；
- 实际可执行 config、shape、format、layout 和 form 继续是有限能力集合；不支持的组合在加载和
  规划期拒绝；
- OpenAI 和 Anthropic 协议行为继续是外部合同。公共 model string 可以与内部执行身份解耦，
  但不得在没有相应 schema/documentation 变更时改变协议行为。

本次迁移不顺带引入多 GPU、跨 Engine context store、动态模型下载、运行时外挂 speculative
weights、逐请求 module 选择、同时执行多个 speculative modules、恶意 artifact 沙箱或新的硬件
平台。

## 3. 终态不变量

最终实现必须同时满足以下不变量。

### 3.1 Architecture 与 Config

1. Architecture Definition 是模型数学、拓扑规则、状态和调度的代码级权威。
2. Qwen3.5、Qwen3.6 和 Qwen3.8 是 checkpoint/release 系列，不是 execution architecture。
3. Dense 与 MoE 是不同 Architecture Definition；相同 architecture 下的不同尺寸由 Config
   表达。
4. `.ninfer` 显式携带 architecture identifier 和完整 NInfer-owned normalized config。
5. loader 不根据 filename、tensor name、shape、object count 或代表性 descriptor 猜测 config。
6. release/checkpoint identity 若继续存在，只属于用户显示、sampling provenance、model card 或
   其他非执行语义。

### 3.2 Weight Realization

1. Architecture 拥有分层 semantic regions、semantic ABI 和允许的 physical-form ABI。
2. artifact 为 semantic regions 显式选择已注册 physical forms 并绑定 physical objects。
3. object descriptors 只表达实际 shape、numeric format、storage layout 和 payload location；
   artifact 不直接选择 kernel。
4. resolver 在加载期根据 config、form、descriptor、hardware 和所有可达执行模式解析 immutable
   realizations。
5. 不同层和不同闭合区域可以自由组合已有 realizations；新增组合不要求新增完整 Program、target
   package 或 `WeightsProfile`。
6. Program hot path 不解析 artifact strings、object descriptors 或 capability rules。
7. `weights_id` 不参与 v3 execution identity，也不能以手写 profile label、canonical signature
   或其他名称重新出现为 binder/workspace/Program 的权威。

### 3.3 Primary model 与 Speculative Modules

1. 一个 artifact 包含一个必需 primary model 和零至多个内嵌 speculative module instances。
2. 每个 module 拥有自己的 architecture、config、attachment contract、semantic bindings 和
   private objects。
3. module 只能使用 primary architecture 注册的 feature taps、verification/state ABI 和共享对象。
4. 一个 Engine 启动时选择 `none` 或至多一个 artifact-local module instance；选择在生命周期内
   不变。
5. 只解析、规划和 materialize primary model 与被选 module 的对象依赖闭包。
6. MTP、DFlash 及未来不同 proposal 数学不是 weight profiles；同一算法的不同物理格式才是
   realization 差异。

### 3.4 v3 Container 与分片

1. v3 是一个逻辑 artifact；其物理表示可以包含一个或多个文件。
2. 分片是 v3 首次发布时已经实现并验证的容器能力，不留作依赖未知 extension 的占位字段。
3. object、shared reference、view 和 module binding 在跨 shard 时保持同一语义和完整性规则。
4. reader/materializer 不假设所有 payload 都属于一个 mmap、一个 file descriptor 或一个绝对
   file-offset 空间。
5. schema 继续是受版本控制的闭合合同；面向未来需求不等于接受任意 metadata 或未知 physical
   forms。

### 3.5 v2 兼容边界

1. 最终 Engine 和普通 v3 tooling 不读取 v2。
2. v2 reader 仅作为离线升级工具的输入实现存在，不链接到 Engine，也不成为通用 legacy lane。
3. 升级器只接受本项目五个精确已发布 v2 artifact，不推断或修复未知 v2 文件。
4. v2 缺失的 config、semantic bindings、module inventory 和 form facts 由仓库内、与精确发布物
   配对的升级资源提供。
5. v2-to-v3 不重新量化、反量化、融合、拆分复制、transpose、swizzle 或 repack 已发布权重。
6. 升级失败不得损坏用户已有 v2 文件；输出的磁盘需求、进度和验证结果必须清晰可见。

## 4. 当前迁移基线

### 4.1 已发布 artifact

当前产品有五个 v2 artifact。精确 SHA、source revision、converter revision、inventory 和 minimum
runtime revision 由各 model-card 目录内的 `artifact-manifest.json` 维护；本计划不复制一份并行
hash 权威。

| 已发布 identity | 文件 | 字节数 | 当前执行 package/profile | v3 architecture witness |
|---|---|---:|---|---|
| `qwen3.6-27b/groupwise-int` | `qwen3_6_27b.ninfer` | 17,495,365,888 | 27B Dense groupwise profile | Dense split/groupwise forms |
| `qwen3.6-27b/nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | 18,324,064,000 | 27B Dense NVFP4 profile | Dense mixed fused forms |
| `qwen3.8-27b/groupwise-int` | `qwen3_8_27b.ninfer` | 18,210,531,328 | 27B Dense groupwise profile | 同一 Dense architecture 的另一 release |
| `qwen3.8-27b/nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | 21,492,695,040 | 27B Dense FP8/NVFP4 profile | Dense mixed FP8/NVFP4/BF16 forms |
| `qwen3.6-35b-a3b/groupwise-int` | `qwen3_6_35b_a3b.ninfer` | 22,783,246,080 | MoE groupwise profile | MoE、MTP、DFlash、shared objects |

### 4.2 当前耦合点

当前实现中需要被整体替换的主要耦合包括：

- v2 root 严格只有 `identity` 和 `objects`；
- `(model_id, weights_id)` 决定 closed registry 和 target-private `WeightsProfile`；
- target `config.h` 编译期拥有 layer count、dimensions、topology 和数学常量；
- profile 同时决定完整 inventory、numeric formats、split/fused parents、binder 分支、workspace
  capacity 和 execution leaf 路径；
- converter 以 checkpoint/profile-specific inventory 和 recipe 生成一个单文件 artifact，HF config
  只被验证并写入外置 report；
- `Reader`、`Materializer` 和 `EngineOptions::artifact_path` 按一个 `.ninfer` 文件工作；
- `LoadSummary`、context-cost、CLI、serving logs、benchmark 和文档传播 `target/model_id/weights_id`；
- MTP/DFlash 通过硬编码 `SpeculativeBackend`、`optional<MtpLayer>`、
  `optional<DFlashPayload>` 及对应 workspace/graph 分支表达。

### 4.3 必须保留的现有能力

迁移不能以架构整洁为由降低以下支持：

| 能力 | 四个 Dense artifacts | 35B-A3B artifact |
|---|---:|---:|
| Text generation | 必须保留 | 必须保留 |
| image/video Vision | 必须保留 | 必须保留 |
| ordinary decode | 必须保留 | 必须保留 |
| MTP | 必须保留 | 必须保留 |
| DFlash | 不适用 | 必须保留 |
| prefix/continuation reuse | 必须保留 | 必须保留 |
| compact batched decode | 必须保留 | 必须保留 |
| CUDA Graph capture/replay | 必须保留 | 必须保留 |
| CausalScoring/perplexity | 必须保留 | 必须保留 |
| CLI 与 OpenAI/Anthropic serving | 必须保留 | 必须保留 |
| benchmark/diagnostics/measurement | 必须保留 | 必须保留 |

## 5. 总体执行策略

本次迁移采用“终态合同先行、基础能力汇合、完整纵向切片、单次产品切换”的顺序：

```text
Phase 0  迁移宪章、验收矩阵和必要基线
    |
Phase 1  Architecture/Config/Realization/Module 领域合同
    |
    +---------------------+----------------------+----------------------+
    |                     |                      |                      |
Phase 2A              Phase 2B               Phase 2C                  |
v3 container/shards   resolver/program input  v2 release catalog       |
    |                     |                      |                      |
    +---------------------+----------------------+                      |
                          |                                             |
Phase 3  v3 converter、artifact 生成、loader/resolver 汇合              |
    |                                                                   |
Phase 4  Dense 完整纵向切片                                              |
    |                                                                   |
Phase 5  MoE + speculative modules 完整纵向切片                         |
    |                                                                   |
Phase 6  Engine/registry/product identity/命名全面切换                   |
    |                                                                   |
Phase 7  五个已发布 v2 artifact 的离线升级                              |
    |                                                                   |
Phase 8  删除旧路径、最终资格验证、active documentation                 |
    |                                                                   |
Phase 9  staged publication 与用户切换                                  |
```

Phase 2A、2B 和 2C 可以在 Phase 1 领域合同稳定后并行推进，但 Phase 3 不得在三者关键输出未汇合
时冻结最终 v3 artifact。其余阶段按依赖顺序完成。

允许新旧内部实现只在未发布开发过程中短暂共存，以保持代表性测试可运行；这种共存不是公共兼容
合同。每个纵向切片结束时必须删除其已经被替代的旧 owner，最终发布不得携带两条可选择 Engine
路径。

## 6. Phase 0：迁移宪章、验收矩阵和基线

### 6.1 工作

1. 确认第 2 节产品边界和第 3 节终态不变量。
2. 为五个 artifact 建立实际资格矩阵，覆盖第 4.3 节所有适用能力。
3. 固化五个本地发布文件与 model-card manifests 的对应关系。
4. 记录只用于识别回归的代表性基线：
   - load bytes、upload bytes、peak staging、HBM residency 和 startup time；
   - 一个 Dense groupwise、一个 Dense FP8/NVFP4 和 35B MoE 的代表性 prefill/decode；
   - MTP 与 DFlash 的代表性 acceptance/throughput；
   - CUDA Graph、prefix reuse 和真实 CLI/serving 小型集成结果。
5. 决定公共 model string、checkpoint provenance、sampling defaults 与 execution identity 的边界。
6. 决定 v3 artifact entry point 对 CLI/Engine 的产品含义，但不在此阶段先定具体 filename spelling。
7. 明确升级器的用户合同：支持的输入、输出位置、磁盘空间、失败恢复、进度和验证行为。

现有已发布性能文档和 manifests 能支持的事实直接复用；不为建立基线重复大规模 benchmark campaign。

### 6.2 产物

- 经确认的迁移宪章；
- 五 artifact 功能资格矩阵；
- 代表性性能与加载基线；
- 尚待 Phase 1 解决的身份和产品入口决策清单。

### 6.3 退出条件

- 没有未决问题会改变 v3 的根本产品模型；
- 每个最终交付都有可观察验收方式；
- 不以“代码已编译”代替 artifact、模型和发布路径的完成定义。

## 7. Phase 1：定义序列化无关的领域合同

本阶段先用 owning C++/Python-neutral 概念定义终态事实，再由 Phase 2 将其序列化。JSON key、binary
prefix 和 shard naming 不得反过来决定数学所有权。

### 7.1 Architecture 与 Config 决策

需要确定：

- Dense 和 MoE 的最终 architecture identifiers；
- primary model config 的完整字段、类型、默认值、规范化和验证；
- Text、Vision 和 MoE 参数在 primary config 中的边界；
- MTP、DFlash 等 module architecture/config 的独立字段；
- layer topology、dimensions、token domain、position semantics、state shapes 和数学常量的 owner；
- 哪些 config 在数学上合法但当前 execution capability 不支持，以及对应加载期诊断；
- checkpoint provenance、sampling defaults 和 public model identity 如何与 execution config 配对。

### 7.2 Semantic hierarchy 与 Physical Forms 决策

需要从完整公式、状态、舍入边界、activation sharing、已有 fused Ops、workspace 和性能推导：

- primary Text、Vision、Dense FFN、Sparse MoE 的 semantic hierarchy；
- attention projection、GDN projection/control、post-mixer 及其他闭合区域的输入、输出、状态和
  weight ABI；
- parent-level fused form 与 child-level composition 的合法边界；
- parent 数量、logical row order、zero-copy slices、auxiliary scalars 和跨区域约束；
- numeric format/storage layout 与 physical form 的分工；
- capability resolver 如何从 form 与 descriptor 选择已有实现；
- workspace 聚合、persistent resources 和 CUDA Graph frontier 如何从 resolved results 派生。

### 7.3 Module 与 Attachment 决策

需要确定：

- primary architecture 提供的有限 attachment sites 和 feature-tap ABI；
- module 对 primary shared objects、feature outputs、verification 和 state transactions 的引用；
- module-local name、architecture/config、private objects 和共享引用的 owning types；
- `none`/selected-module 的 capability、workspace、CUDA Graph 和 residency 规划；
- propose/verify/accept/commit/rollback 中 runtime-common protocol 与 module-private schedule 的边界。

### 7.4 三个共同设计见证

领域合同必须同时完整映射以下三类真实输入，不能先围绕单一 artifact 定型：

1. `qwen3.6-27b/groupwise-int`：split Q/K、gate/value、GDN parents 和 groupwise layouts；
2. `qwen3.8-27b/nvfp4`：fused parents、row-scaled FP8、NVFP4、BF16 和 per-layer mixed allocation；
3. `qwen3.6-35b-a3b/groupwise-int`：MoE、MTP、DFlash、feature taps 和 shared primary objects。

另外两个 Dense artifact 必须能够作为上述同一能力集合的不同组合直接表达。

### 7.5 产物与退出条件

产物包括：

- architecture/config owning model；
- semantic hierarchy 与 physical-form ABI 清单；
- primary/module/attachment owning model；
- resolver 输入、不可变输出和错误分类；
- derived config/realization signature 的用途边界；
- 五个 artifact 从当前 objects 到新 semantic bindings 的完整映射草案。

退出条件：仅使用这些领域事实就能解释五个 artifact 的完整数学和物理绑定，不再需要
`weights_id`、target key、完整 profile 或 checkpoint-specific Program 分支。

## 8. Phase 2A：v3 Container、对象模型与分片

### 8.1 工作

基于 Phase 1 类型确定并实现：

- v3 framing、manifest/schema version 和 payload/shard 地址模型；
- primary model、speculative module inventory 和 selected-module-independent artifact contents；
- architecture/config、form binding、object reference、shared object、alias 和 zero-copy view 表达；
- tensor/resource descriptors、numeric formats、storage layouts 和 payload range；
- shard inventory、shard completeness、跨文件 range/identity 和重复/缺失规则；
- 单 logical artifact 的 entry point 和 discovery 规则；
- generic reader 与 architecture binder/resolver 的所有权边界；
- Python writer/reader/inspector 和 C++ reader/materializer 所需 owning types；
- version evolution、unknown member/form/config 的拒绝规则。

Common artifact 层仍不拥有 Qwen 数学、kernel、workspace 或 source conversion recipe。

### 8.2 代表性证据

- C++/Python 对同一 v3 fixture 的交叉读取；
- direct、groupwise、NVFP4、row-scaled FP8 和 resource 的代表性 object round trips；
- alias/shared reference/module reference 的精确解析；
- missing shard、wrong shard、range、overlap、encoded-size 和 binding-reference failures；
- 至少一个真实大 artifact 的分片读取与 materialization，不只使用小型 metadata fixture；
- shard 边界不改变每个 persistent object 的逻辑字节。

### 8.3 退出条件

- v3 container 能完整序列化 Phase 1 的五 artifact 表达；
- C++ 和 Python 对 framing、schema、object、shard 和 reference 语义一致；
- materializer 不再依赖单 file descriptor、单 mmap 或全局 absolute offset。

## 9. Phase 2B：Config 验证、Resolver 与不可变执行输入

### 9.1 工作

1. 实现 architecture-owned config parse、normalize、validate 和 derived dimensions。
2. 从 config 枚举完整 semantic hierarchy。
3. 将 artifact form bindings 和 object descriptors 绑定到 typed semantic regions。
4. 解析每个可达 execution mode 的 implementation capability。
5. 验证跨 region、shared object、attachment、state 和 layout constraints。
6. 聚合 workspace、persistent resources、KV/State extents 和 CUDA Graph frontiers。
7. 形成 Program 可直接消费的 immutable resolved representation。
8. 从 resolved result 派生 diagnostics、context-cost 和 benchmark 所需 signature。

### 9.2 性能边界

- config 动态化只改变加载和规划事实，不在 token/layer hot path 引入 string lookup、descriptor
  interpretation 或通用 virtual graph execution；
- 允许每个 semantic site 持有已解析 enum/variant、typed payload 或直接实现句柄；
- 已有 fixed-shape、fused、artifact-native 和 target-optimized CUDA leaves 必须能够继续被选中；
- workspace 根据实际 resolved sites 聚合，不使用整 artifact profile 的保守替代；
- CUDA Graph capture 前所有 execution route、address 和 resource frontier 已经稳定。

### 9.3 退出条件

- 五个 artifact 的 metadata 可以在不执行 GPU kernel 的情况下完成全量解析或给出明确加载期错误；
- 不支持的 config/shape/form/layout 在首次请求前失败；
- Program 输入不再携带 `weights_id` 或完整 profile。

## 10. Phase 2C：保全 v2 发布物迁移知识

该工作必须在删除当前 inventories、recipes、binders、config headers 和 model-card manifests 前完成。

### 10.1 工作

为每个已发布 v2 artifact 固化：

- 与发布 manifest 一致的 filename、bytes、SHA-256 和 v2 identity；
- v2 directory/object signature；
- architecture identifier 和完整 normalized config；
- 每个 v2 object 的 v3 semantic region、physical form、role 和 view/alias mapping；
- primary/module ownership、attachment 和 shared-object relationships；
- frontend resources 的归属；
- v3 shard placement 所需的确定性资源；
- 哪些 v3 metadata 来自仓库资源而不是 v2 文件；
- 升级输出验证所需的 expected inventory/signature。

### 10.2 产物

- 五份或一份含五个条目的版本化升级资源包；
- 资源生成/审核工具及其来源说明；
- v2 object 到 v3 logical object 的完整无歧义映射；
- 对未知或修改过的 v2 artifact 的明确拒绝条件。

资源包是迁移工具的输入权威，不进入 v3 Engine execution identity。

### 10.3 退出条件

- 即使随后删除旧 target/converter 源码，也能只用已发布 v2 文件和仓库内资源构造完整 v3
  logical artifact；
- 不依赖 source checkpoint、外置 conversion report、网络或 filename guessing 补全执行事实。

## 11. Phase 3：v3 Converter 与 Loader 汇合

### 11.1 Converter 所有权重构

转换流程拆成三个清晰层次：

```text
Source recipe
    checkpoint tensors/resources -> represented logical parameters

Realization assignment
    semantic regions -> registered physical forms and auxiliary values

v3 artifact writer
    config/forms/objects/modules -> logical artifact and shards
```

需要保留 target/source-specific 的精确 source mapping、pinned revisions 和资格验证，但不能继续为
每种 realization 组合复制完整模型 Program 或完整 profile inventory。

### 11.2 五个 source-to-v3 产物

使用现有 source checkpoints 和精确 recipe 生成五个 v3 artifacts：

- converter 从 HF config 生成并验证 NInfer config，而不是只写 sidecar summary；
- 每个 semantic region 显式选择 form；
- Vision、MTP、DFlash 和 frontend resources 按 primary/module 所有权写入；
- writer 按最终 shard contract 输出；
- converter-side verifier 检查 config、binding、descriptor、payload 和 source probes。

### 11.3 证明组合自由度

另生成至少一个不对应任何现有完整 `WeightsProfile` 的内部资格 artifact。它使用当前已有
realizations 构造新的逐层或逐区域混合分配，并满足：

- converter 不增加 checkpoint/profile-specific inventory；
- Engine/Architecture 不增加新的 Program 或 target branch；
- resolver 只根据已有 form/capability 完成绑定；
- 至少执行一个真实 Text 路径并通过相应数值与行为验证。

该 artifact 不要求公开，也不扩大支持 checkpoint 集合；它是防止将 `weights_id` 伪装成另一种
profile 的架构资格证据。

### 11.4 退出条件

- 五个 source-to-v3 artifacts 均能被 generic reader 和 resolver 完整接受；
- source-to-object 语义和所有 format/layout codecs 通过相应 oracle；
- 新混合组合无需 Engine 代码变更；
- converter 不再把完整 profile 作为模型执行权威。

## 12. Phase 4：Dense Architecture 完整纵向切片

### 12.1 首批对照 artifact

先同时切入：

1. `qwen3.6-27b/groupwise-int`，覆盖 split/groupwise forms；
2. `qwen3.8-27b/nvfp4`，覆盖 fused、FP8/NVFP4/BF16 和 layer-mixed forms。

两者必须进入同一个 Dense Architecture Definition 和同一套 config-driven scheduling。release 名称
可以影响 provenance、sampling defaults 或公开显示，但不能选择另一份 Program、binder 或 runtime
schedule。

### 12.2 完整功能范围

纵向切片一次覆盖：

- primary Text model 与 Vision；
- ordinary prefill/decode；
- MTP module、proposal/verify/accept/commit/rollback；
- prefix reuse 和 continuation/state transactions；
- startup-fixed compact batching；
- workspace 与 CUDA Graph；
- CausalScoring；
- CLI、serving、benchmark 和 diagnostics 的公共 Engine route。

不允许先交付 Text-only 临时 architecture，再为 Vision/MTP 建第二套 config/binding 原理。

### 12.3 其余 Dense artifact

首批通过后，将 `qwen3.6-27b/nvfp4` 和 `qwen3.8-27b/groupwise-int` 作为同一 closed capability set
的另外两种组合接入。它们不应要求新的完整 profile 或 checkpoint-named execution package。

### 12.4 退出条件

- 四个 Dense artifacts 的第 4.3 节功能矩阵通过；
- Qwen3.6/Qwen3.8 不在 architecture scheduling 或 Program 中产生 release branch；
- Dense workspace、graph 和 leaf routing 全部来自 resolved results；
- 内部新混合 artifact 真实执行；
- Dense 范围内旧 `WeightsProfile`、checkpoint-specific binder、profile workspace 和
  `qwen3_6_27b` execution package 已删除。

## 13. Phase 5：MoE Architecture 与 Speculative Modules 完整切片

### 13.1 Primary MoE

迁移 35B-A3B primary model：

- MoE Architecture Config 和 derived dimensions；
- hybrid attention/GDN topology；
- sparse experts、routing、shared experts 和对应 realizations；
- Vision；
- primary State/KV、workspace 和 CUDA Graph planning。

### 13.2 MTP 与 DFlash Modules

同一个 v3 artifact 内表达至少 MTP 和 DFlash 两个 module instances，并验证 Engine startup 选择：

```text
none
MTP
DFlash
```

每种选择必须：

- 验证 attachment contract；
- 只解析和上传 selected dependency closure；
- 不为未选 module 分配 GPU persistent storage、workspace 或 CUDA Graph；
- 复用显式允许的 primary objects 和 feature taps；
- 保持各自 propose schedule、private KV/state 和 graph frontier；
- 通过 target verification 和统一 token publication transaction。

### 13.3 退出条件

- 35B-A3B 的第 4.3 节完整功能矩阵通过；
- `none`/MTP/DFlash residency 和行为符合 selected-module 合同；
- 共享对象只存储和上传一次；
- 不再以硬编码 `SpeculativeBackend`、`optional<MtpLayer>` 或 `optional<DFlashPayload>` 作为
  artifact composition 权威；
- 旧 35B checkpoint package、profile binder 和 profile workspace 路径已删除。

## 14. Phase 6：Engine、Registry、产品身份与命名全面切换

重命名随 owner 切换发生。本阶段统一清理跨模块残留，不先进行与语义无关的全仓机械改名。

### 14.1 概念迁移

| 当前概念 | 终态所有权 |
|---|---|
| `(model_id, weights_id)` execution identity | architecture + normalized config + form bindings |
| checkpoint-named target registry | architecture implementation registry/factory |
| `qwen3_6` release-named family runtime | 与真实 Dense/MoE Architecture Definition 对齐的 packages |
| target `config.h` | artifact config + architecture validation/derived values |
| complete `WeightsProfile` | per-region immutable resolved realizations |
| target-private complete binder | architecture semantic binder + capability resolver |
| hardcoded `SpeculativeBackend` | artifact-local module instance selection |
| `LoadSummary.target/model_id/weights_id` | architecture/config/realization summary + separate provenance |
| context-cost `(model_id, weights_id)` key | 从 resolved config/realization 派生的 canonical signature |
| one-file `artifact_path` | v3 logical artifact entry point and resolved shards |

表中描述所有权，不预先规定最终 type、namespace、directory 或 CLI spelling。

### 14.2 受影响边界

统一更新：

- public Engine PIMPL 与 startup options；
- architecture registry 和 active runtime variant；
- Frontend/public model identity 与 execution identity 的边界；
- LoadSummary、request logs、diagnostics、context-cost presets；
- CLI、serve、perplexity、benchmark、eval 和 smoke 工具；
- CMake targets、source directories、namespaces 和 tests；
- artifact inspector、converter reports 和 model manifests。

OpenAI/Anthropic 请求中的 model string 继续是产品协议字段，不允许通过 architecture identifier
无意改变服务合同。

### 14.3 退出条件

- cold startup 只根据 v3 architecture/config/forms 选择和构造实现；
- request hot path 不读取 checkpoint/release identity；
- execution、workspace、graph 或 binder 不再接受 `weights_id`；
- 所有 active source/build/test names 与实际 owner 一致；
- 没有为了兼容旧内部命名而保留 alias packages 或第二 registry lane。

## 15. Phase 7：五个已发布 v2 Artifact 的离线升级

升级器在 v3 writer、五个最终 v3 mappings 和 Engine loader 已稳定后实现，避免让迁移工具成为 v3
合同的反向设计者。

### 15.1 输入资格

- 只接受 Phase 2C 注册的五个已发布 v2 artifacts；
- 将输入与相应 model-card manifest 和升级资源配对；
- 验证 v2 framing、directory、object inventory、bytes 和精确发布身份；
- 是否在 preflight 或流式读取过程中完成 full SHA-256 由具体设计确定，但成功输出必须证明输入
  对应注册发布物，不能只信 filename；
- 未知、修改过、截断或不匹配的输入直接失败。

### 15.2 转换合同

```text
validated published v2 artifact
    + repository-owned upgrade resource
    -> complete v3 config/forms/modules/shard plan
    -> copy existing persistent object bytes without numerical conversion
    -> validate complete v3 artifact
```

升级器可以改变 metadata、object order、file grouping 和 shard placement，但对每个既有 persistent
object 必须保持精确 payload words。若 v3 将一个 v2 object 暴露为多个 logical views，这些 views
必须引用同一已复制 object，而不是生成新权重字节。

### 15.3 用户行为

- 默认不覆盖输入 v2 文件；
- 输出目标必须显式且不能与输入解析为同一破坏性路径；
- 写入开始前报告预计输出和临时空间；
- 提供单调进度和当前阶段；
- 失败时保留原输入并明确说明可否清理不完整输出；
- 成功时打印 v3 identity/config/module/shard summary 和验证结果；
- 整个过程不访问网络，也不要求 source checkpoint 或 conversion sidecar。

原地替换、reflink、hardlink、copy strategy、resume 和 temporary-file policy 在详细设计中根据文件
系统和安全边界决定，本计划不预设实现。

### 15.4 验证

对五个仓库本地真实发布文件分别执行完整升级：

- v2 input 与发布 manifest 对齐；
- v3 reader/resolver/Engine 接受输出；
- 每个迁移 object payload 与 v2 对应 object 精确相等；
- config/forms/modules 与 source-generated v3 语义一致；
- source-generated 和 upgraded v3 不要求 whole-file byte identity，但必须满足同一执行合同；
- 每个 artifact 至少运行一个真实 Engine integration；
- 35B 分别验证 `none`、MTP 和 DFlash selection。

### 15.5 退出条件

- 五个已发布文件都能在无网络、无 source checkpoint 的环境中升级；
- 升级输出通过最终 v3 Engine 和资格矩阵；
- v2 parser 不被普通 Engine、converter 或 inspector 链接/导入为 fallback。

## 16. Phase 8：删除旧路径与最终资格验证

### 16.1 必须删除的旧权威

- Engine v2 reader 和 v2 artifact dispatch；
- runtime `ArtifactIdentity(model_id, weights_id)`；
- complete `WeightsProfile` enums 和分支；
- checkpoint-named target registry/packages；
- target `config.h` 数学权威；
- profile-specific complete binders、workspace 和 Program mismatch checks；
- 旧 profile inventory/converter entry points；
- hardcoded artifact composition 和 speculative backend storage；
- project-owned compatibility aliases、fallbacks 和双 schema paths；
- 已被新 active authority 取代的 checkpoint/profile 文档。

仅 Phase 7 升级工具保留受隔离的 v2 输入合同和所需小型 fixtures/resources。

### 16.2 Container 与迁移验证门

- C++/Python v3 framing/schema/object/shard/reference tests；
- 一个真实分片 artifact 的完整 materialization；
- 五个 source-to-v3 conversions；
- 五个真实 v2-to-v3 upgrades；
- 精确 codec/object payload checks；
- missing/wrong shard、unsupported config/form/layout 和 attachment failures；
- inspector、conversion report 和 published manifest consistency。

### 16.3 模型与运行时验证门

- 每个新或改变的 realization 直接对独立数学 oracle；
- Text、Vision、MTP、DFlash、prefix reuse 和 state transactions；
- ordinary、speculative、forced-control 和 CausalScoring routes；
- compact batching、workspace lifetime 和 CUDA Graph address stability；
- CLI、OpenAI/Anthropic serving、perplexity、benchmark 和 load diagnostics；
- 不支持的 config/shape/form 在加载期失败，不推迟到首次 GPU execution。

不使用最终文本“看起来合理”替代 operator/state correctness，也不要求概率采样输出逐 token 固定。

### 16.4 性能验证门

在 RTX 5090、`sm_120a`、当前 CUDA/toolchain 上比较 Phase 0 基线：

- artifact open/metadata parse、shard I/O、upload time 和 peak staging；
- HBM persistent residency、workspace 和 selected-module dependency closure；
- 代表性 Dense groupwise、Dense FP8/NVFP4 和 MoE prefill/decode；
- MTP/DFlash throughput 与 acceptance；
- concurrency、CUDA Graph 和 prefix reuse 的代表性端到端行为。

任何显著回归必须归因并解决，或在发布前形成明确且被接受的产品取舍。isolated microbenchmark
不能单独证明 end-to-end 架构迁移没有性能退化。

### 16.5 Documentation 与清理门

- 更新 `AGENTS.md` 中的 current product contract 和 ownership boundaries；
- 将 v3 framing/config/form/module/shard 稳定契约写入 active maintainer references；
- 更新 Engine architecture、artifact、model mathematics、storage/tensor format 和 Op references 中
  受影响的事实；
- 更新 README、CLI、serving、performance、tools README 和 docs map；
- 更新五个 model cards、artifact manifests、SHA 和下载/升级命令；
- 删除被取代的 active checkpoint/profile references，不建立平行 `v3`/`new`/`final` 文档树；
- 删除本文和 2026-08-29 讨论笔记；
- 检查所有 active links、stale names、`weights_id` execution references 和 v2 runtime references；
- `git diff --check` 通过。

## 17. Phase 9：Staged Publication 与用户切换

代码和大体积 artifact 不应在彼此不可用的状态下直接切换默认入口。发布按以下顺序准备：

1. 在非默认或暂未公开的发布位置上传五个最终 v3 artifacts、model cards 和 manifests。
2. 从干净 checkout/build 验证两条用户路径：
   - 新用户下载 v3 并直接启动；
   - 已有用户仅使用本地 v2、仓库升级资源和升级工具生成 v3 后启动。
3. 验证 CLI、serve、container/shard discovery、SHA 和文档命令使用真实发布路径。
4. 准备包含 v3 Engine、升级工具、升级资源、active docs 和 release notes 的代码发布。
5. 协调切换 Hugging Face 默认文件、model cards、项目 README/docs 和代码 release/tag。
6. 保留远端 v2 revision/history 作为审计与恢复来源，但不恢复新 Engine 的 v2 执行兼容。

发布说明必须醒目声明：

- 新 Engine 只接受 v3；
- 已下载 v2 用户无需重新下载权重；
- 升级器支持的五个精确输入及验证方式；
- 升级预计磁盘空间和典型命令；
- 旧 Engine 与 v3 artifact、以及新 Engine 与 v2 artifact 均不兼容；
- 如何确认升级后的 artifact/module/shard 完整。

## 18. 跨阶段风险与控制门

| 风险 | 具体影响 | 控制门 |
|---|---|---|
| 在 semantic contract 前冻结 container | schema 反复变化或把 checkpoint inventory 提升为长期 ABI | Phase 1 先完成五 artifact 领域映射，Phase 2 才冻结 serialization |
| 将 `weights_id` 换名后继续作为隐藏 profile | 新组合仍要求完整 binder/Program 分支 | Phase 3 新混合 artifact 无代码变更资格 |
| runtime Config 进入 hot path | dispatch、workspace 或 kernel 性能退化 | Phase 2B immutable resolution 与 Phase 8 end-to-end performance gate |
| 先删除旧代码导致迁移事实丢失 | v2 无法补齐 config/forms/modules | Phase 2C 在删除前固化五份升级资源 |
| 分片只停留在 metadata fixture | 真实 direct I/O、staging 或错误恢复不可用 | Phase 2A 和 Phase 8 真实大 artifact 分片 materialization |
| Dense 单一 artifact 主导抽象 | form ABI 再次绑定某个 profile | Phase 1 三见证共同设计，Phase 4 两个对照 artifact 同时切入 |
| module 统一破坏状态事务 | accept/commit/rollback 或 residency 错误 | Phase 5 `none`/MTP/DFlash 完整事务与 dependency-closure 验证 |
| 大规模机械重命名与语义迁移分离 | 临时名称掩盖错误 owner，产生 alias tree | owner 切换时同步重命名，Phase 6 统一残留检查 |
| 代码与远端 artifacts 发布错位 | 用户下载到无法执行的组合 | Phase 9 staged publication 和两条真实用户路径 |
| 为未来需求建立任意扩展框架 | 偏离可信 closed-capability 产品模型 | v3 只实现已声明 config/forms/modules/shards，unknown values 失败 |

## 19. 阶段状态与更新纪律

初始状态：

| 阶段 | 状态 |
|---|---|
| Phase 0 | 待开始 |
| Phase 1 | 待开始 |
| Phase 2A | 待开始 |
| Phase 2B | 待开始 |
| Phase 2C | 待开始 |
| Phase 3 | 待开始 |
| Phase 4 | 待开始 |
| Phase 5 | 待开始 |
| Phase 6 | 待开始 |
| Phase 7 | 待开始 |
| Phase 8 | 待开始 |
| Phase 9 | 待开始 |

实施期间只在以下情况更新本文：

- 一个阶段正式开始、完成或其退出条件改变；
- 新证据改变后续阶段依赖；
- 发现会改变最终产品合同的阻塞问题；
- 发布范围或五 artifact 迁移范围被明确修改。

不要把逐文件任务、命令日志、原始 benchmark 输出、临时 bug 列表或每日进度写入本文。稳定决策
直接写入对应 active authority；本文只维护仍然有效的跨阶段执行关系。

## 20. 整体完成定义

只有以下条件全部成立，本次迁移才算完成：

1. v3 container、分片、config、forms、primary/modules 和 shared references 存在一个完整 active
   contract；
2. 五个 source-generated v3 artifacts 通过各自完整功能矩阵；
3. 至少一个不属于旧 profile 的新 realization 组合无需 Engine 代码变更即可执行；
4. Qwen3.6/Qwen3.8 Dense releases 使用同一 Architecture implementation；
5. 35B-A3B 能从同一 artifact 启动选择 `none`、MTP 或 DFlash，未选 module 不常驻；
6. 五个精确已发布 v2 files 都能离线升级，既有 persistent object payload 精确保留；
7. 新用户下载和已有用户本地升级两条发布路径均在干净环境验证；
8. Engine、workspace、CUDA Graph 和 execution leaves 不依赖 `weights_id` 或完整
   `WeightsProfile`；
9. v2 reader 只存在于隔离升级工具，Engine 没有 fallback 或双格式分支；
10. 所有适用 Text、Vision、speculative、prefix、CLI、serving、scoring 和 measurement 行为通过；
11. 数值、状态、内存和性能验证足以支持最终产品声明，没有未解释的重大回归；
12. 旧 target/profile 路径、compatibility aliases 和过期 active docs 已删除；
13. 稳定合同已进入 active authorities，本文与前置讨论笔记已删除。

## 21. 第一项后续工作

计划获准后，第一项工作是执行 Phase 0 和 Phase 1 的设计产物，不是修改 v2 prefix、批量重命名
目录或开始复制 target code。具体顺序为：

1. 完成五 artifact 功能验收矩阵和必要基线清单；
2. 确定 Architecture/Config、semantic hierarchy、physical-form、module/attachment 的
   serialization-independent owning model；
3. 用三个共同设计见证完成五 artifact 映射；
4. 审核领域合同能够表达新混合 realization 和分片；
5. 领域合同通过后再分别启动 Phase 2A、2B 和 2C。
