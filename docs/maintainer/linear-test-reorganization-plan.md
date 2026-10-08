# Linear 测试全面整理执行方案

状态：待实施。采用方案 B，将共享辅助机制、各 Op 的验证执行逻辑和 case 声明分开管理。

整理完成后，普通 case 的新增工作应集中在对应格式和计算 profile 的 case 文件中；
新增特殊数值场景只需增加有明确数学含义的 fixture，再使用既有 runner 和 oracle。
当前各 suite 的真实几何、数学语义、数值标准和回归覆盖是迁移约束。

本文是一次性执行方案。长期测试组织、命令和扩展规范最终写入
[Tests](../../tests/README.md)；数学与资格验证原则继续由
[Op development](op-development.md) 和各 [Op 合同](../../include/ninfer/ops/) 管理。
目标结构成为当前实现并完成验收后，将仍有价值的规则归入既有权威文档，再删除本执行方案。

## 整理范围

| 范围 | 实施内容 |
|---|---|
| Linear | 迁移全部 BF16、Q4、Q5、Q6、Q8、FP8、NVFP4 测试，包括旧几何和实施开始时已加入的新几何 |
| LinearAdd | 迁移全部测试，将 FP8、NVFP4 的独立流程收敛到统一的 Op runner |
| LinearPair | 迁移 Q8 测试，显式管理独立权重和共享 parent 行视图场景 |
| LinearSwiGLU | 迁移全部测试，保留完整融合 oracle、稀疏输入策略和 negative gate 回归 |
| LinearTopK | 建立独立目录，统一辅助机制与入口，保留分数、ID、排序、tie、映射及有效行范围验证 |
| LinearDynamicGroupedConvAdd | 建立独立目录，保留 W、B、卷积分组、batch 边界和 residual 语义 |
| 共享 helper 的其他调用者 | 同步迁移受影响的 include、类型和数学基础函数调用，并运行相应回归测试 |
| 构建和文档 | 更新显式 CMake 注册、support library、CTest 入口以及 Tests 中的组织和使用说明 |

Attention/GDN 等独立 Op 的测试架构不纳入本轮全面重写；它们因共享 helper 迁移产生的
调用修改和验证属于完整交付的一部分。产品执行、kernel 调优、模型接入和新增生产几何
仍按各自任务管理。

实施开始时重新检查当前测试。BF16 几何扩展正在进行，不能仅按本方案编写时的固定
shape 列表迁移，也不能覆盖已有的未提交改动。

## 当前结构与迁移重点

当前四个主要目录共有 20 个测试入口；加上 TopK 和动态卷积融合测试，共 22 个入口。

| 当前位置 | 需要解决的问题 | 迁移后的归属 |
|---|---|---|
| `linear/test_bf16_a16.cpp` | 旧几何独立实现 fixture、reference、Graph，新几何使用 common | 一个 Linear runner，保留两类输入 recipe 与全部几何 |
| `linear/test_q4_a16.cpp`、Q5、Q6 | invocation 静态表、重复入口和局部辅助函数 | 格式/profile case 文件和公共入口 |
| `linear/test_q8_a16.cpp` | shape 条件分支追加调用，workspace 检查和主体混合 | 按几何声明 workload，workspace 场景单独登记 |
| `linear/test_fp8_a16.cpp` | 直接调用私有 FP8 validator | 通过公开 Op 或公开准备接口检查非法 metadata |
| `linear_add/test_fp8.cpp`、NVFP4 | 完整流程各自重复实现，A16 与 A8/A4 混在一起 | 各 profile case 文件，共用 LinearAdd runner |
| `linear_add/linear_add_test_common.cpp` | Q5 且 T=128 的 Graph 行为藏在 common 中 | 显式 Graph case，common 不再按格式和 T 自动增添测试 |
| `linear_pair/linear_pair_test_common.cpp` | K 值隐式决定权重存储关系 | case 显式声明 parent、row view 和存储方式 |
| `linear_swiglu/linear_swiglu_test_common.cpp` | 内部维护支持几何白名单；无 GPU 时返回 77 被 main 转成失败 | runner 只检查测试配置自身的一致性；入口统一处理 skip |
| SwiGLU FP8、NVFP4 case 文件 | negative gate 自行上传、执行和比较 | 专用 fixture 加同一 Op runner，保留所保护的精度风险 |
| 根目录 TopK、动态卷积融合测试 | fixture、Graph、采样和验证代码混合 | 各自独立目录、oracle、case 和 support library |

完整参考文件：
[Linear common](../../tests/ops/linear/linear_test_common.cpp)、
[LinearAdd common](../../tests/ops/linear_add/linear_add_test_common.cpp)、
[LinearPair common](../../tests/ops/linear_pair/linear_pair_test_common.cpp)、
[LinearSwiGLU common](../../tests/ops/linear_swiglu/linear_swiglu_test_common.cpp)。

## 最终目录与构建入口

现有全 Op 基础设施继续放在 `tests/ops/`。投影测试共享机制放入
`linear_common/`，各 Op 的 suite 只依赖这层以及自己的语义合同。

~~~text
tests/ops/
├── op_tester.h
├── op_check.h
├── quantized_weight.h
├── direct_bf16_weight.h
├── linear_common/
│   ├── case.{h,cpp}
│   ├── sample.{h,cpp}
│   ├── fixture.{h,cpp}
│   ├── execution.{h,cpp}
│   ├── reference.{h,cpp}
│   └── tests.cmake
├── linear/
│   ├── main.cpp
│   ├── suite.{h,cpp}
│   ├── oracle.{h,cpp}
│   ├── criteria.h
│   ├── fixtures.{h,cpp}
│   ├── cases/
│   │   ├── bf16_a16.cpp
│   │   ├── q4_a16.cpp
│   │   ├── q5_a16.cpp
│   │   ├── q6_a16.cpp
│   │   ├── q8_a16.cpp
│   │   ├── fp8_a16.cpp
│   │   ├── fp8_a8.cpp
│   │   ├── nvfp4_a16.cpp
│   │   └── nvfp4_a4.cpp
│   └── tests.cmake
├── linear_add/
├── linear_pair/
├── linear_swiglu/
├── linear_topk/
└── linear_dynamic_grouped_conv_add/
~~~

后五个 Op 目录采用同样的职责分工。只有存在 Op 专用 fixture 时才建立
`fixtures.{h,cpp}`，小型实现可以合并对应文件，避免产生空文件和纯转发层。

每个普通格式/profile executable 编译对应 Op 的 `main.cpp` 和一个
`cases/<profile>.cpp`，再链接该 Op 的 support library。
case 文件提供同一个具名的 `make_suite()` 工厂，返回有明确所有权的测试描述；
main 调用该 Op 的 runner。工厂在链接时由显式源码选择，测试字符串只用于筛选。

建议的最终入口集合：

| Op | profile 文件及入口后缀 | 数量 |
|---|---|---:|
| Linear | bf16_a16、q4_a16、q5_a16、q6_a16、q8_a16、fp8_a16、fp8_a8、nvfp4_a16、nvfp4_a4 | 9 |
| LinearAdd | bf16_a16、q4_a16、q5_a16、q8_a16、fp8_a16、fp8_a8、nvfp4_a16、nvfp4_a4 | 8 |
| LinearPair | q8_a16 | 1 |
| LinearSwiGLU | q4_a16、q8_a16、fp8_a16、fp8_a8、nvfp4_a16、nvfp4_a4 | 6 |
| LinearTopK | q4_a16、q8_a16、fp8_a16 | 3 |
| LinearDynamicGroupedConvAdd | q8_a16 | 1 |
| 合计 | 统一采用 `ninfer_<op>_<profile>_test` | 28 |

这是现有测试的重新划分，不代表新增生产格式支持。TopK 的三个输入格式分别成为入口，
动态卷积融合的两个输入几何放在同一 Q8 suite 内。现有已规范命名的入口保持名称；
被拆分或替换的聚合入口和源码注册在切换时完整移除。

## 共享辅助层的职责

| 模块 | 负责内容 | 关键约束 |
|---|---|---|
| case | 元信息、分类、列数 domain、边界展开、稳定 ID | 只描述测试 workload，不维护生产支持或 dispatch registry |
| sample | 全量、等距、quarter 点、显式边界索引与 gather | 行选择和列选择分别配置；actual 与 oracle 使用同一逻辑索引顺序 |
| fixture | 输入 pattern、表示转换、权重上传、只读 payload 和 guard | 保留 BF16 和 packed host 存储的独立生成与解码；所有设备资源显式归 fixture 所有 |
| execution | 参数解析、筛选、环境检查、Graph 生命周期和阶段执行 | Op 提供输入准备、公开调用和阶段验证；共享层不推导数学结果 |
| reference | 独立 naive FP64 点积及投影基础函数 | 固定完整 K 累加顺序；无生产 Op、selector、launcher 或 kernel 调用 |

`op_tester.h` 继续管理全 Op 通用的 CUDA 检查、传输、guard、报告和基本比较入口；
`op_check.h` 继续管理比较算法。矩阵测试专用的批量扫描、采样读取和上传封装
进入 `linear_common/`。各函数只有一个当前实现。

输入生成以命名 recipe 管理。迁移阶段保留现有生成公式、seed、幅度、scale/code pattern
及 NVFP4 divisors；同一个 recipe 可复用于多个 Op，Op 特有的 residual、gate/up 和
权重行关系由其 fixture 管理。把不同生成器改成一个随机分布属于数值 workload 变更，
需要独立记录其理由和影响。

## 各 Op 保留的职责

每个 Op 的 `suite` 管理自己的 case 类型、准备接口、公开重载、workspace 查询、
输出和 mutation 验证；`oracle` 管理完整数学结果；`criteria.h` 管理具名数值标准。

| Op | 独立 oracle 与专用验证 |
|---|---|
| Linear | 从逻辑 FP32 权重与 BF16 表示输入计算完整 K 的 FP64 投影 |
| LinearAdd | FP64 投影加初始 residual，保留 in-place 更新语义 |
| LinearPair | 分别计算两组权重投影，正确解释 shared parent 的两个行视图 |
| LinearSwiGLU | 独立计算 gate、up，再计算 FP64 SiLU 与乘积 |
| LinearTopK | 计算 score，独立确定候选顺序、tie 和 ID 映射；分数近似比较与整数精确验证分开 |
| LinearDynamicGroupedConvAdd | 投影、系数组合、前一列引用和 residual 的完整融合公式，保留 batch 起点边界 |

共享 FP64 投影基础函数可以被融合 oracle 使用，oracle 仍保留高精度中间结果，
不经由生产 Linear，也不加入生产 staging、activation quantization 或额外 BF16 rounding。

SwiGLU 当前通过第一列 dense、后续列精确零与旋转非零值保持全输出 oracle 的可执行成本。
保留这一输入 recipe 和完整公式；跳过精确零项时保持其余 K 项的累加顺序。
TopK 的构造性候选 fixture 也继续保留，避免统一基础函数后无必要地计算整个 vocabulary
的 dense reference。

## Case 数据模型与组织规则

共享类型采用组合。LinearPair 的 row view、TopK 的 ID map、动态卷积的 W/B
由各自 case 类型表达；所有 Op 采用一致的命名、初始化和执行规则。

| 层级 | 信息 |
|---|---|
| Suite | Op、权重格式、命名计算 profile、case 集合 |
| 场景 | 稳定 ID、风险分类、几何、种子与输入/权重 recipe、必要的语义参数 |
| Invocation | 实际列数或 W/B、policy、公开调用形式、eager/Graph、采样计划、replay 计划 |
| 执行阶段 | 当前表示输入、初始 residual、期望结果、有效输出区域和只读检查对象 |

数值 criterion 由 suite 拥有。case 和 invocation 没有任意 tolerance 字段。
policy 表示允许的计算集合，和命名数值验证 profile 分开。
现有 permissive policy 下仍验证 A16 的点归入 A16 suite；
对应 A8/A4 的点归入其 suite，边界两侧都保留。
case 不读取私有 plan 来选择 criterion，也不在 runner 里复制 selector 条件。

声明使用 C++20 designated initializer。以下是目标接口的示意：

~~~cpp
Case make_boundary_case() {
    return Case{
        .id = "n5120_k6144_dense_boundary",
        .category = Category::Boundary,
        .geometry = {.n = 5120, .k = 6144},
        .seed = 429U,
        .input = dense_recipe(),
        .weight = q4_small_hashed_recipe(),
        .invocations = {
            {.columns = 31, .policy = ops::LinearPolicy::A16Only,
             .call_form = CallForm::Policy, .execution = Execution::Eager,
             .samples = full_output()},
            {.columns = 32, .policy = ops::LinearPolicy::A16Only,
             .call_form = CallForm::Policy, .execution = Execution::Graph,
             .samples = full_output(), .replay = changed_activation()},
            {.columns = 33, .policy = ops::LinearPolicy::A16Only,
             .call_form = CallForm::A16Convenience, .execution = Execution::Eager,
             .samples = full_output()},
        },
    };
}
~~~

case 工厂生成的动态 invocation、采样数组和 replay 描述由返回的 suite 自己持有。
`std::span` 只用作执行时视图，不能指向工厂中的局部数组。

文件内按数值基础、边界与尾部、数值回归、接口与资源、Graph 场景分组。
这些维度可以由同一 case 同时覆盖，不展开格式、shape、T、policy、mode 的冗余全笛卡尔积。

较长的几何调用表采用命名 workload builder，例如
`make_n2560_k6144_cases()`。builder 只产生描述，不能上传数据或调用 CUDA。
多个 shape 的共有点和专属点通过显式合并形成 invocation，替换主体中的连续 shape 条件分支。

边界展开接收明确的列数 domain：

- Text/MTP 使用正 T；边界取合法的 b-1、b、b+1，再加入代表性内部点。
- Vision raw patch 按 P 的合法步长与上界展开；merged token 按 V 的 domain 展开。
- composite wave 的 offset 和 tail workload 在 case 中明确声明，公共函数执行展开。
- 去重比较整个测试身份，包括 recipe、seed、policy、重载、执行方式、采样和 replay；
  同一个 T 的不同验证不能被合并掉。
- runner 只检查正 extents、采样索引、row view、invocation 生命周期等测试配置一致性。
  支持域由公开 Op 判定，case 清单不是第二份生产能力注册表。

## 采样与结果检查

采样计划区分输出行和输出列，至少支持全量、等距点、quarter 点和显式索引。
现有 BF16 的结构边界行、七点采样和 32 点采样分别迁移，先保持原集合。
一个 invocation 可做 full-output，其他 invocation 做 sampled-output，从而共享同一 fixture。

数学采样与输出完整性是两个不同检查。对要求有限输出的 fixture：

1. 全部有效输出检查 non-finite；纯输出 Op 在执行前 poison，以发现未写区域。
2. 被选输出与完整 K 的 independent oracle 比较。
3. 每次执行检查有效 payload 边界的 guard。
4. 只读输入和权重检查 payload 保持及 guard。
5. workspace 检查按该 Op 的查询和释放要求执行。

LinearAdd、动态卷积融合的 residual 是输入兼输出，每次执行前写入本阶段的初始值，
使用初始值计算 oracle。为它们套用纯输出 poison 会改变数学输入，runner 应按 Op 区分。

大 shape 的 payload preservation 可在共享 fixture 组结束时检查，Graph 修改输入前还要
检查前一阶段承诺保持的内容，防止测试自身的下一次上传掩盖破坏。
新增的全输出扫描、guard 和 preservation 检查分别登记为覆盖增强，计入运行成本比较。

输出按本 invocation 的有效大小设置 guard；复用更大的 allocation 时仍必须保护实际
T/W/B 的有效边界。共享 input 可以按最大列数分配，Graph 捕获使用的地址保持稳定。

## 数值标准与既有例外

迁移前登记各 Op 的 relative L2、gross absolute、gross relative、non-finite 规则
以及它们的数学理由。数值标准保持由各 Op/profile 命名管理，公共比较算法继续复用
`op_check.h`；共有数值不合并成跨 Op tolerance preset。

以下项目必须明确处理：

| 项目 | 处理要求 |
|---|---|
| BF16 旧流程与公共 Linear 的重复 A16 criterion | 归入同一 Linear A16 criterion，并保留旧输入 recipe 与采样范围 |
| FP8/NVFP4 LinearAdd 的局部 criterion | 迁入 LinearAdd 的命名 criteria，profile 划分保持一致 |
| SwiGLU negative gate 的更严局部 criterion | 保留原有更严证据和精度风险；将精确可表示输入的要求整理为有数学依据的 suite 级性质检查，在完整适用域统一运行 |
| workspace 的零容量和 exact peak 检查 | 分清合同要求与当前实现假设；保留合同验证，不能用较宽条件静默替换 |
| 同一数学 profile 内按 T 切 criterion 的逻辑 | 由声明的验证分组与数值理由解释，不能照搬生产阈值自动选择 tolerance |

negative gate 目前存在单独的更严阈值，和禁止随 case 定义 tolerance 的统一规则需要
在基线审计中对齐。只给常量换名字不能算解决：必须说明精确表示域及其允许误差，
保留实际会检测的投影精度损失。一般 profile 比较继续使用统一 criterion，性质检查
作为有明确依据的附加验证。该项未解决时标记为未完成，不能用一般 A8/A4 阈值覆盖原检查。

数学 reference 保持权重的逻辑 FP32 表示、输入的实际 BF16 表示和 naive FP64 累加。
测试组织变化本身不要求概率输出、kernel parity、生产 route 之间或旧新 oracle 之间
逐 bit 相等；资格验证始终落在公开 Op 对独立数学 oracle 的结果上。
若原有完整 K 顺序和表示边界被改变，先解释并验证变化，不能将其隐含在文件迁移中。

## Graph 执行协议

公共执行层复用现有 `DeviceContext`、`DecodeGraphDefinition` 和
`DecodeGraphExecutable`，统一所有权和异常路径释放。每个 Graph invocation：

1. 完成 fixture 初始化，确保捕获资源和指针有效。
2. 捕获一次公开 Op 调用并实例化。
3. 准备第一阶段输入和初始输出/residual，执行并完整验证。
4. 按 case 的 replay 计划更新输入内容，保持地址不变，再执行并完整验证。
5. 检查只读内容、output guard 和 workspace，恢复后续 case 所需的 fixture 状态。

stage 的数据准备和 oracle 由具体 Op 提供：

| 场景 | 期望结果 |
|---|---|
| Linear 输入取反 | 可以使用原 FP64 reference 的负值 |
| LinearAdd 输入取反 | 原 residual 减去 FP64 projection |
| LinearAdd 输入为零 | 原 residual 的数学值 |
| SwiGLU 输入取反 | 重新计算 SiLU(-gate) 乘以 (-up) |
| TopK 输入改变 | 重新求对应分数与候选顺序，保留 tie 和 ID 规则 |
| 动态卷积输入改变 | 从本阶段 projection、kernel/delta 和初始 residual 重算完整结果 |

保留现有同输入重复 replay；改变输入 replay 用来覆盖捕获地址处数据是否被重新消费。
检查每个被测阶段，替换只验证最后一次 replay 的流程。
LinearAdd 的三阶段原输入、零输入、恢复输入显式放入 replay 计划，
移除 common 中的 Q5/T 特判。

覆盖补齐阶段，LinearPair 的两类存储场景各增加一个代表性 changed-input Graph case；
TopK 和动态卷积融合各增加代表性改变输入验证。公开调用和数学合同保持不变，
新增验证数量单独记录。

## 统一命令与报告

所有新入口采用相同基础命令：

~~~text
<test>
<test> --help
<test> --list
<test> --shape N K
<test> --case CASE_ID
~~~

默认执行完整 suite。`--shape` 选择 case 中声明的逻辑几何；
TopK 使用实际 weight N/K，动态卷积融合使用输出 N 和输入 K，
W/B 继续由选中的 case 描述。`--case` 选择完整 invocation，包括其全部 replay 阶段。

`--list` 输出稳定 ID、几何和语义轴、profile、policy、重载、执行方式、
采样范围和输入 recipe，并可与筛选组合。目录描述和筛选在 CUDA 初始化之前完成，
使 `--help` 和 `--list` 能在无 GPU 环境使用；它们不能生成大权重或分配设备内存。
未知参数、非法参数、重复 ID 和筛选为空明确失败。

进程入口统一负责环境和状态：

- 正常运行通过返回 0，比较失败或异常返回 1。
- 无可用 CUDA 设备的运行请求返回 77，由 CTest 识别为 skip。
- suite 内部只返回比较失败计数或抛异常，不能把 77 混入失败计数。
- CUDA 异常携带完整 case 上下文，不以设备缺失掩盖运行错误。

保留 `NINFER_OP_REPORT_STATS` 和 `OP_ERROR_STATS` 机制，完善其中的 case 标签。
例如：

~~~text
linear/q8_a16/n2560_k6144/dense/t50/policy_a16/form_policy/graph/phase1/sampled
~~~

实际 ID 必须区分 seed、recipe 和不同采样计划；阶段标签还区分输出名称和 replay phase。
有限值、guard、workspace、metadata 等失败共享同一基础 case 身份。
默认成功输出只需 suite 汇总，stats 开关不改变验收判定。

## 共享 helper 调用者与源码归属

`input_projection_test_common.h` 的以下当前调用者要与 helper 迁移同步处理：

| 调用者 | 需要保持的验证 |
|---|---|
| `test_attn_input_proj.cpp` | 输出拆分、多输出、准备权重和投影数学 |
| `gdn_input_proj/test_projection.cpp` | Q/K/V/Z 的行范围和投影数学 |
| `gdn_input_proj/test_snapshot.cpp` | 输出、卷积状态和快照语义 |
| `gdn_input_proj/test_record.cpp` | record 内容与未触及区域 |
| `linear_attention/gated_delta_net/test_fold.cpp` | record/fold 与规定 snapshot reference 的精确等价 |
| `test_linear_dynamic_grouped_conv_add.cpp` | 迁入新目录并保留完整融合公式 |

将其中采样、BF16 tensor、权重设备持有、表示转换、preservation 和投影基础函数
迁入对应的公共模块后，更新所有调用者，移除旧 helper 的已替代实现。
调用者使用新的显式 namespace，不保留旧 namespace alias 或兼容转发头。

`direct_bf16_weight.h` 和 `quantized_weight.h` 继续拥有实际权重 fixture 与独立解码；
新的 fixture 封装统一使用方式，避免为了统一接口给大型 BF16 再复制一份完整 payload。
如其公共接口确需变化，搜索全部真实调用者并同步修改、编译和验证；
SparseMoE、cache materialization 等仅在被实际接口变化影响时纳入回归范围。

每个非 header-only 实现只编译一次：

- 一个 `ninfer_linear_common_test_support` library，包含通用机制与 FP64 投影基础函数。
- 每个 Op 一个 `ninfer_<op>_test_support` library，包含 suite、oracle 和需要的专用 fixture。
- executable 只包含 main 和选定 profile 的 case 源码。
- support library 和 executable 都通过现有测试 include/link helper 注册，链接 Op/core 所需依赖。
- 新 `reference.cpp`、各 `oracle.cpp` 及其所属 target 应用
  `ninfer_op_oracle_options()`；不能假定 executable 的选项会传入静态 library。

TopK 当前的测试源文件没有自定义 device kernel，迁移后采用 `.cpp`；
CPU oracle 因而受 CXX 的 `-fno-fast-math` 与 `-ffp-contract=off` 规则保护。
只有实际编写测试 device 代码的翻译单元使用 `.cu`。

`tests/ops/tests.cmake` 包含公共 support 和六个 Op 的 manifest。
manifest 显式登记全部 profile 源码及 target，沿用
[测试注册 helper](../../tests/cmake/NinferTests.cmake)；
目录文件数量变化不自动改变 CTest 覆盖。

## 分阶段实施任务

各阶段先完成当前迁移和直接验证，再进入下一阶段。
迁移后的测试主体完整切换，删除被替换的入口和 helper，避免长期保留两套流程。

| 阶段 | 工作 | 完成条件 |
|---|---|---|
| P0 基线与数值审计 | 登记当前实际 case、recipe、采样、criterion、Graph、workspace 和非法参数覆盖；运行基线 | 覆盖与已有失败有明确记录，数值例外有处理决定 |
| P1 公共机制与最小切片 | 实现公共 case/sample/fixture/execution/reference；选 Linear Q8 一个已有几何接通 main、case、runner 和构建 | 独立数学比较、Graph、workspace、筛选和日志全链路可用 |
| P2 全量 Linear | 迁移九个 profile，包括 BF16 旧流程、所有新增几何及 Vision domain | 单一 runner 和 oracle；Q8/BF16 筛选覆盖全部已登记几何 |
| P3 LinearAdd | 接入 BF16/Q4/Q5/Q8；拆分 FP8/NVFP4；声明 residual 和 replay 场景 | 八个 profile 入口，精确融合 oracle 与原回归均保留 |
| P4 LinearPair | 显式 parent/row-view fixture；接入共同机制与两输出验证 | 两种存储关系都通过；独立 oracle 和准备接口覆盖保留 |
| P5 LinearSwiGLU | 拆分 profile；迁移稀疏 recipe、negative gate、Graph 与 criterion | 六个入口，无支持白名单重复；数值例外已对齐规则 |
| P6 TopK 与动态卷积融合 | 建立目录；TopK 分为三个 profile；动态卷积保留 W/B case | 四个入口，各自专用数学、整数和边界语义通过 |
| P7 调用者与覆盖补齐 | 完成其他 helper 调用者迁移；补齐已确认的接口、Graph、完整性检查缺口 | 无遗留依赖，新增覆盖有对应风险和成本记录 |
| P8 全量验收与权威文档 | 全部受影响目标和调用者通过，更新 Tests，清理旧文件和注册 | 完成最终核对，报告收益、成本、失败与未验证项 |

P1 的一个几何是接口设计切片，P2 仍需迁移完整 Q8 suite。
每阶段主要目标具备直接 oracle 验证，迁移阶段和覆盖补齐阶段的新增断言分开记录。

可按以下清单跟踪执行：

- [ ] P0 当前覆盖清单与基线结果
- [ ] P1 公共机制与一个完整切片
- [ ] P2 全量 Linear
- [ ] P3 全量 LinearAdd
- [ ] P4 全量 LinearPair
- [ ] P5 全量 LinearSwiGLU
- [ ] P6 TopK 与动态卷积融合
- [ ] P7 helper 调用者与确认缺口
- [ ] P8 验收、文档与旧实现移除

## 基线与验证方法

P0 从现有源码展开实际 case，记录以下身份信息：
Op、格式、数值 profile、几何及语义轴、T/W/B、seed、输入/权重 recipe、
policy、调用重载、采样行列、执行方式、replay 阶段和 workspace 检查。
覆盖记录可放在本次实施的临时结果目录；当前 case 声明仍是长期覆盖的唯一来源。

迁移后用 `--list` 和实际执行核对这些身份信息。拆分入口会增加 executable 数量，
不能用 CTest test 数量比较代替 case 覆盖比较。
任何 case 合并、采样变化或数学标准变化都说明理由和原风险如何继续得到保护。

选择现有构建目录与相同工具链建立基线。需要新配置时使用维护环境的 Python：

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DPython3_EXECUTABLE=/home/neroued/miniconda3/envs/py311/bin/python
cmake --build build -j
ctest --test-dir build -N -R '^ninfer_linear_.*_test$'
ctest --test-dir build -R '^ninfer_linear_.*_test$' --output-on-failure -j 1
~~~

阶段内编译受影响 target 并运行对应 suite；完成 P7 后额外运行以下受影响调用者：

~~~bash
ctest --test-dir build -R '^ninfer_(attn_input_proj|gdn_input_proj|gdn_input_proj_conv_snapshot|gdn_input_proj_conv_record|gdn_replay_fold)_test$' --output-on-failure -j 1
~~~

实际接口变化影响到其他 Op 时，将对应 target 加入该阶段的验证集合。
GPU 资格验证串行执行，构建默认使用 `cmake --build <build-dir> -j`。
stats 用于检查数值例外、误差分布或失败，不要求为每次通过重复保存全部输出。

验收关注真实可观测行为：

| 维度 | 检查 |
|---|---|
| 数学与数值 | 各公开入口直接对独立 oracle；完整 K、融合公式、非有限规则和命名 criterion |
| Case 覆盖 | 原有几何、合法列数 domain、边界、内部点、seed、recipe 和特殊回归 |
| Graph | 捕获、重复执行、内容变化、每阶段结果、稳定地址、residual 复位 |
| 存储与副作用 | 多输出与 row view、preservation、有效区域 guard、完整输出检查 |
| Workspace | 合法/非法区间、区间覆盖实际 point、执行 peak 与调用结束释放要求 |
| 运行入口 | 默认完整运行，help/list、shape/case 筛选、空匹配、退出码 0/1/77 |
| 构建与归属 | 显式源码注册、support 单次编译、oracle 选项、host-only CXX 编译 |
| 受影响调用者 | 投影数学、snapshot 状态，以及 record/fold 的规定精确 reference |
| 资源与维护 | 按原 family/format 聚合耗时，新增 fixture/扫描成本，新增 case 修改位置 |

数值检查由生产 Op 对 independent oracle 决定；旧新 helper 的一致性只用于解释迁移，
不能替代数学验证。原基线失败单独记录和定位；迁移引入的失败需要解决，
已有生产问题不能通过删 case 或放宽标准变成通过。

现有 suite 已包含很大的 N 或 T。输入/权重 fixture 和允许复用的 reference 按几何、
recipe、seed 共享，输出和 mutable buffer 按执行阶段正确复位；一次只驻留当前必要资源。
保持分块输出扫描、sampled oracle 和 SwiGLU 稀疏 recipe。
禁止因数据模型统一而把大 shape 的 sampled reference 默认改成全矩阵 reference。

记录同一机器、构建类型、CPU oracle worker 策略和串行条件下的 suite 耗时。
拆分 profile 后按原 family/format 聚合比较，同时报告最慢 suite 和任何回退。
运行成本变化区分代码迁移、重复 fixture 初始化和新增覆盖；
不据此宣称 Op 或端到端推理变快。必要时对有疑问的 suite 定点复测，
不把测试整理扩大成 kernel benchmark campaign。

公共边界展开、采样或筛选算法若存在非平凡逻辑，可用有限的 host-only 检查保护
越界、遗漏、重复身份和空匹配等实际风险；优先利用真实 case 展开结果验证。
不添加只检查文件名、类名、私有组织或覆盖率数字的测试。

## 覆盖补齐的具体边界

在机械迁移完成后，按以下已确认的缺口增加最小检查集合：

- Linear 的 metadata 拒绝测试通过公开边界执行；BF16 全部几何支持统一筛选。
- LinearAdd、LinearSwiGLU 的 convenience 和 policy-bearing 公开重载都有直接数学证据，
  保留 permissive policy 下的合法 A16 场景。
- NVFP4 SwiGLU A16 convenience form 保留 T=16 合法边界，并验证 T=17 的合同拒绝；
  较大合法 extents 通过 AllowA4 入口验证。
- LinearPair 检查公开 workspace query 的零容量和非法区间，两个输出分别验收；
  两种存储场景各有代表性 changed-input Graph。
- FP8/NVFP4 LinearAdd 补齐原来仅采样位置的有限值检查，以及逐阶段 Graph 验证。
- TopK 保留完整返回 ID 的精确规则，动态卷积保留 batch 起点无前一列贡献；
  两者补充代表性内容变化 replay。
- 所有数值入口执行自身合同要求的 preservation、guard 和 workspace 检查。
  新断言登记其风险、适用域和运行成本，避免扩展出无依据的全组合覆盖。

SwiGLU 更严负 gate 标准、workspace exact peak 与数学 profile 分组若审计发现不一致，
先完成数值/合同依据的核对再进入相应阶段验收。
生产 kernel 的实际缺陷单独列出；测试迁移不能掩盖它，也不自动授权调优或扩充生产支持域。

## 完成条件与交付报告

全面整理完成必须同时满足以下条件：

- 六个 Op 目录按统一职责组织，全部已保留 profile 都有明确入口。
- 原 common 中的测试场景选择被移到 case 声明，case 文件不重复完整 GPU 测试流程。
- 全部现有 workload 和回归已迁移，新增检查和有理由的覆盖调整可以逐项解释。
- 数学 oracle、表示边界和命名 criterion 明确；数值例外完成对齐。
- 每个被测 Graph 阶段有结果验证，0/1/77 状态和筛选行为一致。
- helper 的所有受影响调用者迁移完成，旧文件、别名、转发和注册被移除。
- oracle 编译选项、单一源码归属和 CTest 完整注册经过检查。
- 受影响 suite 和调用者完成直接资格验证，运行成本和资源回退已解释并有处置。
- Tests 成为当前组织与命令的权威，旧 common 名称、聚合入口和错误描述全部更新。
- 文档明确已完成、未完成和未验证事项，未解决的数学或覆盖问题不被报告为完成。

交付报告至少说明最终目录与入口、case 覆盖保留/增加/减少、
数值与 Graph 验证结果、受影响调用者、构建条件、耗时分布和最坏回退、
数值例外的处理、现存失败及其处置。收益主要体现在新增 case 所需修改的位置、
共享函数的唯一归属、筛选与失败定位能力；未测量的编译、内存和运行收益不作结论。

新增普通 case 的维护流程写入 Tests：

1. 选择对应 Op 与 profile 的 case 文件。
2. 声明风险、几何、recipe、invocation 和采样/replay 计划。
3. 使用现有 fixture 与 runner；有新数学场景时增加具名 fixture。
4. 通过稳定 ID 运行新 case，再运行受影响 suite。
5. 仅新增格式/profile 时修改 manifest；新增 production shape 的支持仍由其独立任务负责。
