# Linear 测试整理待办

状态：未完成。当前仍有 Linear 9、LinearAdd 6、LinearPair 1、LinearSwiGLU 4、
TopK 和动态卷积融合各 1 个入口，共 22 个。共享 packed-weight fixture、三个 common
support library 和部分 changed-input Graph 已存在；统一的 case/runner 组织与入口协议
尚未完成。本次文档整理不表示这些实现或验收已经完成。

长期原则由 [Op development](op-development.md) 和 [Op 合同](../../include/ninfer/ops/)
管理；当前命令由 [Tests](../../tests/README.md) 管理。以下待办完成后，将仍需长期维护的
使用说明归入 Tests，再删除本文。范围限于六个 Op 的测试及实际受影响的 helper 调用者。

## 待完成的收敛

保留方案 B 的职责分离：共享采样、fixture、执行和 FP64 投影基础机制；各 Op 独立管理
完整 oracle、runner 和 criterion；格式/profile 文件声明 case。普通 case 不重复 GPU
执行流程，共享 runner 不选择测试场景或复制生产支持白名单。允许小型实现合并文件，
不为目录形式建立空层或转发层。

| Op / 机制 | 当前问题与待完成结果 |
|---|---|
| Linear | BF16 旧流程仍独立，`--shape` 只覆盖新增几何；接入同一 runner，并覆盖全部原有和新增几何。FP8 metadata 拒绝从私有 validator 迁到公开 Op/准备接口。 |
| LinearAdd | FP8/NVFP4 各自重复执行流程，A16 与 A8/A4 混合；接入共同 runner、按 profile 声明 case。把 common 中 Q5/T128 的三阶段 Graph 特判移入 case。 |
| LinearPair | K 隐式决定独立权重或 shared-parent row view；case 显式声明存储关系，两输出各自验证。 |
| LinearSwiGLU | common 仍维护支持几何白名单，无 GPU 的 77 被 main 转成失败；统一入口状态，删除重复 admission。negative-gate fixture 接入共同 runner，并保留更严证据。 |
| LinearTopK | 根目录单体混合 fixture、数学与整数检查；按 Q4/Q8/FP8 声明 profile。无 device kernel 的 `.cu` 改为 `.cpp`，使 CPU oracle 获得 CXX 算术选项。 |
| LinearDynamicGroupedConvAdd | 根目录单体混合 fixture、Graph、采样和融合验证；建立独立 suite，保留 W/B、分组、batch 边界及 residual 语义。 |
| 共享执行 | 统一 help/list、shape/case 筛选、Graph 生命周期、阶段验证与 0/1/77；消除旧辅助实现和转发别名。 |

预期 profile 入口数为 Linear 9、Add 8、Pair 1、SwiGLU 6、TopK 3、动态卷积 1。
拆分现有入口不代表新增生产格式支持，也不能用入口数证明 case 覆盖保留。

case 显式持有几何、seed、recipe、policy、公开调用形式、采样和 replay 描述，
`span` 只作执行视图。使用具名字段初始化，避免位置参数错位。共享边界展开按 Text
正 T、Vision raw-patch 步长/上界或 merged-token domain 工作；去重比较完整 invocation
身份，同一 T 的不同 recipe、重载或 replay 不得合并。runner 只验证测试描述自身一致性。

## 必须保留的样本与数学约束

迁移前从当前源码登记真实 case，涵盖全部几何、T/W/B、seed、生成公式、幅度、
scale/code pattern、NVFP4 divisors、policy、公开重载、采样行列与 Graph 阶段。
实施时重新读取源码，保留已有扩展和未提交改动。case 声明是长期覆盖的唯一来源，
临时基线清单不成为第二份注册表。

- 保留旧 BF16 recipe、结构边界行、七点与 32 点采样，以及各格式完整/采样比较范围。
  大 shape 保持 sampled reference 和分块输出扫描，不因统一接口改成默认全矩阵 oracle。
- oracle 独立解码实际权重与 BF16 输入，保持完整 K 的 naive FP64 累加顺序。
  Add、Pair、SwiGLU、TopK 和动态卷积各自保留完整融合公式；不调用生产 Linear 作为 oracle，
  不复制私有 activation quantization、staging、reduction tree 或额外 BF16 rounding。
- 保留 SwiGLU 第一列 dense、后续精确零与旋转非零值的稀疏 recipe；跳过零项时保持其余
  K 项顺序。TopK 保留构造性候选、完整返回 ID、tie、映射和有效行范围检查。
- 数学 criterion 按 Op/profile 命名，保留现有 relative L2、gross absolute/relative 和
  non-finite 规则；不得新增 case tolerance 或从 private selector 自动选择阈值。
- FP8/NVFP4 negative-gate 的精确表示输入当前使用
  `{relative_l2=1/256, gross_absolute=0, gross_relative=2/256}`。迁移必须保留其检测投影
  精度损失的能力，并明确为有数学依据的性质检查；不能用一般 A8/A4 阈值覆盖，也不能
  只改常量名称就宣告数值例外已解决。
- workspace 的零容量和 exact-peak 断言先区分公开合同与实现假设，保留合同验证。
  同 profile 按 T 切 criterion 的现状也须说明数学依据，不能机械固化生产阈值。

比较通过不能来自删除原 case、放宽门限或替换 oracle。已有生产失败独立记录和定位；
测试整理不授权 kernel 调优、新生产几何或模型接入。

## Graph、存储和已确认缺口

每个 Graph invocation 捕获一次公开调用，验证原输入、同输入重复 replay 和声明的
changed-input 阶段。每阶段使用其实际输入计算期望，保持捕获地址稳定；不能只验最后
一次 replay。LinearAdd 的原输入、零输入、恢复输入三阶段显式声明，阶段开始前复位
初始 residual。SwiGLU 取反输入后重算 SiLU(-gate)*(-up)，不直接取反原输出。

纯输出按当前有效大小 poison 并检查 non-finite/guard；residual 是输入兼输出，不能
poison。只读 payload 和 guard 在被测试代码承诺的边界检查，Graph 下一次上传不得掩盖
上阶段破坏；大 fixture 可按组复用检查。workspace 保留区间容量、实际 peak、释放和
外层 arena 约束。新增完整扫描、guard 或 preservation 检查须单列成本。

以下是仍需落实的已确认缺口，不扩展成无依据的全组合测试：

- Add、SwiGLU 的 convenience 与 policy-bearing 重载都保留直接数学证据，包括 permissive
  policy 下合法 A16；NVFP4 SwiGLU convenience 保留 T=16 合法与 T=17 拒绝，较大合法
  extent 经 AllowA4 验证。
- Pair 验证公开 workspace query 的零容量和非法区间；两种存储关系各有代表性
  changed-input Graph。两个输出独立验收。
- FP8/NVFP4 Add 补齐 sampled 位置之外的输出有限值检查，并逐阶段验证已有 Graph。
- TopK、动态卷积各补代表性 changed-input replay；保留动态卷积 batch 起点无前一列贡献。

## 入口、构建与验收

各入口默认运行完整 suite，提供 `--help`、`--list`、`--shape N K` 和 `--case CASE_ID`。
列举与筛选在 CUDA 初始化和大 fixture 分配前完成；未知/非法参数、重复 ID、空匹配失败。
通过返回 0，比较失败或异常返回 1，无可用 CUDA 的运行请求由 main 返回 77；suite 内部
只返回失败计数或抛异常。保留 `NINFER_OP_REPORT_STATS`，稳定标签区分 recipe、seed、
policy、重载、采样、Graph phase 与输出；日志开关不改变判定。

每个共享非 header-only 实现只编译一次，reference/oracle 所属 target 直接应用
`ninfer_op_oracle_options()`。只有含测试 device 代码的翻译单元使用 `.cu`。
显式 CMake/CTest 注册随入口切换，完整移除被替代的源码、别名、转发和注册。

`input_projection_test_common.h` 的真实调用者同步迁移：Attention input projection、
GDN projection/snapshot/record、GDN Replay fold 与动态卷积融合。投影仍对数学 oracle，
record/fold 仍对同一物理 block 的指定 snapshot 精确 reference。`direct_bf16_weight.h`
和 `quantized_weight.h` 继续拥有权重 fixture 与独立解码，避免复制大型 BF16 payload；
仅当接口变化实际影响其他 Op 时才扩大回归范围。

验收先对照原 case 身份核对覆盖、表示边界、criterion、各 Graph 阶段和有效写入区域，
再运行受影响 suite 与调用者。使用相同机器、构建类型、oracle worker 策略和串行 GPU
条件；按原 family/format 聚合耗时，报告最慢 suite、资源变化及全部回退，并区分迁移和
新增检查的成本。没有证据的编译/内存/运行收益不作结论，不扩成 kernel benchmark campaign。
最终报告明确已完成、未完成、未验证和原有失败；Tests 只在实现切换后更新为新命令。
