# Expert 权重来源与 GPU 副本缓存

本文约定已实现的 bank 视图、放置加载、GPU 副本容器与 MoE 计算交接。
数学与量化布局由 [artifact 格式](artifact-container.md)和各 Op 合同拥有；
执行与生命周期由 [Engine 架构](engine-architecture.md)拥有。

当前公开 Engine 仍使用全 GPU 权重。这里提供后续 CPU/GPU MoE 接入所需的底层能力，
GPU 路由、专家子集计算与完整融合计算已有独立 Op；CPU expert 算子、混合 Program 调度
和 Engine 显存预算选项尚未接入。

## 来源与身份

`core/weight_bank.h` 的 `WeightBankLayout` 描述一个 affine 矩阵集合，包含本地 N/K、
矩阵数量，以及 codes/high/scales 各 plane 的 parent 偏移、单矩阵字节数和矩阵步长。
`WeightBankRegion` 显式指定展平 parent 上的起始行、矩阵行数、数量和行步长。
几何来自 `WeightGeometry`；bank 不另行定义 codec。

`WeightBankView` 借用 Model 的不可变 `WeightParent`。`matrix_weight` 可以取得一个
原生矩阵视图，包括每次 Use 独立传入的 activation divisor。矩阵字节数与步长允许不同：
例如 GU bank 中所有 gate 构成的视图，矩阵行数是 I、步长是 2I。

- Groupwise INT 使用已有 rank-2 parent `[E*N,K]`，E/N 由模型 Binding 明确给出。
- NVFP4 使用 rank-3 `[E,N,K]`，codes、swizzled G16 scales、FP32 divisors 分别成 plane。
  切片保持 128 行边界，每个本地矩阵位于同一 divisor 域。
- Block-FP8 使用 rank-2/3 codes 和 BF16 `[E,N/128,K/128]` scales；不含 matrix divisor。
- Paged NF4 由其按行读取合同管理，不表示成 affine expert bank。

`ExpertBankSource` 将同一 logical expert 的 GU/down 两个 bank 组合，并为 Host 来源提供
对应 parent 的 `CudaHostView`。CPU 使用数学视图的 Host 地址，上传使用注册后的传输视图。
`ExpertKey={ExpertBankId,logical_expert}` 不随缓存分配变化。
Qwen3.5 的 bank ID 按 Main 层顺序建立，选中的 MTP bank 追加在末尾。
模型 loader 逐个核对 gate/up/down Binding 的 parent、起点和形状，再建立 bank；
不从物理对象 ID 或参数名称猜测编码。

## 逻辑加载与放置

加载顺序是 `plan_load → resolve_load → materialize_model`：

1. `Binder` 检查逻辑引用、Use 和所选资源；辅助值按对应元素区间取得 owning Host 值。
   `PreparedResources` 持有已读资源，移动到最终 backing 后保留 frontend 借用的地址。
2. `LoadPlan` 保留配置、逻辑引用、Use 和 expert source 描述。
   `resolve_load` 指定目标 GPU，以及 Main experts 使用 GPU 或 CPU 来源。
   MTP、router、shared expert 和其余固定权重仍在 GPU；放置沿显式 WeightId 展开。
3. `plan_materialization` 按 `(physical object, Device)` 去重，并用 `LayoutBuilder`
   分别规划 Host/Device backing。Host 的 `CpuOnly` 与 `CudaTransfer` 需求合并为一份来源；
   注册对象集中在 Host backing 的对齐前缀，普通 CPU 对象使用其后的独立区域。
   同一 parent 同时需要 CPU/GPU 两份 backing 时，两份容量均计入。
4. Materializer 为 CPU tensor 分配可缓存的 `HostBuffer`，为 GPU tensor 分配
   `DeviceBuffer`。有界 direct-I/O 块一次读取后可同时分发到 CPU 和 GPU。
   `CudaTransfer` 区域加载后由 `CudaHostRegistration` 长期注册，仍是 CPU 内存。
   Model 持有 allocation 与 registration；Program 结束使用后，先注销，再释放 Host backing。
   仅有 `CpuOnly` 需求的加载不创建 CUDA stream、event 或 pinned buffer。
5. `bind_view(reference, backing, device)` 明确选择实际位置。
   Model 持有最终 backing，所有原生权重与 bank 视图借用它，不依赖 Reader 生命周期。

这里保留完整 parent 原编码，不运行期 repack，也不把 Host 来源保存成大量单 expert 对象。
`Parameters` 保存 bank 身份、矩阵 profile 与 Use 元数据，可以描述 CPU 来源；
resident FFN 的 workspace/执行边界要求 CUDA 常驻视图，尚不执行混合放置。
Flash Next 的 NVFP4/Block-FP8 bank 可通过通用 artifact/core 接口加载；
这不代表已接入完整 Qwen4Exp 模型执行。

## 副本容量与 GPU 寻址

`describe_expert_cache` 先返回无指针的布局及容量：数据池 C、各 bank entry 布局、
Host 目录条目数、GPU offset/divisor 表和 pinned 索引快照容量。
`ExpertCache` 随 Program 建立；调用者显式提供 upload execution view，并在 acquire/release
提供 consumer execution view。Model 来源、DeviceContext 和被借用的 streams 必须比缓存活得更久。
全 resident 来源要求 C=0，不创建动态 offset/divisor/上传槽存储。
注册来源的 Host 容量由 Model 计费；注册字节数是其中的子集，不再加一份权重容量。

CPU 来源的每个 entry 是完整 expert 的 GU/down 数据块，只包含 codes/high/scales。
各 plane 至少 16-byte 对齐，entry 按 256-byte 对齐；不同 bank 使用自己的 entry 大小。
所有 bank 共用一个 byte pool，不永久分层，也不把小 expert 填充成最大 expert。
entry 不是序列化 parent，不追加虚假的 divisor trailer。

NVFP4 的 matrix divisor 是静态元数据：CPU 来源在启动时镜像到 GPU，常驻 GPU 来源
直接引用原 divisor plane。Host 元数据与 payload 的 divisor 必须表示同一组值，
由来源建立者保证；artifact materializer 从原 payload 提取并验证。
运行期上传只搬 codes/high/scales，divisor 始终按 logical expert ID 寻址。

`device_weight_bank` / `device_expert_bank` 从原生 CUDA bank 建立借用视图，不分配或重排权重。
`ExpertBankDeviceView` 同时记录逻辑 expert 数和 CUDA device。
`ExpertMatrixDeviceView` 的寻址合同：

```text
resident: offsets == nullptr; plane_address = plane_base + expert * plane_stride
cached:   offsets != nullptr; plane_address = plane_base + offsets[expert]
divisor_address = divisor_base + expert * divisor_stride
```

缓存的 plane_base 是 pool_base 加该 bank 的 plane 偏移；`indexed_storage` 描述整个被借用的
pool 范围，用于检查输出和 scratch 不与权重重叠。offset table、pool 和 divisor
分配地址终身稳定，可供 Graph 捕获。`UINT64_MAX` 是未发布哨兵；消费者只可读取
acquire 成功所保护的 keys，不能把表里其他条目的内容当作可用性保证。

当前真实权重的编码数据量如下，均不包含分配器及执行 workspace：

| 来源 | 单 expert 动态数据 | 完整 bank 动态数据 |
|---|---:|---:|
| 35B Q4 GU + Q5 down，256 experts | 1,802,240 B | 461,373,440 B |
| 35B Q4 GU + Q6 down，256 experts | 1,933,312 B | 494,927,872 B |
| Flash Next NVFP4 GU + down，512 experts | 2,764,800 B | 1,415,577,600 B |

Flash Next Main 48 个 bank 的两组 divisor 共 196,608 B；offset 表同为 196,608 B。
其 MTP 原始 block-FP8 expert payload 为 2,516,889,600 B，按固定 GPU 来源计入容量。

## 准备、发布和消费

`prepare(required,prefetch)` 替换目标集合；重复 key 只计一次。
required 必须能放入 C。prefetch 是可放弃的建议，不自动决定 expert 由 GPU 计算。

`progress(ExpertUploadBudget)` 按本次 payload 字节数与批次数预算提交工作，不显式等待 GPU。
启动配置同时限制单批 payload、复制描述符数和在途批次数；零预算只检查完成状态。
返回值包含 required 是否已发布、是否等待消费者或上传槽，以及本次提交的批数和字节数。

权重直接从注册来源复制，不经过 CPU packing。较大的连续 bank 区间使用 pitched H2D，
小集合与离散区间使用批量 H2D；选择仅依赖实际地址和规模。传输描述符预分配并复用。
普通 Host 地址与 CUDA 映射地址分别保存，批量复制消费后者，CPU 数学继续消费前者。
一个 expert 可以跨批次；完整 payload 提交后才发布它的 offset。编码内 padding 保留原字节，
entry 对齐空隙不计入传输 payload。

上传槽仅持有描述符、dirty index 快照与完成事件。数据批次之后提交索引批次；
完成前不重写该槽的快照和描述符。Host 来源没有有效注册视图时，在建立缓存前拒绝。

发布顺序：

```text
encoded data DMA → changed offset entries DMA → slot completion event
    → acquire: consumer stream wait → kernels / Graph replay
    → release: consumer completion event → reuse / eviction
```

索引只更新已可安全修改的条目，可合并相邻 dirty 条目，不覆盖正在被其他 use 读取的表项。
每个已发布 entry 记录上传槽及发布代次；acquire 只等待所需代次，已回收的旧槽代次意味着
该次上传已完成。后续无关预取不会成为旧数据消费者的新依赖。

`acquire(keys,execution)` 在发布已经排入流后即可成功，不要求 Host 已观察到 DMA 完成。
未发布时返回空；耗尽启动时配置的 use 数量属于调用错误。
成功取得的每个 use 都须在同一 consumer stream 调用一次 `release`，包括 resident bank。
release 必须位于最后一次真实 kernel/Graph replay 之后；这两个交接操作在 capture 外执行。
use 完成事件 ready 前，所需 entry 保持 pin，不被覆盖。

回收按 LRU 选择不在 required、没有 pin 的 entry，并合并空闲区。
可选预取先确认回收后能形成足够的连续空间，不驱逐后仍无法装入，也不触发碎片重建。
required 总量能容纳但碎片阻塞时，等待旧使用者完成后清空可重建
副本，重新连续装入 required；来源不可变，所以无需 D2D 移动或写回。
CUDA 提交失败后缓存停止接受工作，销毁前排空已提交传输和消费者，再释放存储。

缓存不包含 CPU 线程池、不执行 MoE 数学，也不拥有 prefill/decode 策略。
上层通过 required/prefetch 和 progress 预算表达实际需求；已经提交的复制不能撤回。
注册及首次上传的准备成本在启动阶段处理。CPU 计算与 DMA 可以同时读取不可变来源，
但共享系统内存带宽；数据上传和小 bridge 传输的推进由后续 Program 接入决定。

## MoE 计算交接

数学、输入输出与 alias 合同由 [MoE Op](../../include/ninfer/ops/moe.h) 和
[SumContributions](../../include/ninfer/ops/sum_contributions.h) 拥有。

- `topk_softmax_route` 写出 logical expert IDs、原始 top-k softmax 系数和独立 shared sigmoid
  系数。路由张量为 I32/FP32 `[K,T]`，shared 系数为 FP32 `[T]`。
- `routed_swiglu` 消费同一次路由，可以计算 All、None、mask 的 1 侧或 0 侧；不重新归一化。
  mask 表示计算分工，offset table 表示物理位置。两者独立，未选 expert 的偏移和权重不被读取。
  可选 shared term 在 GPU 上计算一次，也适用于 None。
- 分项计算可写 FP32 `[H,T]`；最终计算可写 BF16，或在一次最终舍入前加上原 BF16 residual。
  `sum_contributions` 接受一到两份 FP32 分项与可选 BF16 residual，产生 BF16 输出。
- `topk_softmax_swiglu_moe` 定义完整 router、routed/shared SwiGLU 与可选 residual 的融合计算。
  它与独立 Op 共用 kernel 机制，保有自己的格式/shape 分派和调优策略；全 GPU Main/MTP
  通过这一入口执行。

当前 CUDA 数学支持 35B 的 Q4+Q5、Q4+Q6 和 Q8+Q8 expert profile，shared 为 Q8。
NVFP4/Block-FP8 的来源与寻址可描述，不表示这些 expert 数学已实现。
`CudaSwiGLUExperts` 借用 affine 或 indexed GPU bank；`CpuSwiGLUExperts` 借用完整原生 Host bank，
留给 CPU 数学消费者。activation divisor 属于 Use，matrix divisor 属于权重表示，二者不互相替代。
融合 GU/down 分别取相关 Use 的 activation policy 交集；NVFP4 A4 需要共同的正值 activation divisor。

Op 不分配长期资源、不上传权重，也不读取 Host 路由值。Program 提供 scratch 和执行 view，
负责发布、acquire/release 及 bridge 完成依赖。固定 profile 的 Graph 可以重放新的路由、mask 和
索引内容；更新必须发生在旧消费者结束之后。shared 始终驻留 GPU，不进入动态 expert 集合。

## 后续接入

后续计算与 Program 接入保留以下执行约定；缓存自身不实现这些策略：

- 在读取完整权重前决定 resident 或 Host 来源。完整 GPU 放置可行时，使用完整融合
  `topk_softmax_swiglu_moe`，不建立完整 Host expert 副本。Host 模式保留完整原编码来源，运行期不从
  磁盘补 expert，也不写回权重。
- 大块 prefill、Replay 和 CausalScoring 在实际 MoE 消费前取得整个 bank；容量至少能
  容纳最大的动态 bank。预取可以分批，计算不拆成部分 expert 子集。下一 bank 来自
  模型调用顺序；独立的 embedding/mixer 可与上传重叠。
- decode、verification 和 forced control 使用启动时确定的各层 GPU 集合 D，其余
  被路由选中的 expert 由 CPU 计算。两侧 assignment 互斥且完备，消费同一次路由及其
  原系数，不分别重新归一化。verify 的实际 T 可以大于 batch，不能把 CPU 数学限制为 T=1。
- 真正进入下一执行类别时才切换集合：完成旧消费者，补齐新 required，再开始消费。
  D 完整就绪后在连续 decode rounds 中保持布局；相同阶段的后续 chunk/round 复用已有
  副本。数据池不永久划出 decode 专区。首 token 发布不等待恢复 D，副本布局不进入
  StateImage，也不随 speculative rejection 回滚。
- CPU 使用已有固定 CpuExecutor；Program 预分配 activations、routes 和 partial buffers。
  CPU 分支产生 FP32 partial[H,T]；GPU 分支可同时计算自身 routed 子集和 shared，产生另一份
  FP32 partial[H,T]。随后在 GPU 合并两份分项与 residual，shared 只计一次。
  激活交接缓冲区的复用服从各自完成依赖，与权重上传槽分别计费。
- resident 保留完整 CUDA Graph；混合执行使用经过验证的 GPU 分段 Graph 或显式 eager
  路径，在启动时确定。所选 MTP 仍常驻 GPU，DFlash/DFlash2 的 Main verification 经过
  同一混合 FFN，draft dense 权重沿用现有路径。
- 总显存预算包含固定权重、Program 状态、KV、workspace、bridge、cache、静态表和
  Graph/运行时保留。先按最低 KV 判断 resident 是否可行；Host 模式在满足最低 KV 后
  分配 expert 容量，显式 KV 配置可以扩大上下文池。Host 来源及其注册容量单独计费。

CPU expert 数学、indexed GPU expert kernel、两侧交接和最终归约均需完成各自的数学与状态
验证，再开放公共 Engine 的预算与混合执行选项。CPU 计算与 DMA 的带宽竞争仍需实测。

测试与实测入口分别见 [Tests](../../tests/README.md) 和 [Benchmarks](../../bench/README.md)。
