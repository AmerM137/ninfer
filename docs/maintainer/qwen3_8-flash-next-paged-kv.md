# Qwen3.8-Flash-Next：共享 KV 页组扩展设计

更新日期：2026-10-08。**设计已确定，生产实现待完成。**

本文拥有 Qwen4Exp 所需的不同 plane 行密度扩展：`tokens_per_row`、Device/Host 布局、
容量、coverage 和页组迁移。已实现的通用存储合同仍由
[Paged KV Context Store](paged-kv-cache.md)维护；模型 inventory、数学和 consumer 接口由
[QSA](qwen3_8-flash-next-qsa.md)维护。整体进度见
[支持总览](qwen3_8-flash-next-model.md#11-支持进度与后续工作)。

阅读入口：[字段与布局](#2-geometry-与布局)、[有效范围](#3-coverage-与消费者)、
[迁移](#4-分配cow-与-offload)、[实施边界](#5-实现范围与验收)。

## 1. 既定选择

主 K/V 与 compressed index keys 使用同一逻辑页组、block table、physical page ID、epoch、
coverage、引用和迁移事务。一个页组覆盖 64 个原 token，各 plane 使用自己的行数，
放在独立稳定 slab 中。主 K/V 每页 64 行，index key 每页 16 行。

Main 的 12 个 QSA 层属于 Main pool；MTP 的一个 QSA 层属于独立 Backend pool。
两个 pool 的物理 ID 和前沿分别解释。不增加独立 index pool、第二张页表或 index cursor。
未完成四-token 块仍由 QSA StateImage 保存，不占一个提前有效的 compressed row。

## 2. Geometry 与布局

在现有 `src/core/paged_kv_cache.h` 的 `KVPlaneGeometry` 中增加一个明确字段：

```cpp
namespace ninfer {

struct KVPlaneGeometry {
    DType dtype = DType::BF16;
    std::int32_t leading_extent = 0;
    std::int32_t head_extent = 0;
    std::uint32_t tokens_per_row = 1;
    std::size_t alignment = 256;

    friend bool operator==(const KVPlaneGeometry&, const KVPlaneGeometry&) = default;
};

} // namespace ninfer
```

`tokens_per_row=s` 表示一条完整行覆盖的原 token 数。`KVPageGeometry.page_tokens=P`
仍是原 token 页跨度 64；P 在本存储公式中专指页跨度，不是展平的调用列数 M。
仅接纳以下闭合组合：

| Plane | s | 每页行数 P/s | 有效原 token 数为 c 时的有效行数 |
|---|---:|---:|---:|
| 既有 K/V data 与 scales | 1 | 64 | c |
| QSA compressed index K | 4 | 16 | floor(c/4) |

必须满足 P%s=0。本字段不是 feature 轴的量化 group size：同一 token 的 code/scale
planes 都使用 s=1。s=4 只用于 BF16、leading extent=128、head extent=1 的 page-major
index plane；不引入任意采样函数或未使用的 layout/profile 组合。

对 leading extent X、head extent H、物理页数 G：

```text
page-major: [X,P/s,H,G]
head-major: [X,P/s,G,H]       // 本扩展不注册 s=4 的 head-major

PlanePageBytes = sizeof(dtype) * X * H * (P/s)
PageBytes      = sum_plane(PlanePageBytes)
```

每行的 feature 维连续，地址使用真实 Tensor strides 和足够宽的整数。不同 planes
不要求物理相邻。Host packed offsets、Device slab span、对齐和 capacity query 都从同一
geometry 推导；不能继续把所有 planes 按 P 行计算，再只修改 consumer 的 view。

QSA index plane 的具体 shape 为 BF16 `[128,16,1,G]`，每层每页 4096 B。主 cache 的
code/scale dtypes 和 leading extents 沿用各 codec；完整实例容量表只维护在
[QSA 容量](qwen3_8-flash-next-qsa.md#55-容量)。

## 3. Coverage 与消费者

```text
逻辑页 l 覆盖原位置 [l*P,(l+1)*P)
前沿 F 对应的页内有效原 token 数 c = clamp(F-l*P,0,P)
plane 有效 rows                    = [0,floor(c/s))
plane 完整行 r 覆盖                = [l*P+r*s,l*P+(r+1)*s)
```

只有一份原 token coverage。准备阶段可以写候选后缀，但未提交或只完成一部分的行不能
成为可读的 committed row。复制整个页的无效后缀字节不扩大 coverage。
Indexer 的可见块数还受每个 query 的因果位置限制，不能直接使用 chunk 末端的 coverage。

主 cache 沿用 `PagedKVBatchLayerView`；index plane 由 QSA 的 `QsaIndexCacheView` 表达。
模型从同一 pool、同一 layer、同一个 block-table view 构造两者，wrapper 检查可见
geometry 与 table 的地址/shape/strides。这里只扩展 storage geometry，不重复声明 QSA
Op views，也不在 generic core 中解释 QSA norm、选块或 tail 数学。

## 4. 分配、COW 与 offload

分配、reservation、共享引用、COW、回收和迁移仍以完整页组为单位：

- GPU unit 前按全部允许写入的原 token 区间准备 COW；执行期间 mapping 固定。
- Device COW 和 Device↔Host 迁移覆盖所有 planes，按实际 bytes/offset 搬运原 words。
- epoch、protected coverage、replica completeness 均在同一个原 token 域解释。
- source lease 与 destination reservation 保留至复制完成；完整后才发布 destination。
- 模型 Snapshot 同时包含该前沿的 QSA raw tail 和其他 StateImage 字段；只有页组到达
  不能构成完整恢复点。

ResourceManager 消费 Program 查询出的页组/StateImage 成本，不另行计算压缩比或保存
index pages 的副账本。Op 不分配页、不触发 offload，也不等待尚未物化的 Device page。

## 5. 实现范围与验收

| 所属 | 必须同步完成的内容 |
|---|---|
| Core geometry/layout | 字段检查与 equality、Device shapes/strides/span、Host packed offsets/alignment |
| Core page/replica transfer | 全 plane COW、Host/Device 复制、payload bytes 与 transfer-work 计费 |
| Qwen4Exp planner/binder | Main/Backend plane inventory、QSA views、容量与 memory summary |
| Program 事务 | 原 token coverage、读写 reservation、StateImage 与页组的联合采用/恢复 |
| 长期合同 | 实现与验收完成后，将稳定字段和行为归入 `paged-kv-cache.md` |

验收覆盖 s=1 的既有 page-major/head-major 路线，以及新 s=4 index plane。检查精确
布局/容量、63/64/65 原 token 边界、四-token 边界、物理页 ID 重排、同页不同 coverage、
COW、共享 source、Host round-trip 和中止后的实际占用。QSA 数学另按专题资格化。

这项扩展是 QSA 集成的前置条件。仅增加一个 Tensor view 不代表分配、迁移或恢复已经支持。
实现完成后将长期内容迁入稳定合同并删除本计划，避免维护两份通用存储定义。
