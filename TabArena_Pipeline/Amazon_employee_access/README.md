# Amazon employee access — 最终 Pipeline

## 运作机制

数据表包含 9 个高基数（high-cardinality）类别型 ID 列和一个二分类目标变量；评估指标为 `1 - ROC-AUC`（越低越好）。TabFM 在对 context 样本行进行 attention 时会自行汇聚标签信息，因此没有向其传入任何由标签衍生出的特征——实验表明 out-of-fold（袋外/折外）目标编码正例率反而会显著降低其性能。

模型自身无法恢复的是无标签的共现结构（label-free co-occurrence structure），因此 `engineer` 在原始 ID 的基础上，额外提供了基于训练集与测试集联合计算的三个特征块：
* 单特征列取值的对数频数（log count）；
* 任意两列取值组合的双特征对数频数；
* 对于每一对有序特征列，第一列的每个取值所关联的第二列相异取值数量（出度/关联度数）。

近似重复列和常数列会被剔除（由于角色层级结构是嵌套的，机械式的特征网格交叉会产生完全相同的副本）。更宽的特征块——如三列共现频数（triple counts）、双列 PMI（点互信息）、共现矩阵 SVD、分组均值等——均进行过尝试，但每一项都会导致精度下降，因此直接不予生成，而非在后续进行剪枝。

两个仅在当前 fold 训练样本上拟合的辅助侧模型（side models），用于覆盖 TabFM 最薄弱的那部分假设空间（hypothesis space）：
1. **稀疏 One-Hot Logistic Regression**：在单列/双列/三列键值组合上构建（仅出现一次的值会共享该组合下的“rare/罕见”槽位），在 5 种不同正则化强度下求平均，为每个类别取值提供显式的权重。
2. **梯度提升树（Gradient Boosting）**：在 5 种（迭代次数、学习率、叶子数）配置下取平均，在相同的稠密频数特征块上拟合与坐标轴对齐的切分（axis-aligned splits），并以 Stacking 方式将线性模型的 out-of-fold logit 作为输入特征——其测试端特征列由训练端相同的内部折（inner folds）生成，从而确保该特征在两端的含义一致。

`engineer` 将两者的测试端预测值写入 `.aux_pred.npz`；`postprocess` 随后将其读取出来（各 hook 在独立进程中运行），并在标准化 logit 空间中以 TabFM 0.50 / 线性模型 0.24 / 提升树 0.26 的权重进行融合（blend）。TabFM 依然是占比最大的单个组件，融合后的预测排序与 TabFM 自身排序的相关性仍达到约 0.95。在 3 折 CV 上的验证结果为：融合模型为 0.1260，而仅基于这些特征的纯 TabFM 为 0.1318。

## build.sh

无。`final/build.sh` 仅打印提示并退出（exit 0）。该 pipeline 不读取任何预计算缓存——所有特征和两个辅助侧模型均在 `engineer` 内部由 judge 传入的 DataFrame 推导生成。运行期间未生成任何手工构建的缓存，因此不存在 `final/build/` 目录。零耗时，零磁盘占用。

## 读者可能产生的误区

- 工作区根目录下的 `.aux_pred.npz` **并非**先决依赖。它是由 `engineer` 写入并在同一次运行中由 `postprocess` 读取的，纯粹是为了跨越各 hook 之间的进程边界。切勿预先构建该文件；过期的副本会被 `postprocess` 中的行数/行 ID 校验机制拦截并丢弃，随后直接返回未融合的 TabFM 预测值（虽然静默降级好过产生错误，但这并非最终评分时的配置）。
- 由于该校验机制以及 `_aux` 中的单类别提前返回逻辑，运行 `python judge.py check` 以及任何针对小切片数据样本的测试运行，都会合情合理地产生仅包含 TabFM 的预测结果。这是符合预期的正常现象，而非 hook 发生故障。
- 频数和关联度数（degrees）是在训练集**和**测试集样本行上联合计算的。这种做法属于直推式（transductive），但由于完全无标签（label-free），因此不会产生跨越数据集划分的目标变量信息泄露。
- 运行耗时主要集中在辅助侧模型上：在超宽稀疏矩阵上进行 5 次 liblinear 拟合加上 3 次 stacking 拟合，以及 5 次 `HistGradientBoostingClassifier` 拟合，这些全部在 `engineer` 内部且在 TabFM 开始运行前完成。在全量训练集上，预计这部分会占去非 GPU 耗时的大部分。
- `_frame` 刻意将列索引强制转换为 `object` 类型：judge 宿主环境在没有安装 pyarrow 的情况下对 hook 输出进行反序列化（unpickle），此时 pandas 3 基于 arrow 后端的字符串索引将无法被加载。切勿将其擅自“简化”掉。