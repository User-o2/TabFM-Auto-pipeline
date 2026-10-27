# superconductivity — TabFM Pipeline (RMSE，越低越好)

## Pipeline 的运作机制

冻结且 zero-shot 的 TabFM 通过对整个训练集表格进行 in-context learning（上下文学习）来预测 `critical_temp`（超导临界温度）；此处最终交付的所有工作，核心都在于让 8 个 ensemble 成员观察到**多大程度的多样性（variety）**，因为事实证明这是该任务上唯一能显著提升得分的杠杆。没有对任何行进行丢弃、重采样、去重或合成，也没有增加、筛选或旋转（rotated，如 PCA 等变换）任何列——pipeline 的 docstring 中记录了一长串消融实验（ablations），显示这些操作每一项都会导致 0.03 到 0.28 的 RMSE 性能损失。

最终交付了四项改动：

- **四种归一化视角（normalisation views）**：在 8 个成员之间分配四种归一化方式（`none`、`power`、`quantile`、`robust`），而非 TabFM 默认的两种；并通过 `n_svd_features="sqrt"` 为奇数索引的一半成员添加 SVD 特征列——即添加至 `power` 和 `robust` 视角中，该组合在 6 组对比实验中胜出。
- **对 skew > 2.5 的非负特征列进行 `log1p` 变换**（全部 81 列中的 9 列——主要是 FusionHeat 相关的统计量），这些列的数值跨越了 4 个数量级，且绝大部分数值都集中在 0 附近。这主要改善了使用 `none` 归一化的成员的表现，否则它们会遇到尖峰加长尾的分布，进而被 z-score 裁剪（clipping）压平。
- **样本行排序抽取（Row-order draws）。** TabFM 对上下文样本行**并不具备**置换不变性（permutation-invariant），因此该 pipeline 向 judge 提供了完整训练集的 4 种索引排序（原始顺序加上 3 组带随机种子的排列）。这些抽取（draws）所见的数据行与列完全相同，因此对其进行平均不会损失任何信息，也不会增加期望误差。实测增益约为 0.001——处于噪声范围内；最终选择交付它是基于理论逻辑，而非实测指标。
- **第二目标参数化（A second target parameterisation）。** judge 会对 `y_train` 的每一列分别拟合一个 TabFM，因此 `preprocess` 传入了 `[y, sqrt(y)]`；`postprocess` 将第二个预测值重新平方还原，以 `0.17` 的权重进行融合（blend），随后裁剪（clip）到训练集目标变量的数值区间内。两次拟合的相关系数为 0.989，低于保本阈值（break-even）`sigma_raw/sigma_sqrt = 0.993`，因此赋予正权重在先验上是合理的；实测显示能带来约 0.003 的收益。

评分结果：**在 9 个 TabArena fold 上的平均 RMSE 为 8.7946**，相比之下，基础 TabFM baseline 为 8.9517，ensemble baseline 为 8.9176（搜索阶段的 CV 得分为 9.35488）。

## build.sh 构建了什么

什么都没有。`pipeline.py` 不读取任何文件——它仅对 judge 传递给其 4 个 hook 的 DataFrame 进行变换——因此 `final/build.sh` 仅包含一行注释和 `exit 0`。其运行耗时不到一秒，且不需要磁盘空间。`final/build/` 目录为空也是同样的原因：运行过程没有写入任何辅助脚本，每次实验都是直接修改 `pipeline.py` 并运行 `python judge.py val`。

```bash
bash final/build.sh && python final/run.py     # 生成 final/submission.csv
```

## 读者可能产生的误区

- **`_DRAWS = 4`，而非 8。** docstring 中关于行序平均的长篇论证是在 K=8 作为当前配置时编写的；第 83 次评测（eval 83）验证了 K=4 是否能以一半的算力换取相同的结果，最终在测试划分上评分采用的也是 4。K 值的收益曲线主要被噪声主导（在 K=1, 3, 8, 16 时得分在 9.3572 到 9.3598 之间波动），因此这是一个运行耗时层面的取舍，而非性能退化。
- **`_W_CBRT = 0.00`。** 立方根（cbrt）目标列的代码逻辑已经接入但被停用了——第 79 至 81 次评测发现它与 sqrt 的共线性过强，无法提供有效的第三个表征方向。将其设为非零值还会额外增加一次完整的 TabFM 拟合开销。
- **没有设置 `max_num_rows`。** 这是有意为之：设置行数上限意味着在 3-fold CV 中使用 9450 行上下文，但在最终评测中会使用 14175 行，即两处使用的数据比例不同。在 judge 默认的 65536 行上限下，该表格规模足够小，完全可以直接全量使用。
- **运行耗时主要集中在预测阶段，而非拟合阶段。** 9 个 fold 的最终评分耗时约 37 分钟：其中拟合仅约 167 秒，而预测耗时约 2014 秒，因为 4 次抽取 × 8 个成员针对每个测试分块（chunk）都需要在全量上下文上重新进行 attention。将 `_DRAWS` 减半大致能将实际运行时间（wall clock）缩短一半。
- **docstring 消融实验列表中引用的数字均为 CV 得分**，来自对 fold 0 训练集划分进行 3-fold 搜索的结果，处于约 9.36 的基准水平。它们不能与前文给出的 8.79 测试指标直接比较，后者是在训练样本更多的 9 个 TabArena fold 上的平均值。