# TabFM-Auto：TabArena 最终 Pipeline 代码与 Write-up

本目录收录指定 Explorer 网页中的全部 **51 个数据集**：13 个回归、30 个二分类、8 个多分类。

来源：[Web Report](https://deqingfu.github.io/tabfm-auto/) | [Pipeline Explorer](https://deqingfu.github.io/tabfm-auto/explorer.html)。

每个数据集目录使用 Explorer 的完整 `name`，大小写、连字符和下划线均保留。论文中的简写可能与此不同，例如 `concrete_strength` 对应网页中的 `concrete_compressive_strength`。

- `pipeline.py`：Code → All code 的完整源码（网页 `code` 字段）。
- `README.md`：Write-up → README.md 的原始 Markdown（网页 `readme` 字段）。

所有原文均以 UTF-8 无损保存，未添加标题、行号或中文注释，未调整格式、辅助函数、导入语句或已有注释。

## 论文与代码的对应关系

已阅读本地 `paper_pdf/TabFM-Auto.pdf` 的正文及相关附录。该工作固定 TabFM 的模型权重，由编码智能体结合数据集元信息与训练集内部三折交叉验证，搜索数据处理流水线。四个主要接口分别是 `preprocess`（清洗与目标变换）、`engineer`（特征工程）、`sample`（上下文行选择）、`postprocess`（预测校准与逆变换）；`TABFM_KWARGS` 用于配置基础模型的推理预设。

论文在每个数据集的 fold 0 训练集上搜索一次，再固定所选代码用于各官方划分。本目录保存网页当前提供的每个数据集的一份最终代码和对应说明。

## 完整性核验

已确认 51 个目录、51 份完整源码、51 份非空 Write-up；102 个文件逐字节匹配网页原始字段，全部 Python 源码通过语法解析。未执行数据集流水线或模型推理。

`manifest.json` 保存每份文件的绝对路径、字节数、SHA-256、原网页定位链接，以及网页快照的 SHA-256 和保存时间。

网页快照与提取脚本保存在工作区 `.tabfm_extract/` 下。可在工作区根目录激活 `tabular` 环境后，运行 `python .tabfm_extract/extract_pipelines.py` 重复核验；已有内容不同时会拒绝覆盖。

## 典型案例

建议先读这 **4 个案例，按下表顺序学习**。我逐份检查了 51 份源码及对应说明，选择时主要考虑：**机制有代表性、四例之间互补、实际启用的代码容易辨认**，没有按性能提升幅度排序。

| 阅读顺序 | 案例 | 代码行数 | 主要学习内容 |
|---|---|---:|---|
| 1 | **airfoil_self_noise** | 196，但有效路径较短 | 把领域知识写成适合 TabFM 的输入表示 |
| 2 | **superconductivity** | 67 | 目标变换、上下文视图与预测融合 |
| 3 | **kddcup09_appetency** | 153 | 匿名表格上的统计编码、交叉拟合与特征筛选 |
| 4 | **Amazon_employee_access** | 218 | 关系图统计，以及完整实现中的辅助模型融合 |

这个组合覆盖了论文强调的领域特征、统计特征和上下文控制，也补充了目标处理与真实代码中的模型组合。[Web Report 的机制分类](https://deqingfu.github.io/tabfm-auto/)可以用作对照，但具体执行行为应以本地源码为准。

**1. airfoil_self_noise：最适合入门，理解“为什么 LLM 的领域知识能帮助冻结 TabFM”。**

阅读：[pipeline.py](./airfoil_self_noise/pipeline.py) | [README.md](./airfoil_self_noise/README.md)

原始表记录频率、气流速度、弦长、攻角和边界层厚度。智能体没有让 TabFM 自己从这些数值中摸索全部物理关系，而是显式构造：

- Strouhal 数，例如 $f\delta/U$；
- Reynolds 数，例如 $Uc/\nu$；
- 对数频率、对数厚度、厚弦比和声学幅值缩放项。

同时，它删除部分已被新表示替代的原始列。这体现的搜索特点是：**不仅增加特征，还会改变坐标系、去掉冗余表示，让冻结模型更容易利用已有知识。**

推荐阅读顺序：第 4 行 `OPT` → 第 12 行 `_phys()` → 第 151 行 `engineer()`。

特别注意：文件里保留了邻域插值、目标变换、bagging 等候选实现，但**最终配置没有启用它们**。本例最终目标不变、使用单份完整上下文、后处理直接返回预测。它很适合训练“先看配置，再判断实际执行路径”的读代码习惯。

**2. superconductivity：篇幅最短，却能看到四个 hook 如何配合。**

阅读：[pipeline.py](./superconductivity/pipeline.py) | [README.md](./superconductivity/README.md)

这个案例的重点是**给同一个冻结模型提供不同视图**：

- 第 26 行 `preprocess()` 构造 `raw=y` 与 `sqrt=√y` 两列目标；按代码注释和 Write-up 所述，由评估框架分别调用 TabFM。
- 第 56 行 `postprocess()` 将平方根目标的预测平方还原，再按 **0.83 / 0.17** 融合。
- 第 20 行 `sample()` 返回四份上下文顺序：一份原顺序、三份随机排列。**四份包含相同的全部训练行，并非四个子样本。**
- 第 40 行 `engineer()` 根据训练数据的偏度，选择部分非负数值列进行 `log1p` 变换。

它能帮助你理解：**搜索对象还包括“让模型预测什么尺度的目标”和“给模型怎样的上下文”，并不局限于造特征。**

建议先连着读 `preprocess → postprocess`，再看 `sample → engineer → TABFM_KWARGS`。立方根目标分支的权重为零，当前未启用；也不要把 Write-up 对排列集成的解释理解为在其他数据集上必然有效。

**3. kddcup09_appetency：最值得认真学习的统计特征工程案例。**

阅读：[pipeline.py](./kddcup09_appetency/pipeline.py) | [README.md](./kddcup09_appetency/README.md)

这个数据集包含匿名字段、高基数字符串和大量缺失值，难以直接套用物理公式。智能体转而处理**类别表示、训练与查询侧的一致性，以及哪些列值得保留**。

最值得看的三处是：

- 第 37 行 `_cat_features()`：为类别构造频次和目标均值编码。训练行按五个块交叉拟合，类别目标统计排除所在块；训练侧频次也排除该块，再乘以 \(5/4\)，使其尺度接近查询侧使用的全训练集频次。
- 第 77 行 `_uni_score()`：分别把缺失值放在数值序列最低端和最高端计算 AUC，从而保留“缺失本身有预测价值”的列。
- 第 90 行 `_select()`：只保留排名靠前的 200 列，而不是把全部生成特征交给 TabFM。

它与 Amazon 案例的区别很有教学价值：**这里主要使用当前训练划分内的监督统计编码；Amazon 的图统计部分则使用训练与测试特征联合计算的无标签关系。**

本例没有辅助预测器的输出融合，最终预测仍由 TabFM 完成。阅读时重点追踪：**训练行的编码怎么产生，测试新类别怎么办，目标信息在哪些地方被使用。**

**4. Amazon_employee_access：论文中的经典案例，但完整代码比图示丰富。**

阅读：[pipeline.py](./Amazon_employee_access/pipeline.py) | [README.md](./Amazon_employee_access/README.md)

第一层是论文最突出的关系特征：经理、角色、部门、资源等 ID 本身的数值大小没有意义，但它们的**共现结构**有意义。

第 167 行 `engineer()` 联合训练与测试特征，计算单列频数、二列组合频数，以及“每位经理连接多少种资源”这样的不同邻居数，再去掉常数列和高度重复的列。这部分不使用测试标签。

第二层是必须读完整代码才能发现的内容：

- 第 85 行 `_aux()` 实际训练了逻辑回归和 `HistGradientBoostingClassifier`，还包含 OOF stacking。
- 第 187 行确实调用 `_aux()`，它不是停用的候选分支。
- 第 192 行 `postprocess()` 在标准化 logit 空间，以 **0.50 TabFM、0.24 逻辑回归、0.26 boosting** 的权重融合。

因此，本例既代表**从 ID 表挖掘关系结构**，也说明：**TabFM 权重冻结，并不意味着这份最终流水线完全没有拟合其他模型**。不能把全部收益都解释为图特征带来的收益。

建议分两遍读：先读 `engineer()` 及其统计辅助函数；再读 `_aux()` 和 `postprocess()`。跨 hook 传递预测的 `.aux_pred.npz` 是同次运行生成的临时状态。

---

学习这四份时，始终区分三件事：**最终真正启用的操作、文件中残留但关闭的候选分支、README 回述的历史实验**。这些文件是搜索后的产物；仅凭最终源码，无法完整还原智能体的整个搜索过程。