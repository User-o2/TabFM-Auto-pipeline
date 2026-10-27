# airfoil_self_noise — TabFM Pipeline

## 运作机制

预测完全由冻结（frozen）、zero-shot 的 TabFM 完成，模型在训练集样本行上进行 attention；其外围的 pipeline 本质上完全是一个特征空间上的“坐标变换”。NASA 的 airfoil self-noise（翼型自噪声）数据集包含 106 种工况配置（弦长、自由流速度、攻角），每种配置都在 1/3 倍频程（1/3-octave）频带上进行了扫频测试；而决定声压级的物理机理是乘性的，且存在于对数尺度上。

因此，`engineer()` 将每行数据重写为 Brooks–Pope–Marcolini 半经验模型实际采用的变量：
* 对数频率（log frequency）、对数位移厚度（log displacement thickness）和对数弦长（log chord）；
* 两个斯特劳哈尔数（Strouhal numbers）`f·δ/U` 与 `f·c/U`；
* 基于弦长与厚度的雷诺数（Reynolds numbers）；
* 厚度弦长比；
* 承载模型整体声级缩放尺度的 `10·log10(δ) + 50·log10(M)` 幅值项。

除了被剔除的 `frequency`、`free-stream-velocity` 和 `chord-length` 外，其余原始列均予以保留——这三者的对数形式和无量纲形式以更有效的方式表达了相同信息，若同时保留两者只会稀释模型的 attention。目标变量（target）未作改动，完整的训练集划分作为单次抽取（single draw）输入给 TabFM，`TABFM_KWARGS` 为空（采用 judge 默认值），`postprocess()` 仅作透传。仅凭这一坐标变换就带来了显著收益：RMSE 从恒等 pipeline（不做变换）的 1.377 降至约 1.247。

该模块还保留了在搜索阶段被停用的备选机制——Rank-Gauss 目标变量变换、Bagged 或 Bootstrapped 采样抽取、置换（Permutation）抽取、针对每种工况配置的留一法邻近区间（leave-one-out neighbour brackets）、工况配置均值编码（mean-encoding）、列复制（column duplication）以及预测裁剪（prediction clipping）。在最终评分的 `OPT` 配置字典下，这些机制均未被启用；代码 docstring 中记录了舍弃各项的原因（值得注意的是：留一法频谱邻域特征的得分为 1.62，表现差得多——1/3 倍频程扫频粒度过粗，无法进行有效的跨频段插值）。

## build.sh 构建了什么

什么都没有。`pipeline.py` 没有指定任何文件名，除了 judge 传给其 hook 的 DataFrame 之外不读取任何内容，且运行过程未生成任何辅助脚本或缓存——因此 `final/build/` 目录为空。
`bash final/build.sh` 仅打印一行信息并退出（exit 0）。运行耗时：瞬时。磁盘占用：0。

## 读者可能产生的误区

- **数据并非来自 `data/`。** 这是一个 `tabarena:` 任务。`judge.load_task` 会读取位于 `~/.cache/tabarena/airfoil_self_noise/fold_0/{train,test}.csv` 的 fold 缓存（可通过 `TABARENA_DIR` 覆盖）；如果该文件不存在，则会抛出断言提示 "fold cache missing; run the tabarena downloader"。当前工作区中的 `data/train.csv` 和 `data/test.csv` 只是 judge 实体化提供给 Agent 查看的无标签副本，并非重新运行时所加载的数据。预先填充该缓存是唯一的先决条件，且属于本目录的上游依赖——`build.sh` 刻意没有去拉取它。
- **最终评分是 30-fold 的均值，而非本工作区里的 fold 0。** 最终评测是在已发布的 TabFM baseline 所用的全部 30 个 fold 上独立运行该 pipeline，并对 RMSE 取均值。运行期间执行的 `python judge.py val` 只是 fold 0 训练集划分**内部**的 3-fold CV，因此其指标（即上文提及的约 1.247）是搜索过程中的探索信号，并非最终报告的测试得分。
- **在该任务上运行 `python final/run.py` 无法正常跑完，但这并非 pipeline 的原因。** 在完成 TabFM 处理后，代码会调用 `pd.read_csv(T.competition.sample_submission)` 来对提交文件进行排序，而 `load_tabarena_task` 从未设置过 `T.competition`——因为 TabArena 的 fold 并不存在 sample submission。因此在该行代码处会抛出 `AttributeError`；实际上其正上方的 `T.submission(pred, T.X_test)` 已经成功生成了包含 `row_id` / `scaled-sound-pressure` 的 DataFrame。执行 `python final/run.py --check` 会在此之前终止，能够正常通过。这一结论是我通过阅读 `final/judge.py:266-296` 和 `final/run.py:119-124` 代码分析得出的，而非实际执行所得（当前会话没有 shell 环境）——因此请将其作为代码静态分析参考；同时注意 `run.py` 与 `judge.py` 并不在我允许修改的两个文件范围内，故保持了原样。
- **故意未定义 `folds()`**：一旦工况配置结构被显式编码进特征中，fold 0 训练集内部的样本行就满足了可交换性（exchangeable）。因此验证过程直接使用了 judge 自带的 `KFold(3, shuffle, seed 0)`，其 2/3 的训练集比例与 TabArena 最终各个 fold 所见的比例完全一致。