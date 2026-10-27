'''
典型案例：输入变换、目标变换和多视图集成
- TABFM_KWARGS：控制模型**内部的**归一化与 SVD 特征视图
- preprocess()：[目标视图]构造 raw=y 与 sqrt=√y 两列目标
- engineer()：对选中的非负、长尾特征做 log1p 变换
- sample()：[上下文视图]构造 4 种不同训练行顺序的上下文视图
- 预测：[内部成员预测视图]框架的 8 个集成成员，使用不同归一化与特征配置，分别预测 raw 与 sqrt（2目标×4上下文×8内部成员=64，这些组合共用冻结的预训练权重）
- postprocess()：平方还原、预测加权融合、范围裁剪
'''

import numpy as np
import pandas as pd

# 开启配置项
# 在 8 个模型成员（members）上使用 4 种归一化视图，而不是默认的 2 种；并且为其中索引为奇数的那一半成员加上 SVD 特征列。 SVD 特征列是对原始特征矩阵做奇异值分解（SVD）后得到的低维表示（本质上是降维）
# 这两者（不同归一化和 SVD 特征）都是为了增加多样性（多样性对于本任务而言效果显著）
# engineer() 没有显式新增列，但 TabFM 内部仍然可以通过配置追加 SVD 特征
TABFM_KWARGS = {
    "n_svd_features": "sqrt", # 应用 SVD 特征的成员数量，"sqrt"表示一半成员使用 SVD 特征
    "norm_methods": ["none", "power", "quantile", "robust"], # 4种归一化操作，改变数据的分布形态
}

# 上下界[0,200]
_Y_LO = 0.0         # critical_temp is a temperature in K: it cannot be negative
_Y_HI = 200.0

_LOG_X_SKEW = 2.5 # 对偏度大于此值的非负列进行 log1p 变换（基于 X_train 决定）
_ID_COLS = ("row_id",)
_W_SQRT, _W_CBRT = 0.17, 0.00 # 重塑目标值后拟合的权重；剩余权重分配给原始拟合
_PERM_SEED = 0 # 用于行置换（打乱）的随机种子
_DRAWS = 4 # 在整个训练集上使用不同行顺序进行拟合的抽取次数，最终由评测程序取平均(4 个 Context Views)


def sample(X_train, y_train, X_test, max_rows):
  '''
  返回的4个上下文包含相同样本，区别在行顺序：1份原顺序+3份随机排列。没有删行、重复采样或改变类别比例。框架应同时按索引排列特征和目标，保持样本对应关系。

  尝试利用上下文顺序带来的预测差异，再通过多视图组合获得更稳定的结果。但是 Write-up 说明，排列带来的改善约为 0.001，处于噪声范围内。因此，本操作不是已证明显著有效的核心贡献。
  '''
  n = len(X_train)
  rng = np.random.default_rng(_PERM_SEED)
  return [np.arange(n)] + [rng.permutation(n) for _ in range(_DRAWS - 1)]


def preprocess(X_train, y_train, X_test):
  '''
  把一个目标变成两种预测视图，返回的 y_train 是一个两列的 DataFrame: y 与 √y

  看到 y 有两列，就会在后台启动两个 TabFM 实例：用 raw 列作为 In-Context 标签，去拟合原空间；用 sqrt 列作为 In-Context 标签，去拟合平方根空间。两种表示可能产生不同的预测误差，后续便有机会互补（加权融合）。
  '''
  if _W_SQRT or _W_CBRT:
    # Extra target columns mean the judge fits one TabFM per column: further, partly decorrelated
    # fits of the same model on the same rows, which no TABFM_KWARGS can buy.
    y = np.maximum(np.asarray(y_train, dtype=np.float64).ravel(), 0.0)
    cols = {"raw": y}
    if _W_SQRT:
      cols["sqrt"] = np.sqrt(y)
    if _W_CBRT:
      cols["cbrt"] = np.cbrt(y)
    y_train = pd.DataFrame(cols, index=X_train.index)

  return X_train, y_train, X_test


def engineer(X_train, y_train, X_test):
  '''
  识别长尾列，再原地进行 log1p。

  它与 TABFM_KWARGS 也有分工：engineer() 先进行一次任务特定的输入处理；随后，不同 TabFM 成员再使用各自的内部归一化视图。
  '''
  # 有九列数据的数值跨度高达四个数量级，且几乎所有的数据质量都集中在接近零的区域
  # 对于那些使用 norm_methods='none'（即不做归一化）的模型成员来说，它们会看到一个极端的尖峰和一条长尾（随后会被 z-score 截断操作强行压平）；而 log1p 变换能让这些特征变得易于模型读取（平滑），同时不会改变它们原有的排序关系。
  Xtr, Xte = X_train.copy(), X_test.copy()
  for c in Xtr.columns:
    col = Xtr[c]
    # 跳过id列和非数值类型的列
    if c in _ID_COLS or not pd.api.types.is_numeric_dtype(col):
      continue
    # 只处理非负，且偏度极高的列
    if col.min() >= 0 and float(col.skew()) > _LOG_X_SKEW:
      # 核心变换：对训练集和测试集做 log1p 映射
      Xtr[c] = np.log1p(col)
      Xte[c] = np.log1p(Xte[c].clip(lower=0))
  return Xtr, Xte


def postprocess(pred, y_orig, X_test):
  '''
  对预测结果进行还原、加权融合和范围裁剪，返回最终的预测值。
  '''
  # 提取出真实的第一列（原始标签）作为 y
  y = y_orig.iloc[:, 0] if isinstance(y_orig, pd.DataFrame) else y_orig
  if _W_SQRT or _W_CBRT:
    out = (1.0 - _W_SQRT - _W_CBRT) * pred["raw"]
    if _W_SQRT:
      out = out + _W_SQRT * pred["sqrt"].clip(lower=0.0) ** 2
    if _W_CBRT:
      out = out + _W_CBRT * pred["cbrt"].clip(lower=0.0) ** 3
    pred = out

  lo, hi = max(_Y_LO, float(np.min(y))), min(_Y_HI, float(np.max(y))) # 取物理范围和实际范围的交集
  # 只有当待预测样本的真实目标确实落在裁剪区间内时，裁剪才不会增大该样本的平方误差。
  return pred.clip(lower=lo, upper=hi)
