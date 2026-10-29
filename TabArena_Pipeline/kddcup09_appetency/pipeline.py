'''
典型案例：根据当前训练数据，把高基数类别转换成统计特征，再挑出最有用的列交给冻结的 TabFM。主要解决匿名字段、类别编码、缺失信息和特征筛选问题。
（高基数类别：比如“用户ID列”可能有 10 万个不同的用户，基数 = 100000）这些字段大多匿名，难以从列名推导物理公式。因此，代码从数据本身寻找统计结构。

- TABFM_KWARGS：配置四种内部归一化视图
- preprocess()：不做数据处理，只在必要时调整训练目标的数据类型，方便跨进程传递
- engineer()：[核心：特征工程]分为数值列和特征列，数值列简单处理，特征列通过构造类别频次编码和目标编码。然后一共筛选最多 200 个预测特征返回
- sample()：保持默认
- postprocess()：原样返回 TabFM 预测
'''
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# TabFM 的配置参数字典
# 只有“归一化”一项配置，让内部集成成员从不同数值表示观察同一份特征表
# 没有显式设置 SVD、额外交叉特征或上下文行数；其余设置由运行框架的默认值决定
TABFM_KWARGS = {
  'norm_methods': ['none', 'power', 'quantile', 'robust'] # 四种不同的归一化方法
  }

# 常量，由 engineer() 及其辅助函数使用
KEEP_STR_MAX = 30 # 类别数不超过 30 时，把原始类别列也加入候选特征
TE_FOLDS = 5 # 将训练行分成五块，交叉拟合类别编码
TE_SMOOTH = 60.0 # 目标编码向总体正类比例收缩的强度
N_KEEP = 200 # 只保留前 200 个 AUC 排名靠前的列
MIN_LEVEL = 10 # 某个类别值出现不足 10 次时，其目标编码回退到总体正类比例


def _as_str(s):
  """plain object-dtype strings: an arrow-backed str column does not survive the hook pickle."""
  v = s.astype(object).to_numpy(dtype=object).copy()
  v[pd.isna(v)] = "__NA__"
  return pd.Series(v, dtype=object, index=s.index)


def preprocess(X_train, y_train, X_test):
  '''
  没有进行数据处理
  只是解决某些字符串类型跨进程序列化的问题
  '''
  if not isinstance(y_train, pd.DataFrame) and y_train.dtype.kind not in "fiuO":
    y_train = y_train.astype(object)
  return X_train, y_train, X_test


def _codes(st, sv):
  """integer codes for a categorical column; query values unseen in training get -1."""
  ct, uniq = pd.factorize(st, use_na_sentinel=False)
  lut = pd.Series(np.arange(len(uniq)), index=uniq)
  cv = sv.map(lut).to_numpy()
  cv = np.where(pd.isna(cv), -1, cv).astype(np.int64)
  return ct.astype(np.int64), cv, len(uniq)


def _cat_features(ct, cv, n, yb, rng):
  """count + cross-fitted target encoding for one categorical column."""
  cnt = np.bincount(ct, minlength=n).astype(np.float64)
  pos = np.bincount(ct, weights=yb, minlength=n)
  prior = yb.mean()

  known = cnt >= MIN_LEVEL

  # cross-fitted on the context rows, full-fit map on the query rows
  te_t, cnt_t = np.empty(len(ct)), np.empty(len(ct))
  order = rng.permutation(len(ct))
  scale = TE_FOLDS / (TE_FOLDS - 1.0)
  for p in np.array_split(order, TE_FOLDS):
    c_out = np.bincount(ct[p], minlength=n).astype(np.float64)
    p_out = np.bincount(ct[p], weights=yb[p], minlength=n)
    c_in, p_in = cnt - c_out, pos - p_out
    m = np.where(known, (p_in + prior * TE_SMOOTH) / (c_in + TE_SMOOTH), prior)
    te_t[p] = m[ct[p]]
    cnt_t[p] = c_in[ct[p]] * scale
  m_full = np.where(known, (pos + prior * TE_SMOOTH) / (cnt + TE_SMOOTH), prior)
  te_v = np.where(cv >= 0, m_full[np.clip(cv, 0, None)], prior)
  cnt_v = np.where(cv >= 0, cnt[np.clip(cv, 0, None)], 0.0)
  return cnt_t, cnt_v, te_t, te_v

# 此函数没有被使用
def _pair_codes(at, av, bt, bv, nb):
  """codes of the (a, b) pair, renumbered densely; a query row with either side unseen gets -1."""
  kt, uniq = pd.factorize(at * nb + bt, use_na_sentinel=False)
  lut = pd.Series(np.arange(len(uniq)), index=uniq)
  raw = np.where((av < 0) | (bv < 0), -1, av * nb + bv)
  kv = pd.Series(raw).map(lut).to_numpy()
  kv = np.where(pd.isna(kv), -1, kv).astype(np.int64)
  return kt.astype(np.int64), kv, len(uniq)


def _auc(v, yb, npos, nneg):
  r = rankdata(v)
  return (r[yb > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def _uni_score(v, yb, npos, nneg):
  """|AUC-0.5| of one column, taking the better of ranking NaN lowest / highest, so that a column
  whose signal lives in its missingness pattern scores as high as one whose signal is in its values."""
  if v.dtype.kind not in "fiu":
    return 0.0
  m = np.isnan(v)
  if not m.any():
    return abs(_auc(v, yb, npos, nneg) - 0.5)
  lo = np.where(m, -np.inf, v)
  hi = np.where(m, np.inf, v)
  return max(abs(_auc(lo, yb, npos, nneg) - 0.5), abs(_auc(hi, yb, npos, nneg) - 0.5))


def _select(ft, cat, yb):
  """Top-N columns by in-fold univariate AUC; a raw string column inherits its target encoding's score."""
  npos, nneg = float(yb.sum()), float(len(yb) - yb.sum())
  score = {}
  for c, v in ft.items():
    score[c] = _uni_score(np.asarray(v), yb, npos, nneg)
  for c in cat:
    if c in score:
      score[c] = max(score.get(c + "_te", 0.0), score.get(c + "_cnt", 0.0))
  order = sorted(score, key=lambda c: -score[c])
  return set(order[:N_KEEP])


def engineer(X_train, y_train, X_test):
  '''
  本pipeline.py的核心函数

  '''
  y = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else y_train
  yb = (y.astype(str).to_numpy() == "1").astype(np.float64) # yb是由 0.0 和 1.0 组成的浮点数组
  rng = np.random.default_rng(0) # 随机种子：因为后面要用 5 折交叉验证来算目标编码

  cols = [c for c in X_train.columns if c != "row_id"] # 排除 row_id 列
  # 将特征的字段分成数值列(1,2,...)和类别列("A","B",...)两类
  num = [c for c in cols if pd.api.types.is_numeric_dtype(X_train[c])]
  cat = [c for c in cols if c not in num]

  '''
  下面开始进行特征工程。主线就是，每个类别列先生成两三个候选特征，然后与前面保留的数值列一起筛选，最后重新组装成训练表和测试表。
  '''
  ft, fv = {}, {} # 存放列的字典：ft训练集特征，fv测试集特征

  # 对数值列：去掉全缺失列和非缺失值只有一种的列，其余数值列原样进入候选集合。这里没有主动填补数值缺失
  for c in num:
    v = X_train[c]
    # 如果如果一列全空或全是同一个值，直接过滤掉
    if v.notna().sum() == 0 or v.nunique(dropna=True) <= 1:
      continue
    # 将X_train和X_test的数值列强转为numpy数组、float64
    ft[c] = v.to_numpy(dtype=np.float64)
    fv[c] = X_test[c].to_numpy(dtype=np.float64)

  # 对类别列：生成频次编码和目标编码特征
  codes = {}
  for c in cat: # cat是一个列名列表，例如["city", "job"]
    st, sv = _as_str(X_train[c]), _as_str(X_test[c]) # 将类别列强转为字符串类型，缺失值用"__NA__"表示
    ct, cv, nu = _codes(st, sv) # 建立映射：ct是训练集的整数编号，cv是测试集的整数编号，nu是训练样本的类别数
    # 这些编号主要用于后续统计，不会直接作为新特征列写入 ft

    if nu <= 1:
      continue
    codes[c] = (ct, cv, nu)

    # 计算出这个 ID 在全表中出现了多少次（频次编码）以及这个 ID 对应的样本有多大比例是正类（目标编码）
    a, b, d, e = _cat_features(ct, cv, nu, yb, rng)

    # 把四个数组写入 ft、fv，就形成两组同名新特征
    ft[c + "_cnt"], fv[c + "_cnt"] = a, b
    ft[c + "_te"], fv[c + "_te"] = d, e
    if nu <= KEEP_STR_MAX: # 原始类别列也进入候选集合
      ft[c] = st.to_numpy()
      fv[c] = sv.to_numpy()

  # 在全部候选列中选择最多 200 列
  keep = _select(ft, cat, yb)
  # 过滤
  ft = {c: v for c, v in ft.items() if c in keep}
  fv = {c: v for c, v in fv.items() if c in keep}

  # 过滤完把“列字典”转为DataFrame
  Xt = pd.DataFrame(ft, index=X_train.index)
  Xv = pd.DataFrame(fv, index=X_test.index)

  # 先后两个类型转换处理
  for c in Xt.columns:
    if Xt[c].dtype.kind not in "fiu":
      Xt[c] = Xt[c].astype(object)
      Xv[c] = Xv[c].astype(object)
  # 加回 row_id 列
  if "row_id" in X_train.columns:
    Xt["row_id"] = X_train["row_id"].to_numpy()
    Xv["row_id"] = X_test["row_id"].to_numpy()
  Xt.columns = pd.Index(list(Xt.columns), dtype=object)
  Xv.columns = pd.Index(list(Xv.columns), dtype=object)

  return Xt, Xv

def postprocess(pred, y_orig, X_test):
  '''
  保持默认配置
  '''
  return pred
