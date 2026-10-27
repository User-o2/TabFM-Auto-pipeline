'''
典型案例：利用领域知识语义构造特征，展示 LLM 语义和领域知识能力

- TABFM_KWARGS={}：使用框架默认配置
- preprocess()：原样返回
- engineer()：【重点】构造物理特征并删除部分原始列
- sample()：返回全部训练行的一份索引
- postprocess()：原样返回预测
'''

import numpy as np
import pandas as pd

'''
- OPT：这份流水线自己定义的控制字典
- TABFM_KWARGS：交给外部框架、用于配置 TabFM 的字典。论文提到的归一化、内置交叉特征、SVD、上下文行数和 NNLS 等设置，可通过这个入口配置
TABFM_KWARGS = {} 的含义是：当前流水线没有显式传入上述配置项，启用默认值
'''
OPT = {'groups': ['raw', 'logs', 'st', 're', 'ratio', 'bpm'],
       'drop': ['frequency', 'free-stream-velocity', 'chord-length']}
TABFM_KWARGS = {}

C_SOUND = 340.46      # m/s
NU = 1.5e-5           # kinematic viscosity of air, m^2/s
CFG = ['chord-length', 'free-stream-velocity', 'attack-angle']


def _phys(df, groups):
  '''
  本函数服务于engineer()
  '''
  f = df['frequency'].to_numpy(dtype=float) # 频率，Hz
  a = df['attack-angle'].to_numpy(dtype=float) # 攻角，度
  c = df['chord-length'].to_numpy(dtype=float) # 弦长，m
  u = df['free-stream-velocity'].to_numpy(dtype=float) # 自由流速度，m/s
  d = df['suction-side-displacement-thickness'].to_numpy(dtype=float) # 吸力侧位移厚度，m

  # 物理公式计算
  m = u / C_SOUND
  re_c = u * c / NU
  re_d = u * d / NU
  st_d = f * d / u

  #  构造各类特征组
  out = pd.DataFrame(index=df.index)
  if 'raw' in groups: # 保留原始特征
    for col in df.columns:
      out[col] = df[col].to_numpy()
  if 'logs' in groups: # 对数变换特征
    out['log_f'] = np.log10(f)
    out['log_d'] = np.log10(d)
    out['log_c'] = np.log10(c)
  if 'st' in groups: # 斯特劳哈尔数
    out['log_St_d'] = np.log10(st_d)
    out['log_St_c'] = np.log10(f * c / u)
  if 're' in groups: # 雷诺数
    out['log_Re_c'] = np.log10(re_c)
    out['log_Re_d'] = np.log10(re_d)
  if 'ratio' in groups: # 几何比例
    out['log_d_over_c'] = np.log10(d / c)
  if 'bpm' in groups: # [最具领域知识] BPM 半经验声学模型
    out['bpm_scale'] = 10.0 * np.log10(d) + 50.0 * np.log10(m)
  if 'logu' in groups:
    out['log_u'] = np.log10(u)
  if 'ord' in groups: # the design is a small discrete grid
    for nm, v in (('o_f', f), ('o_a', a), ('o_c', c), ('o_u', u)):
      out[nm] = pd.Series(v).rank(method='dense').to_numpy() - 1.0
  if 'st1r' in groups:                                          # distance to the BPM peak Strouhal
    out['log_St1r'] = np.log10(st_d / (0.02 * m ** -0.6))
  if 'ang' in groups:
    out['a2'] = a ** 2
    out['a_stall'] = np.maximum(a - 12.5, 0.0)
    out['sin_a'] = np.sin(np.deg2rad(a))
    out['a_x_logf'] = a * np.log10(f)
  if 'bpm2' in groups:
    st1 = 0.02 * m ** -0.6                                     # peak Strouhal, zero incidence
    shift = np.where(a < 1.33, 1.0,
                     np.where(a <= 12.5, 10.0 ** (0.0054 * (a - 1.33) ** 2), 4.72))
    st2 = st1 * shift
    k1 = np.where(re_c < 2.47e5, -4.31 * np.log10(re_c) + 156.3,
                  np.where(re_c > 8.0e5, 128.5, -9.0 * np.log10(re_c) + 181.6))
    dk1 = np.where(re_d < 5000.0, a * (1.43 * np.log10(np.maximum(re_d, 1.0)) - 5.29), 0.0)
    out['log_St1r'] = np.log10(st_d / st1)
    out['log_St2r'] = np.log10(st_d / st2)
    out['log_St_shift'] = np.log10(shift)
    out['bpm_K1'] = k1
    out['bpm_dK1'] = dk1

  return out


def _brackets(X_train, y_train, X_test, slim):
  """For each row, the training levels immediately below and above it in log-frequency inside its own
  configuration. Training rows are served leave-one-out, so the columns mean the same thing on both
  sides. No configuration-mean column: a leave-one-out mean is (S - y_i)/(n-1), which moves against
  y_i inside a configuration and taught TabFM the wrong sign (RESULT 17, 2.49)."""
  y = np.asarray(y_train, dtype=float).ravel()
  lf_tr = np.log10(X_train['frequency'].to_numpy(dtype=float))
  lf_te = np.log10(X_test['frequency'].to_numpy(dtype=float))
  k_tr = list(zip(*(X_train[c].to_numpy() for c in CFG)))
  k_te = list(zip(*(X_test[c].to_numpy() for c in CFG)))
  groups = {}
  for i, k in enumerate(k_tr):
    groups.setdefault(k, []).append(i)
  sortd = {}
  for k, ix in groups.items():
    ix = np.asarray(ix)[np.argsort(lf_tr[np.asarray(ix)], kind='stable')]
    sortd[k] = (lf_tr[ix], y[ix])

  def make(n):
    return {c: np.full(n, np.nan) for c in
            (('nb_interp', 'nb_slope') if slim else
             ('nb_lo', 'nb_hi', 'nb_dlo', 'nb_dhi', 'nb_interp', 'nb_slope'))}

  def put(dst, row, ylo, yhi, dlo, dhi):
    if not slim:
      dst['nb_lo'][row], dst['nb_hi'][row] = ylo, yhi
      dst['nb_dlo'][row], dst['nb_dhi'][row] = dlo, dhi
    if not np.isnan(ylo) and not np.isnan(yhi):
      dst['nb_interp'][row] = ylo + (yhi - ylo) * dlo / (dlo + dhi)
      dst['nb_slope'][row] = (yhi - ylo) / (dlo + dhi)
    elif not np.isnan(ylo):
      dst['nb_interp'][row] = ylo
    elif not np.isnan(yhi):
      dst['nb_interp'][row] = yhi

  ctr, cte = make(len(lf_tr)), make(len(lf_te))
  for k, ix in groups.items():
    gl, gy = sortd[k]
    n = len(gl)
    order = np.asarray(ix)[np.argsort(lf_tr[np.asarray(ix)], kind='stable')]
    for p, row in enumerate(order):
      put(ctr, row,
          gy[p - 1] if p > 0 else np.nan, gy[p + 1] if p + 1 < n else np.nan,
          gl[p] - gl[p - 1] if p > 0 else np.nan, gl[p + 1] - gl[p] if p + 1 < n else np.nan)
  for row, (k, lf) in enumerate(zip(k_te, lf_te)):
    if k not in sortd:
      continue
    gl, gy = sortd[k]
    p = int(np.searchsorted(gl, lf))
    put(cte, row,
        gy[p - 1] if p > 0 else np.nan, gy[p] if p < len(gl) else np.nan,
        lf - gl[p - 1] if p > 0 else np.nan, gl[p] - lf if p < len(gl) else np.nan)
  return ctr, cte


def _ecdf_grid(n):
  return (np.arange(n) + 1.0) / (n + 1.0)


def preprocess(X_train, y_train, X_test):
  '''
  负责模型输入前的清洗与目标处理。
  本例最终不执行变换。
  '''
  # 实际上没有'ytrans'这个键
  if OPT.get('ytrans') == 'rankgauss': # reshape the target, inverted in postprocess from y_orig
    from scipy.stats import norm, rankdata
    y = np.asarray(y_train, dtype=float).ravel()
    z = norm.ppf(rankdata(y, method='average') / (len(y) + 1.0))
    y_train = pd.Series(z, index=y_train.index, name=getattr(y_train, 'name', 'y'))
  return X_train, y_train, X_test


def sample(X_train, y_train, X_test, max_rows):
  """
  决定哪些训练行作为上下文，返回索引列表。
  本例返回的是一份完整上下文，没有额外做子采样、bootstrap 或行排列。

  TabFM-Auto 可以通过对训练数据进行不同的上下文采样来实现集成预测，但由于模型推理使用相同的随机种子，因此不同采样之间的多样性只能来自上下文训练样本的差异，而不是模型自身的随机性。
  """
  n, k, fr = len(X_train), OPT.get('draws', 1), OPT.get('draw_frac', 1.0)
  rng = np.random.default_rng(12345)

  if OPT.get('perm_draws'): # same rows, different order: is TabFM's context order-sensitive?
    return [rng.permutation(n) for _ in range(OPT['perm_draws'])]
  if k <= 1 and fr >= 1.0:
    return [np.arange(n)]
  m = max(2, int(round(fr * n)))
  if OPT.get('bootstrap'):
    return [np.sort(rng.choice(n, m, replace=True)) for _ in range(k)]
  return [np.sort(rng.choice(n, m, replace=False)) for _ in range(k)]


def engineer(X_train, y_train, X_test):
  '''
  负责模型输入前的特征构造或重写。
  本例中，负责将原始变量改写成物理特征。具体来说，原本 5 个特征，新构造了 9 个特征，并删去 3 个原始特征。得到：5+9-3=11个预测特征
  '''
  g = OPT['groups']
  tr, te = _phys(X_train, g), _phys(X_test, g)

  if 'cfgid' in g: # a string key TabFM can match exactly on
    for frame, src in ((tr, X_train), (te, X_test)):
      frame['cfg_id'] = [f'{c1:g}_{u1:g}_{a1:g}' for c1, u1, a1 in
                         zip(*(src[c].to_numpy() for c in CFG))]
  if 'nbr' in g: # bracketing levels of the same sweep, LOO on train
    for frame, cols in zip((tr, te), _brackets(X_train, y_train, X_test, OPT.get('nbr_slim'))):
      for k2, v2 in cols.items():
        frame[k2] = v2
  if 'cfgmean' in g: # leave-one-out mean level of the configuration
    y = np.asarray(y_train, dtype=float).ravel()
    k_tr = list(zip(*(X_train[c].to_numpy() for c in CFG)))
    k_te = list(zip(*(X_test[c].to_numpy() for c in CFG)))
    tot, cnt = {}, {}
    for k, v in zip(k_tr, y):
      tot[k] = tot.get(k, 0.0) + v
      cnt[k] = cnt.get(k, 0) + 1
    tr['cfg_mean'] = [(tot[k] - v) / (cnt[k] - 1) if cnt[k] > 1 else np.nan for k, v in zip(k_tr, y)]
    te['cfg_mean'] = [tot[k] / cnt[k] if k in cnt else np.nan for k in k_te]
  fr = OPT.get('max_rows_frac')
  if fr:                       # per-member row subsampling: bagging inside the 8-member ensemble,
    TABFM_KWARGS['max_num_rows'] = max(2, int(round(fr * len(X_train))))   # at no extra compute
  
  for c in OPT.get('drop', []):
    tr, te = tr.drop(columns=[c]), te.drop(columns=[c])

  for c, k in OPT.get('dup', []): # a repeated column takes more of TabFM's attention
    for j in range(k):
      tr[f'{c}_r{j}'], te[f'{c}_r{j}'] = tr[c].to_numpy(), te[c].to_numpy()
  for c in ('row_id',):
    if c in X_train.columns and c not in tr.columns:
      tr[c], te[c] = X_train[c].to_numpy(), X_test[c].to_numpy()

  return tr, te


def postprocess(pred, y_orig, X_test):
  '''
  负责将模型输出转换为最终预测。
  本例直接返回。前面没有改变目标，后面也不需要逆变换。
  '''
  if OPT.get('ytrans') == 'rankgauss':
    from scipy.stats import norm
    ys = np.sort(np.asarray(y_orig, dtype=float).ravel())
    u = norm.cdf(np.asarray(pred, dtype=float).ravel())
    back = np.interp(u, _ecdf_grid(len(ys)), ys)
    return pd.Series(back, index=pred.index, name=getattr(pred, 'name', None))
  if OPT.get('clip'):
    y = np.asarray(y_orig, dtype=float).ravel()
    return pred.clip(y.min(), y.max())
  
  return pred
