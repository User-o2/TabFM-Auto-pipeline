import numpy as np
from scipy.stats import norm as _NORM, rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import QuantileTransformer

TABFM_KWARGS = {'norm_methods': ['none', 'power', 'quantile', 'robust']}

TOP_K = 52            # columns handed to TabFM, chosen inside each fold
AUX_W = 0.12          # ExtraTrees' share of the final ranking
AUX_FILE = '.aux_pred.npy'
ID_COLS = ('row_id',)   # present in the final's frames, absent in val's; the judge drops it


def folds(X_train, y_train):
  """Nine folds with the geometry of the nine the final is scored on: 2/3 train, 1/3 held out.

  The judge's default 3-fold left a 0.0086 standard error on the mean, which is larger than
  anything measured here; three repeats cut the split noise without changing the train size the
  score describes.
  """
  y = np.asarray(y_train).ravel()
  cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=20260908)
  return [(tr, va) for tr, va in cv.split(np.zeros(len(y)), y)]


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _add_features(X):
  """Candidates for the selection: gaps between near-duplicate readings, and a few ratios.

  The gaps are the point. Nine pairs of columns here correlate above 0.999, so a rank transform
  gives TabFM two almost identical marginals and no way back to the difference between them.
  """
  d = X['Debt_Ratio_Percent']
  return X.assign(
      roa_x_debt=X['Net_Income_to_Total_Assets'] * (1.0 - d),
      ni_minus_bd=X['Net_Income_to_Total_Assets'] - X['Borrowing_Dependency'],
      eps_over_debt=X['Persistent_EPS_Last_4_Seasons'] / (d + 1e-6),
      altman=(1.2 * X['Working_Capital_to_Total_Assets']
              + 1.4 * X['Retained_Earnings_to_Total_Assets']
              + 3.3 * X['Net_Income_to_Total_Assets']
              + 0.6 * X['Equity_to_Liability']
              + 1.0 * X['Total_Asset_Turnover']),
      cf_cover=X['Cash_Flow_to_Liability'] / (X['Current_Liability_to_Assets'] + 1e-6),
      liq=X['Quick_Ratio'] - X['Current_Ratio'],
      roa_spread=X['ROA_C_Before_Interest_Depreciation'] - X['ROA_A_Before_Interest_After_Tax'],
      roa_spread2=(X['ROA_B_Before_Interest_Depreciation_After_Tax']
                   - X['ROA_A_Before_Interest_After_Tax']),
      margin_gap=X['Operating_Gross_Margin'] - X['Realized_Sales_Gross_Margin'],
      pershare_gap=X['Net_Profit_Before_Tax_Per_Share'] - X['Operating_Profit_Per_Share'],
      nv_gap=X['Net_Value_Per_Share_A'] - X['Net_Value_Per_Share_C'],
      int_rate_gap=X['PreTax_Net_Interest_Rate'] - X['AfterTax_Net_Interest_Rate'],
      cont_gap=X['Continuous_Interest_Rate_After_Tax'] - X['AfterTax_Net_Interest_Rate'],
      te_ratio=X['Total_Income_to_Total_Expense'] / (X['Total_Expense_to_Assets'] + 1e-6),
      wc_eq=X['Working_Capital_to_Equity'] - X['Current_Liabilities_to_Equity'])


def _importance(X, y, seeds=(0, 1, 2)):
  """Averaged RF gini importance -- one seed is noisy at 98 positives, six were no better."""
  imp = np.zeros(X.shape[1])
  for s in seeds:
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                class_weight='balanced_subsample', max_features='sqrt',
                                random_state=s, n_jobs=8)
    rf.fit(X.values, y)
    imp += rf.feature_importances_
  return imp / len(seeds)


def engineer(X_train, y_train, X_test):
  ids = [c for c in X_train.columns if c in ID_COLS]
  base = [c for c in X_train.columns if c not in ID_COLS]
  X_train = X_train[ids].join(_add_features(X_train[base].astype(float)))
  X_test = X_test[ids].join(_add_features(X_test[base].astype(float)))
  feats = [c for c in X_train.columns if c not in ID_COLS]

  y = (np.asarray(y_train).ravel() == np.unique(np.asarray(y_train).ravel())[-1]).astype(int)
  imp = _importance(X_train[feats].astype(float).fillna(0.0), y)   # fold's own rows: no leakage
  keep = [feats[i] for i in sorted(np.argsort(-imp)[:TOP_K])]
  _stash_aux(X_train[feats].astype(float), y, X_test[feats].astype(float))

  # One rank-to-normal map per column, fitted on the training and test rows pooled: no labels are
  # involved, and it is the only way both sides arrive under the same transform (TabFM fits its
  # normalisers on the context alone).
  A, B = X_train[keep].astype(float).copy(), X_test[keep].astype(float).copy()
  qt = QuantileTransformer(output_distribution='normal', random_state=0,
                           n_quantiles=len(A) + len(B))
  qt.fit(np.vstack([A.to_numpy(), B.to_numpy()]))
  A[:] = qt.transform(A.to_numpy())     # in place: a rebuilt frame would carry an arrow-backed
  B[:] = qt.transform(B.to_numpy())     # column Index, which the judge process cannot unpickle
  return X_train[ids].join(A), X_test[ids].join(B)


def _stash_aux(Xtr, y, Xte):
  """ExtraTrees scores for the test rows, fitted on this fold's training rows only.

  postprocess() is handed pred / y_orig / X_test but not X_train, and each hook stage runs in its
  own subprocess, so the workspace is the only way to carry a fitted model's output across.
  The forest reads all 108 columns rather than the 52 TabFM gets: it wants the width TabFM does
  not (eval 27), and the blend is paid for being unlike TabFM rather than for being strong -- a
  four-model aux with a better standalone score blended worse (eval 21).
  """
  A, B = Xtr.to_numpy(dtype=float), Xte.to_numpy(dtype=float)
  if len(np.unique(y)) < 2:                      # `judge.py check` slices can be single-class
    np.save(AUX_FILE, np.zeros(len(B)))
    return
  et = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, max_features='sqrt',
                            class_weight='balanced_subsample', random_state=0, n_jobs=8)
  et.fit(A, y)
  np.save(AUX_FILE, rankdata(et.predict_proba(B)[:, 1]) / len(B))


def postprocess(pred, y_orig, X_test):
  """Rank-average the ExtraTrees score into TabFM's ranking at AUX_W."""
  aux = np.load(AUX_FILE)
  if pred.shape[1] != 2 or not aux.any():
    return pred
  out = pred.copy()
  pos = out.columns[-1]
  n = len(out)
  probit = lambda u: _NORM.ppf(np.clip(u, 0.5 / n, 1.0 - 0.5 / n))
  blend = (1.0 - AUX_W) * probit(rankdata(out[pos].to_numpy()) / n) + AUX_W * probit(aux)
  blend = rankdata(blend) / n
  out[pos] = blend
  out[out.columns[0]] = 1.0 - blend
  return out
