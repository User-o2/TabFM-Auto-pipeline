import os
import re

import numpy as np
import pandas as pd

TABFM_KWARGS = {'n_feature_crosses': 'sqrt', 'n_svd_features': 'sqrt'}

POWER = 0.89
COLLAPSE = False
COMPOSITES = True
COMPOSITES2 = True
COMPOSITES3 = True
MISSING_FLAGS = False
COMPOSITES4 = False
COMPOSITES5 = False
COMPOSITES6 = True
CAUSE_Z = True
CAUSE_Z_X = False
SENTINEL = False
MISS_COLS = True
# composite columns to drop again after everything is built (the ablation pass; order-safe,
# because the cause scores that consume them have already been computed by this point)
DROP_FEATS = ('n_relapse', 'n_opioid', 'n_nsaid', 'n_drugs', 'n_missing', 'acute',
              'prior_burden')
SENTINEL_COLS = ('R_AB_1_n', 'R_AB_2_n', 'R_AB_3_n', 'NA_R_1_n', 'NA_R_2_n', 'NA_R_3_n',
                 'NOT_NA_1_n', 'NOT_NA_2_n', 'NOT_NA_3_n', 'NA_KB', 'NOT_NA_KB', 'LID_KB')
DROP_EMPTY = True
AUX_TREES = 0       # ExtraTrees size for the stacked log-probability columns; 0 disables them
CONTEXT_FRAC = 1.00   # fraction of the training fold TabFM gets as context (1.0 = all of it)
DRAWS = 7           # TabFM passes the judge averages: 1 full-context + DRAWS-1 subsamples
DRAW_FRAC = 0.94    # size of each subsample draw
ALIVE_ONLY = False  # subsample only the majority class in the extra draws, never a death row
ROW_SHUFFLE = False # extra draws are row permutations of the full fold rather than subsamples
SCALES = (0.97, 0.94, 0.91, 0.88, 0.85)   # per-draw context fractions; overrides DRAWS/DRAW_FRAC
OVERSAMPLE = 0      # extra copies of every non-'alive' training row put into TabFM's context
DEATH_SHIFT = 1 / 3.0   # label-shift correction for that oversampling, applied in postprocess

DUMP_DIR = os.path.join(os.getcwd(), "_oof")
DUMP = os.path.isdir(DUMP_DIR)

# one-hot blocks: prefix -> the columns, filled in from the frame (order matters, it is the code)
ONEHOT = ('nr_', 'np_', 'endocr_', 'zab_leg_', 'ritm_ecg_p_', 'n_r_ecg_p_', 'n_p_ecg_p_', 'fibr_ter_')
DROP = ('KFK_BLOOD', 'IBS_NASL')   # 99.6% / 95.8% missing


def folds(X_train, y_train):
  from sklearn.model_selection import RepeatedStratifiedKFold
  rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
  return list(rskf.split(X_train, y_train))


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _collapse(df, with_count):
  """Replace each one-hot block by an ordinal code (+ a count for the blocks in `with_count`)."""
  out = df.copy()
  for pre in ONEHOT:
    cols = [c for c in df.columns if c.startswith(pre)]
    if len(cols) < 2:
      continue
    B = df[cols].to_numpy(dtype=float)
    allnan = np.isnan(B).all(axis=1)
    # code = 1-based index of the last active column, 0 when none fire, NaN when the block is NaN
    idx = np.where(np.nan_to_num(B) > 0, np.arange(1, len(cols) + 1)[None, :], 0).max(axis=1)
    out = out.drop(columns=cols)
    out[pre + 'code'] = np.where(allnan, np.nan, idx).astype(float)
    if pre in with_count:
      out[pre + 'n'] = np.where(allnan, np.nan, np.nansum(B, axis=1))
  return out


def _composites(df):
  """Clinical composites: the raw table has the pieces but not the ratios/differences."""
  f = {}
  g = lambda c: df[c].to_numpy(dtype=float) if c in df.columns else np.full(len(df), np.nan)
  so, do_ = g('S_AD_ORIT'), g('D_AD_ORIT')
  sk, dk = g('S_AD_KBRIG'), g('D_AD_KBRIG')
  f['pp_orit'] = so - do_
  f['map_orit'] = (so + 2 * do_) / 3.0
  f['pp_kbrig'] = sk - dk
  f['d_sys'] = so - sk                              # blood pressure change between ambulance and ICU
  im = np.c_[g('ant_im'), g('lat_im'), g('inf_im'), g('post_im')]
  f['im_walls'] = (np.nan_to_num(im) > 0).sum(1).astype(float)
  f['im_max'] = np.nanmax(im, axis=1)
  f['im_sum'] = np.nansum(im, axis=1)
  alt, ast = g('ALT_BLOOD'), g('AST_BLOOD')
  f['ast_alt'] = ast / np.where(alt > 0, alt, np.nan)
  f['log_alt'] = np.log1p(alt)
  f['log_ast'] = np.log1p(ast)
  f['k_dev'] = np.abs(g('K_BLOOD') - 4.2)
  f['na_dev'] = np.abs(g('NA_BLOOD') - 140.0)
  f['acute'] = np.nansum(np.c_[g('K_SH_POST'), g('O_L_POST'), g('FIB_G_POST'),
                               g('MP_TP_POST'), g('GT_POST'), g('SVT_POST')], axis=1)
  f['n_relapse'] = np.nansum(np.c_[g('R_AB_1_n'), g('R_AB_2_n'), g('R_AB_3_n')], axis=1)
  f['n_opioid'] = np.nansum(np.c_[g('NA_R_1_n'), g('NA_R_2_n'), g('NA_R_3_n')], axis=1)
  f['n_nsaid'] = np.nansum(np.c_[g('NOT_NA_1_n'), g('NOT_NA_2_n'), g('NOT_NA_3_n')], axis=1)
  f['n_drugs'] = np.nansum(np.c_[g('LID_S_n'), g('B_BLOK_S_n'), g('ANT_CA_S_n'), g('GEPAR_S_n'),
                                 g('ASP_S_n'), g('TIKL_S_n'), g('TRENT_S_n')], axis=1)
  f['n_missing'] = df.isna().sum(axis=1).to_numpy(dtype=float)
  if COMPOSITES2:
    age = g('AGE')
    f['sbp_min'] = np.nanmin(np.c_[so, sk], axis=1)
    f['dbp_min'] = np.nanmin(np.c_[do_, dk], axis=1)
    f['sbp_low'] = (f['sbp_min'] < 100).astype(float)
    f['age_over_sbp'] = 100.0 * age / np.where(f['sbp_min'] > 0, f['sbp_min'], np.nan)
    f['age_x_zsn'] = age * g('ZSN_A')
    f['age_x_inf'] = age * g('INF_ANAM')
    f['wbc_esr'] = g('L_BLOOD') * g('ROE')
    f['log_wbc'] = np.log1p(g('L_BLOOD'))
    f['killip'] = 2.0 * np.nan_to_num(g('K_SH_POST')) + np.nan_to_num(g('O_L_POST'))
    f['prior_burden'] = (np.nan_to_num(g('INF_ANAM')) + np.nan_to_num(g('STENOK_AN'))
                         + np.nan_to_num(g('IBS_POST')) + np.nan_to_num(g('ZSN_A')))
    if MISS_COLS:
      for c in ('S_AD_KBRIG', 'S_AD_ORIT', 'K_BLOOD', 'NA_BLOOD', 'ALT_BLOOD', 'ROE',
                'NA_KB', 'DLIT_AG'):
        f['miss_' + c] = (df[c].isna().to_numpy(dtype=float) if c in df.columns
                          else np.zeros(len(df)))
  if COMPOSITES3:
    # The three per-day ICU blocks (pain relapse, opioid days, NSAID days) are only recorded while
    # the patient is still in the unit: they are missing for 0.3% of the survivors and for 40-47% of
    # the deaths. How far into the stay the record runs is both the strongest death signal in the
    # table and the main clue to which cause it was, so make it explicit instead of leaving it as NaN.
    day = [['R_AB_1_n', 'NA_R_1_n', 'NOT_NA_1_n'],
           ['R_AB_2_n', 'NA_R_2_n', 'NOT_NA_2_n'],
           ['R_AB_3_n', 'NA_R_3_n', 'NOT_NA_3_n']]
    miss = []
    for d, cols in enumerate(day, 1):
      M = np.c_[[df[c].isna().to_numpy(dtype=float) if c in df.columns else np.zeros(len(df))
                 for c in cols]].T
      f['miss_day%d' % d] = M.mean(axis=1)
      miss.append(M)
    M = np.concatenate(miss, axis=1)
    f['n_missing_day'] = M.sum(axis=1)
    f['icu_days'] = 3.0 - f['miss_day2'] - f['miss_day3']
    f['icu_days_x_killip'] = f['icu_days'] * f['killip']
    f['icu_days_x_acute'] = f['icu_days'] * f['acute']
    f['icu_days_x_age'] = f['icu_days'] * g('AGE')
  if COMPOSITES4:
    # More of the day structure: was there activity on a day (not just a record), how the pain and
    # the analgesia trend over the three days, and which whole blocks of the chart are absent.
    r1, r2, r3 = g('R_AB_1_n'), g('R_AB_2_n'), g('R_AB_3_n')
    o1, o2, o3 = g('NA_R_1_n'), g('NA_R_2_n'), g('NA_R_3_n')
    s1, s2, s3 = g('NOT_NA_1_n'), g('NOT_NA_2_n'), g('NOT_NA_3_n')
    act = lambda *v: (np.nan_to_num(np.c_[v]) > 0).any(axis=1).astype(float)
    f['act_day1'], f['act_day2'], f['act_day3'] = act(r1, o1, s1), act(r2, o2, s2), act(r3, o3, s3)
    f['d_relapse'] = np.nan_to_num(r3) - np.nan_to_num(r1)
    f['d_opioid'] = np.nan_to_num(o3) - np.nan_to_num(o1)
    f['d_nsaid'] = np.nan_to_num(s3) - np.nan_to_num(s1)
    f['analgesia1'] = np.nan_to_num(o1) + np.nan_to_num(s1)
    f['no_ambulance'] = np.c_[[df[c].isna().to_numpy(dtype=float) if c in df.columns
                               else np.zeros(len(df))
                               for c in ('NA_KB', 'NOT_NA_KB', 'LID_KB')]].T.mean(axis=1)
    f['labs_missing'] = np.c_[[df[c].isna().to_numpy(dtype=float) if c in df.columns
                               else np.zeros(len(df))
                               for c in ('K_BLOOD', 'NA_BLOOD', 'ALT_BLOOD', 'AST_BLOOD',
                                         'ROE', 'L_BLOOD')]].T.sum(axis=1)
    ecg = [c for c in df.columns if c.startswith(('ritm_ecg_p_', 'n_r_ecg_p_', 'n_p_ecg_p_'))]
    f['ecg_missing'] = df[ecg].isna().any(axis=1).to_numpy(dtype=float) if ecg else np.zeros(len(df))
    f['icu_days_x_imwalls'] = f['icu_days'] * f['im_walls']
    f['icu_days_x_zsn'] = f['icu_days'] * np.nan_to_num(g('ZSN_A'))
    f['icu_days_x_time'] = f['icu_days'] * g('TIME_B_S')
  if COMPOSITES5:
    # Block counts alongside the indicators they come from: how many arrhythmias in the history,
    # how many rhythm / conduction disturbances on the ECG. The blocks stay one-hot (e16 preferred
    # that to collapsing them), so these are additive, not a replacement.
    for pre in ONEHOT:
      cols = [c for c in df.columns if c.startswith(pre)]
      if cols:
        f['cnt_' + pre.strip('_')] = np.nansum(df[cols].to_numpy(dtype=float), axis=1)
    f['ecg_total'] = f.get('cnt_n_r_ecg_p', 0.0) + f.get('cnt_n_p_ecg_p', 0.0)
    if COMPOSITES4:   # last_active_day is built from the fourth batch's per-day activity flags
      f['last_active_day'] = np.where(f['act_day3'] > 0, 3.0,
                                      np.where(f['act_day2'] > 0, 2.0,
                                               np.where(f['act_day1'] > 0, 1.0, 0.0)))
  if COMPOSITES6:
    # One textbook risk profile per cause of death. Predicting which cause is 54% of the loss and
    # the model is close to the prior on the rarest four; the ingredients are all in the table but
    # each cause needs a different handful of them combined, which 906 rows and 6-14 examples of a
    # class are not enough to learn. Coarse integer sums on purpose -- these are directions from
    # cardiology (rupture: first, large anterior infarct in an older patient with no prior angina
    # and preserved pressure; shock: hypotension with a wide infarct; embolism: late arrival,
    # high ESR, atrial fibrillation), not coefficients fitted to this split.
    z = lambda v: np.nan_to_num(v, nan=0.0)
    b = lambda cond: cond.astype(float)
    age = g('AGE')
    f['sc_shock'] = (2 * z(g('K_SH_POST')) + b(so < 100) + b(do_ < 60) + b(f['im_walls'] >= 2))
    f['sc_rupture'] = (b(z(g('INF_ANAM')) == 0) + b(g('ant_im') >= 3) + b(age >= 65)
                       + b(so >= 140) + b(z(g('ZSN_A')) == 0))
    f['sc_chf'] = (2 * z(g('ZSN_A')) + b(z(g('SEX')) == 0) + b(g('lat_im') >= 2)
                   + b(age >= 65) + b(g('ROE') >= 20))
    f['sc_vf'] = (2 * z(g('FIB_G_POST')) + b(age < 65) + z(g('NITR_S')) + b(g('NA_BLOOD') < 136))
    f['sc_thrombo'] = (b(g('TIME_B_S') >= 5) + b(g('ROE') >= 20) + z(g('ritm_ecg_p_02'))
                       + z(g('MP_TP_POST')) + b(z(g('ant_im')) == 0))
    f['sc_edema'] = (2 * z(g('O_L_POST')) + b(age >= 70) + b(g('STENOK_AN') >= 4)
                     + b(g('K_BLOOD') < 4.0) + z(g('NITR_S')))
    f['sc_asystole'] = (b(g('GB') >= 2) + b(g('ant_im') >= 3) + b(age >= 65) + z(g('GIPER_NA')))

  return pd.concat([df, pd.DataFrame(f, index=df.index)], axis=1)


def _plain(df):
  """The judge host unpickles our frames; give it an object-dtype column Index, not the
  pyarrow-backed str Index pandas 3 builds inside pd.concat (host raises ImportError on it)."""
  df = df.copy()
  df.columns = pd.Index(list(map(str, df.columns)), dtype=object)
  return df


def _aux(Xtr, y, Xte):
  """Log-probabilities from an ExtraTrees, as extra columns for TabFM to attend over.

  Out-of-fold for the training rows and the mean of the same fold models for the test rows, so the
  two sides carry the same amount of noise. Fit strictly inside whatever rows engineer() was given.
  """
  from sklearn.ensemble import ExtraTreesClassifier
  from sklearn.model_selection import StratifiedKFold
  A = Xtr.drop(columns=[c for c in ('row_id',) if c in Xtr.columns]).to_numpy(dtype=float)
  B = Xte.drop(columns=[c for c in ('row_id',) if c in Xte.columns]).to_numpy(dtype=float)
  A, B = np.nan_to_num(A, nan=-999.0), np.nan_to_num(B, nan=-999.0)
  y = np.asarray(y)
  classes = np.unique(y)
  mk = lambda: ExtraTreesClassifier(AUX_TREES, min_samples_leaf=3, random_state=0, n_jobs=4)
  oof = np.full((len(A), len(classes)), np.nan)
  te = np.zeros((len(B), len(classes)))
  k = 0
  for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(A, y):
    m = mk().fit(A[tr], y[tr])
    cols = np.searchsorted(classes, m.classes_)
    oof[np.ix_(va, cols)] = m.predict_proba(A[va])
    te[:, cols] += m.predict_proba(B)
    k += 1
  te /= k
  lp = lambda P: np.log(np.clip(P, 1e-4, 1))
  for j, c in enumerate(classes):
    Xtr['aux_' + str(c)] = lp(oof[:, j])
    Xte['aux_' + str(c)] = lp(te[:, j])
  return Xtr, Xte


CAUSE_Z_COLS = ('S_AD_ORIT', 'D_AD_ORIT', 'INF_ANAM', 'ant_im', 'lat_im', 'inf_im', 'ZSN_A', 'SEX',
                'ROE', 'AGE', 'NA_BLOOD', 'NITR_S', 'TIME_B_S', 'STENOK_AN', 'K_BLOOD', 'GB')


def _cause_z(df, stats):
  """Smooth version of the per-cause risk scores: signed sums of z-scores instead of thresholds.

  The z-scores use the training fold's own mean/std (passed in as `stats`), and the signs are the
  directions the class-conditional means show -- shock is hypotension, rupture is a first large
  anterior infarct at a preserved pressure, embolism is a late arrival with a high ESR.
  """
  z = {}
  for c in CAUSE_Z_COLS:
    m, sd = stats.get(c, (0.0, 1.0))
    v = df[c].to_numpy(dtype=float) if c in df.columns else np.full(len(df), np.nan)
    z[c] = np.nan_to_num((v - m) / (sd if sd > 0 else 1.0))
  ev = lambda c: (np.nan_to_num(df[c].to_numpy(dtype=float)) if c in df.columns
                  else np.zeros(len(df)))
  out = {
      'z_shock': -z['S_AD_ORIT'] - z['D_AD_ORIT'] + 2 * ev('K_SH_POST'),
      'z_rupture': -z['INF_ANAM'] + z['ant_im'] + z['S_AD_ORIT'] - z['ZSN_A'],
      'z_chf': z['ZSN_A'] - z['SEX'] + z['lat_im'] + z['ROE'],
      'z_vf': -z['AGE'] - z['NA_BLOOD'] + z['NITR_S'] + 2 * ev('FIB_G_POST'),
      'z_thrombo': z['TIME_B_S'] + z['ROE'] - z['ant_im'] + z['inf_im'] + ev('ritm_ecg_p_02'),
      'z_edema': z['AGE'] + z['STENOK_AN'] - z['K_BLOOD'] + 2 * ev('O_L_POST'),
      'z_asystole': z['GB'] + z['ant_im'] + z['AGE'],
  }
  if CAUSE_Z_X and 'icu_days' in df.columns:
    d = df['icu_days'].to_numpy(dtype=float)
    for k in list(out):
      out[k + '_x_days'] = out[k] * d
  return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


def _missing_flags(Xtr, Xte):
  """One indicator per distinct missingness pattern, for the patterns that occur often enough.

  Whether a cell was recorded is a signal in its own right here (no ambulance report, no lab drawn,
  the patient was no longer in the unit) and TabFM only ever sees the NaN. Columns that go missing
  together -- the ECG blocks, the per-day ICU blocks, the paired blood pressures -- share one flag.
  """
  base = [c for c in Xtr.columns if c != 'row_id']
  seen, groups = {}, []
  for c in base:
    m = Xtr[c].isna().to_numpy()
    r = m.mean()
    if r < 0.02 or r > 0.98:
      continue
    key = m.tobytes()
    if key in seen:
      continue
    seen[key] = c
    groups.append(c)
  for c in groups:
    Xtr['nan_' + c] = Xtr[c].isna().to_numpy(dtype=float)
    Xte['nan_' + c] = Xte[c].isna().to_numpy(dtype=float)
  return Xtr, Xte


def engineer(X_train, y_train, X_test):
  if COMPOSITES:
    X_train, X_test = _composites(X_train), _composites(X_test)
  if SENTINEL:
    # "Not recorded" is a state, not a missing measurement, for the per-day ICU columns and the
    # ambulance drug chart: the patient had left the unit, or no ambulance team filled a form.
    # TabFM imputes a NaN, which puts those rows at the column average; -1 keeps them separate
    # and below every real value (all of these are counts in 0..4).
    for c in SENTINEL_COLS:
      if c in X_train.columns:
        X_train[c] = X_train[c].fillna(-1.0)
        X_test[c] = X_test[c].fillna(-1.0)
  if CAUSE_Z:
    stats = {c: (float(X_train[c].mean()), float(X_train[c].std()))
             for c in CAUSE_Z_COLS if c in X_train.columns}
    X_train, X_test = _cause_z(X_train, stats), _cause_z(X_test, stats)
  if DROP_FEATS:
    X_train = X_train.drop(columns=[c for c in DROP_FEATS if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in DROP_FEATS if c in X_test.columns])
  if DROP_EMPTY:
    X_train = X_train.drop(columns=[c for c in DROP if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in DROP if c in X_test.columns])
  if MISSING_FLAGS:
    X_train, X_test = _missing_flags(X_train.copy(), X_test.copy())
  if not COLLAPSE:
    return _plain(X_train), _plain(X_test)
  drop = [c for c in DROP if c in X_train.columns]
  Xtr, Xte = X_train.drop(columns=drop), X_test.drop(columns=drop)
  with_count = {pre for pre in ONEHOT
                if np.nanmax(np.nansum(Xtr[[c for c in Xtr.columns if c.startswith(pre)]]
                                       .to_numpy(dtype=float), axis=1)) > 1}
  Xtr, Xte = _collapse(Xtr, with_count), _collapse(Xte, with_count)
  Xtr, Xte = _plain(Xtr), _plain(Xte[Xtr.columns])
  if AUX_TREES:
    Xtr, Xte = _aux(Xtr, y_train, Xte)
  return _plain(Xtr), _plain(Xte[Xtr.columns])


def sample(X_train, y_train, X_test, max_rows):
  """One full-context draw plus DRAWS-1 stratified subsamples; the judge averages the passes.

  n_estimators is fixed at 8, so extra draws are the only way to enlarge the ensemble. The
  subsamples are stratified because thromboembolism has 6 rows in a training fold and a plain
  uniform 88% would sometimes drop the class out of the context entirely.
  """
  idx = np.arange(len(X_train))
  if CONTEXT_FRAC < 1.0:      # calibration probe: how does the optimal POWER move with context size
    from sklearn.model_selection import StratifiedShuffleSplit
    keep, _ = next(iter(StratifiedShuffleSplit(1, train_size=CONTEXT_FRAC, random_state=3)
                        .split(idx, np.asarray(y_train))))
    return [np.sort(keep)]
  if DRAWS <= 1:
    return [idx]
  yv = np.asarray(y_train)
  if ROW_SHUFFLE:
    # Diversity with no information loss, if TabFM's context is not perfectly order-invariant:
    # every draw is the whole training fold, just in a different row order. If the model is
    # order-invariant the draws collapse to one and the score simply reproduces DRAWS=1.
    rng = np.random.default_rng(11)
    return [idx] + [rng.permutation(idx) for _ in range(DRAWS - 1)]
  if SCALES:
    # Multi-scale draws: the full fold plus one subsample at each fraction in SCALES. Members that
    # saw 85% and 97% of the rows are more different from each other than two independent 94%
    # subsamples are, and diversity is the only thing an extra draw can buy.
    from sklearn.model_selection import StratifiedShuffleSplit
    out = [idx]
    for i, fr in enumerate(SCALES):
      keep, _ = next(iter(StratifiedShuffleSplit(1, train_size=fr, random_state=100 + i)
                          .split(idx, yv)))
      out.append(np.sort(keep))
    return out
  if ALIVE_ONLY:
    # Every death row is in every draw: the rarest class has 6 rows in a training fold, and a draw
    # that happens to drop one of them loses a sixth of what there is to learn about it. The
    # diversity comes from resampling the 84% majority instead, which can spare the rows.
    rng = np.random.default_rng(7)
    keep_all = idx[yv != 'alive']
    alive = idx[yv == 'alive']
    k = int(round(DRAW_FRAC * len(alive)))
    return [idx] + [np.sort(np.concatenate([keep_all, rng.choice(alive, k, replace=False)]))
                    for _ in range(DRAWS - 1)]
  from sklearn.model_selection import StratifiedShuffleSplit
  sss = StratifiedShuffleSplit(n_splits=DRAWS - 1, train_size=DRAW_FRAC, random_state=7)
  return [idx] + [np.sort(keep) for keep, _ in sss.split(idx, yv)]


def _dump(pred, X_test):
  if not DUMP:
    return
  try:
    out = pd.DataFrame(np.asarray(pred, dtype=float), columns=[str(c) for c in pred.columns])
    out.insert(0, "row_id", np.asarray(X_test["row_id"]))
    n = len(os.listdir(DUMP_DIR))
    out.to_csv(os.path.join(DUMP_DIR, "part_%03d.csv" % n), index=False)
  except Exception:
    pass


def postprocess(pred, y_orig, X_test):
  _dump(pred, X_test)
  Q = np.clip(np.asarray(pred, dtype=float), 1e-12, None)
  if OVERSAMPLE > 0:
    w = np.array([1.0 if c == 'alive' else DEATH_SHIFT for c in pred.columns])
    Q = Q * w
    Q = Q / Q.sum(axis=1, keepdims=True)
  Q = Q ** POWER
  Q = Q / Q.sum(axis=1, keepdims=True)
  return pd.DataFrame(Q, columns=pred.columns, index=pred.index)
