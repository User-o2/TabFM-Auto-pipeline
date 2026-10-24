import numpy as np
import pandas as pd

pd.set_option('mode.string_storage', 'python')   # judge process cannot unpickle arrow-backed strings

TABFM_KWARGS = {'norm_methods': ['none', 'quantile_rtdl'],
                'outlier_threshold': 20.0}

DRUGS = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
         'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
         'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
         'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
         'metformin-pioglitazone']
# constant or <150 non-'No' rows in the training split
DEAD = ['acetohexamide', 'examide', 'citoglipton', 'glimepiride-pioglitazone']
# univariate AUC <= 0.5035 on the training split; dropped after the med counts are built
WEAK = []
DIAG = ['diag_1', 'diag_2', 'diag_3']
LUMP = {'medical_specialty': 1, 'admission_source_id': 1}
MIN_DIAG = 1
AGE_MID = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45, '[50-60)': 55,
           '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95}
WT_MID = {'[0-25)': 12, '[25-50)': 37, '[50-75)': 62, '[75-100)': 87, '[100-125)': 112,
          '[125-150)': 137, '[150-175)': 162, '[175-200)': 187, '>200': 212}
DOSE = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': -1}


def _icd_num(s):
  s = s.fillna('').astype(str)
  return pd.to_numeric(s.str.extract(r'^(\d+\.?\d*)', expand=False), errors='coerce')


def _icd_group(s):
  """ICD-9 code -> chapter, with 250.x split out as its own 'Diabetes' chapter."""
  s = s.fillna('').astype(str)
  v = _icd_num(s)
  g = np.full(len(s), 'Other', dtype=object)
  ok = v.notna().to_numpy()
  vv = v.to_numpy()
  for lo, hi, lab in [(1, 140, 'Infect'), (140, 240, 'Neoplasm'), (240, 280, 'EndoMeta'),
                      (280, 290, 'Blood'), (290, 320, 'Mental'), (320, 390, 'Nervous'),
                      (390, 460, 'Circ'), (460, 520, 'Resp'), (520, 580, 'Digest'),
                      (580, 630, 'Genito'), (630, 680, 'Pregnancy'), (680, 710, 'Skin'),
                      (710, 740, 'Musculo'), (740, 760, 'Congenital'), (760, 780, 'Perinatal'),
                      (780, 800, 'Symptoms'), (800, 1000, 'Injury')]:
    g[ok & (vv >= lo) & (vv < hi)] = lab
  g[ok & (vv >= 250) & (vv < 251)] = 'Diabetes'
  g[ok & (vv == 785)] = 'Circ'
  g[ok & (vv == 786)] = 'Resp'
  g[ok & (vv == 787)] = 'Digest'
  g[ok & (vv == 788)] = 'Genito'
  a = s.to_numpy()
  g[np.char.startswith(a.astype(str), 'V')] = 'V'
  g[np.char.startswith(a.astype(str), 'E')] = 'E'
  g[a == ''] = 'NA'
  return pd.Series(g, index=s.index, dtype=object)


def _disch_group(s):
  """24 discharge dispositions (readmission rate 0.00-0.45) -> 7 clinical buckets."""
  t = s.fillna('NA').astype(str)
  low = t.str.lower()
  g = np.full(len(t), 'other', dtype=object)
  g[low.str.contains('short term hospital').to_numpy()] = 'shortterm'
  g[low.str.contains('snf|icf|nursing facility|long term care').to_numpy()] = 'nursing'
  g[low.str.contains('rehab|psychiatric|inpatient care institution|swing bed').to_numpy()] = 'rehab'
  g[low.str.contains('home health|home iv').to_numpy()] = 'homehealth'
  g[(t == 'Discharged to home').to_numpy()] = 'home'
  g[low.str.contains('expired|hospice').to_numpy()] = 'terminal'
  g[(t == 'NA').to_numpy()] = 'NA'
  return pd.Series(g, index=s.index, dtype=object)


def _lump(col_tr, col_te, min_count):
  """Levels rarer than min_count in the fold's training rows -> 'rare'."""
  a = col_tr.fillna('NA').astype(str)
  b = col_te.fillna('NA').astype(str)
  keep = set(a.value_counts()[lambda v: v >= min_count].index)
  return (a.where(a.isin(keep), 'rare').to_numpy(),
          b.where(b.isin(keep), 'rare').to_numpy())


def _lump_to_group(col_tr, col_te, g_tr, g_te, min_count):
  """Rare ICD code -> its 3-digit trunk; a rare trunk -> its chapter."""
  a = col_tr.fillna('NA').astype(str)
  b = col_te.fillna('NA').astype(str)
  ta = a.str.split('.').str[0]
  tb = b.str.split('.').str[0]
  keep = set(a.value_counts()[lambda v: v >= min_count].index)
  keep_t = set(ta.value_counts()[lambda v: v >= min_count].index)
  out = []
  for k, t, g in ((a, ta, g_tr), (b, tb, g_te)):
    v = ('g:' + g.astype(str)).to_numpy()
    v = np.where(t.isin(keep_t).to_numpy(), 't:' + t.to_numpy(), v)
    out.append(np.where(k.isin(keep).to_numpy(), k.to_numpy(), v))
  return out[0], out[1]


def _feat(df):
  df = df.copy()
  df = df.drop(columns=[c for c in DEAD if c in df.columns])
  df['age_ord'] = df['age'].map(AGE_MID).astype(float)
  df['weight_mid'] = df['weight'].map(WT_MID).astype(float)
  df = df.drop(columns=['age', 'weight'])

  for c in DIAG:
    df[c + '_g'] = _icd_group(df[c])
    df[c + '_n'] = _icd_num(df[c])
  gs = np.stack([df[c + '_g'].to_numpy() for c in DIAG], 1)
  df['n_diab_diag'] = (gs == 'Diabetes').sum(1)
  df['n_circ_diag'] = (gs == 'Circ').sum(1)

  df['disch_g'] = _disch_group(df['discharge_disposition_id'])
  df['service_util'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
  live = [c for c in DRUGS if c in df.columns]
  M = np.stack([df[c].map(DOSE).fillna(0).to_numpy() for c in live], 1)
  df['n_meds_taken'] = (M != 0).sum(1)
  df['n_meds_up'] = (M == 2).sum(1)
  df['n_meds_down'] = (M == -1).sum(1)
  df['n_meds_changed'] = df['n_meds_up'] + df['n_meds_down']

  t = df['time_in_hospital'].astype(float)
  df['med_per_day'] = df['num_medications'] / t
  df['lab_per_day'] = df['num_lab_procedures'] / t
  df['proc_per_day'] = df['num_procedures'] / t
  return df.drop(columns=[c for c in WEAK if c in df.columns])


def folds(X_train, y_train):
  """5 folds, not the judge's 3: bigger training folds (38.1k, against 47.7k at the final)
  and less split noise, to re-rank candidates that were inside the noise at 3 folds."""
  from sklearn.model_selection import StratifiedKFold
  kf = StratifiedKFold(5, shuffle=True, random_state=0)
  return list(kf.split(X_train, np.asarray(y_train).astype(str)))


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def engineer(X_train, y_train, X_test):
  a, b = _feat(X_train), _feat(X_test)
  for c, m in LUMP.items():                      # frequencies from the fold's training rows only
    a[c], b[c] = _lump(a[c], b[c], m)
  for c in DIAG:
    a[c], b[c] = _lump_to_group(a[c], b[c], a[c + '_g'], b[c + '_g'], MIN_DIAG)
  cols = [c for c in a.columns if c in set(b.columns)]
  a, b = a[cols].copy(), b[cols].copy()
  for d in (a, b):                               # plain object strings: the judge process
    for c in cols:                               # cannot unpickle an arrow-backed StringArray
      if not pd.api.types.is_numeric_dtype(d[c]):
        miss = d[c].isna().to_numpy()
        v = np.array(d[c].astype(str), dtype=object, copy=True)
        v[miss] = np.nan
        d[c] = v
  return a, b


def postprocess(pred, y_orig, X_test):
  return pred
