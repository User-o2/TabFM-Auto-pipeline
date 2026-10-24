import numpy as np

TABFM_KWARGS = {}

DUMP = 'final'


def folds(X_train, y_train):
  from sklearn.model_selection import RepeatedStratifiedKFold
  return list(RepeatedStratifiedKFold(
      n_splits=10, n_repeats=8, random_state=1234).split(X_train, y_train))


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _feat(df):
  sbp = df["SystolicBP"].astype(float)
  dbp = df["DiastolicBP"].astype(float)
  hr = df["HeartRate"].astype(float)
  d = df.drop(columns=["DiastolicBP"])          # = SystolicBP - PulsePressure
  d["HeartRate"] = hr.mask(hr < 40, np.nan)     # two rows read 7 bpm
  d["PulsePressure"] = sbp - dbp
  d["ShockIndex"] = hr / sbp
  d["BPratio"] = sbp / dbp
  d["FracPP"] = (sbp - dbp) / ((sbp + 2 * dbp) / 3.0)
  return d


def engineer(X_train, y_train, X_test):
  return _feat(X_train), _feat(X_test)


def postprocess(pred, y_orig, X_test):
  if DUMP:
    try:
      import hashlib, os
      d = os.path.join(".dump", DUMP)
      os.makedirs(d, exist_ok=True)
      rid = (X_test["row_id"].to_numpy(dtype="int64") if "row_id" in X_test.columns
             else np.arange(len(X_test), dtype="int64"))
      np.savez(os.path.join(d, hashlib.sha1(rid.tobytes()).hexdigest()[:12] + ".npz"),
               p=pred.to_numpy(dtype="float64"), rid=rid,
               cols=np.array([str(c) for c in pred.columns]))
    except Exception:
      pass
  return pred
