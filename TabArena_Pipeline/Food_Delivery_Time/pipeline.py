import numpy as np
import pandas as pd

TABFM_KWARGS = {'permute_categorical': True, 'enable_nnls': True}


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _feat(df):
  out = pd.DataFrame(index=df.index)
  rlat = df['Restaurant_latitude'].astype(float).abs()
  dlat = df['Delivery_location_latitude'].astype(float).abs()
  rlon = df['Restaurant_longitude'].astype(float).abs()
  dlon = df['Delivery_location_longitude'].astype(float).abs()
  # the two deltas agree to 1e-6; average them and round the float noise off
  delta = ((dlat - rlat) + (dlon - rlon)).abs() / 2.0
  age = df['Delivery_person_Age'].astype(float)
  rat = df['Delivery_person_Ratings'].astype(float)

  out['age'] = age
  out['rating'] = rat
  out['delta'] = delta.round(2)
  out['vehicle'] = df['Type_of_vehicle'].fillna('').astype(str).str.strip()
  out['both'] = ((age == 29) & (rat == 4.6)).astype(np.int8)

  # step codes: the drivers are plateaus with hard edges, and the numeric normalisation TabFM
  # applies before attending smears an edge that a code states outright
  out['age_ge30'] = (age >= 30).astype(np.int8)
  out['rat_bucket'] = np.select([rat < 3.95, rat < 4.45], [0, 1], default=2).astype(np.int8)
  out['delta_bucket'] = np.select(
      [out['delta'] < 0.025, out['delta'] < 0.035, out['delta'] < 0.045, out['delta'] < 0.065],
      [0, 1, 2, 1], default=3).astype(np.int8)   # 0.04 (25.5 min) sits above 0.05/0.06 (23.9), 9 sigma
  out['veh_slow'] = out['vehicle'].isin(['motorcycle', 'bicycle']).astype(np.int8)

  # unordered views: neither is monotone (rating 5.0 -> 25.5 min against 4.8 -> 23.9, delta 0.04
  # above 0.05), so let TabFM keep neighbouring levels apart. The same view of `age` is *not* here:
  out['rating_s'] = rat.round(1).astype(str)
  out['delta_s'] = out['delta'].astype(str)

  # one coarse conjunction, 48 levels at ~630 rows each. Rows whose person record is imputed or
  # sentinel get their own level rather than being filed as young-and-highly-rated.
  unknown = (out['both'].to_numpy() == 1) | np.isin(age.to_numpy(), [15, 50])
  age_st = np.where(unknown, 'u', out['age_ge30'].astype(str))
  rat_st = np.where(unknown, 'u', out['rat_bucket'].astype(str))
  out['cell'] = (out['delta_bucket'].astype(str) + out['veh_slow'].astype(str)
                 + pd.Series(age_st, index=out.index) + pd.Series(rat_st, index=out.index))
  return out


def engineer(X_train, y_train, X_test):
  return _feat(X_train), _feat(X_test)


def postprocess(pred, y_orig, X_test):
  return pred
