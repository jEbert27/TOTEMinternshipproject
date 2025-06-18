#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Sparse + Δ_year + Δ_wk + rolling + season → 8-week direct forecasts via GBR

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

# === 0) SETTINGS ===
USE_LOG    = True        # work in log(I+1)
WINDOW_REC = 8           # number of recent lags to include
LAG_YEAR   = 52          # one-year lag
HORIZON    = 4           # forecast 8 weeks ahead
TRAIN_FRAC = 0.8         # train/validation split
LR         = 0.1         # learning rate for GBR
N_EST      = 200         # number of boosting stages
MAX_DEPTH  = 3           # max depth of each tree

os.makedirs("results", exist_ok=True)
np.random.seed(42)

# === 1) LOAD & PREPROCESS DATA ===
df = pd.read_csv("CDC_weekly_flu_US.csv", parse_dates=["week_ending_date"])
df = df.rename(columns={"week_ending_date": "date", "infected": "I"})
df = df[df["location_name"] == "New Mexico"].sort_values("date").reset_index(drop=True)

# Cast raw counts to float32
I_raw = df["I"].astype(np.float32).values
T     = len(df)

# Log‐transform if requested
if USE_LOG:
    df["I_proc"] = np.log(I_raw + 1.0).astype(np.float32)
else:
    df["I_proc"] = I_raw.copy()

# Seasonal features: sin/cos of week-of-year
woy = df["date"].dt.isocalendar().week.astype(np.float32)
df["sin_woy"] = np.sin(2 * np.pi * woy / 52.0).astype(np.float32)
df["cos_woy"] = np.cos(2 * np.pi * woy / 52.0).astype(np.float32)

# 4-week rolling average (in log‐space)
df["roll4_log"] = df["I_proc"].rolling(window=4, min_periods=1).mean().astype(np.float32)

# === 2) BUILD FEATURE MATRIX & TARGETS ===
# We will create a design matrix X of shape (n_samples, 14) where each row corresponds to
# a forecasting “anchor” at index j such that:
#   - The “current” week (t) we use for features is j + WINDOW_REC (i.e. end of recent block).
#   - We need j + WINDOW_REC - 1 >= LAG_YEAR   to have a valid year-lag in log.
#   - We need j + WINDOW_REC + (HORIZON) <= T     to have actual targets for all 8 horizons.
#
# Therefore, j goes from  LAG_YEAR  up to  T - HORIZON - WINDOW_REC  inclusive.

feature_list = []
targets_list = []  # we’ll collect 8-dimensional targets for each j

start_j = LAG_YEAR
end_j   = T - HORIZON - WINDOW_REC

# Loop over all valid training “anchors”
for j in range(start_j, end_j + 1):
    # “Current” week index (end of the recent WINDOW_REC block) = j + WINDOW_REC - 1
    t_cur = j + WINDOW_REC - 1

    # 2a) Last 8 log-counts: I_proc[t_cur - k], k = 0..7
    recent_block = df["I_proc"].values[t_cur - WINDOW_REC + 1 : t_cur + 1]  
    # shape (8,)

    # 2b) One-year ago log: I_proc[t_year] where t_year = t_cur - LAG_YEAR
    t_year = t_cur - LAG_YEAR
    lag_year_log = float(df["I_proc"].iat[t_year])

    # 2c) Seasonal at t_cur
    sin_cur = float(df["sin_woy"].iat[t_cur])
    cos_cur = float(df["cos_woy"].iat[t_cur])
#i dont know what the code does after this exact line \ god speed to chatgpt and its makers for they have achieved a level of hubris beyond what man intended.
    # 2d) 4-week rolling log at t_cur
    roll4 = float(df["roll4_log"].iat[t_cur])

    # 2e) Δ_year = I_proc[t_cur] - I_proc[t_cur - LAG_YEAR]
    delta_year = float(df["I_proc"].iat[t_cur] - df["I_proc"].iat[t_year])

    # 2f) Δ_wk   = I_proc[t_cur] - I_proc[t_cur - 1]
    delta_wk   = float(df["I_proc"].iat[t_cur] - df["I_proc"].iat[t_cur - 1])

    # Concatenate all 14 features:
    #   [recent_block (8 dims),
    #    lag_year_log (1 dim),
    #    sin_cur, cos_cur (2 dims),
    #    roll4 (1 dim),
    #    delta_year (1 dim),
    #    delta_wk (1 dim)]
    feat_vec = np.concatenate([
        recent_block,
        [lag_year_log, sin_cur, cos_cur, roll4, delta_year, delta_wk]
    ], axis=0).astype(np.float32)  # shape (14,)

    feature_list.append(feat_vec)

    # 2g) Build 8-dimensional target: next HORIZON values of I_proc
    #     at indices t_cur + 1, ..., t_cur + 8
    target_vec = df["I_proc"].values[t_cur + 1 : t_cur + 1 + HORIZON].astype(np.float32)
    # shape (8,)
    targets_list.append(target_vec)

# Convert to numpy arrays
X = np.stack(feature_list, axis=0)     # shape = (n_samples, 14)
Y = np.stack(targets_list, axis=0)     # shape = (n_samples,  8)

# === 3) TRAIN/VALID SPLIT ===
n_samples = X.shape[0]
n_train   = int(TRAIN_FRAC * n_samples)

X_tr, X_va = X[:n_train], X[n_train:]
Y_tr, Y_va = Y[:n_train], Y[n_train:]

# We’ll also need the index of the first “validation anchor” to later reconstruct t₀
j_va0 = start_j + n_train

# === 4) SCALE FEATURES ONLY ===
# (For GBR, we do not need to scale targets; but scaling features often helps.)
feat_scaler = MinMaxScaler().fit(X_tr)
X_tr_s = feat_scaler.transform(X_tr)
X_va_s = feat_scaler.transform(X_va)

# === 5) TRAIN ONE REGRESSOR PER HORIZON STEP ===
models = []
for h in range(HORIZON):
    # Train a GradientBoostingRegressor to predict Y[:, h]
    gbr = GradientBoostingRegressor(
        n_estimators=N_EST,
        learning_rate=LR,
        max_depth=MAX_DEPTH,
        random_state=42
    )
    gbr.fit(X_tr_s, Y_tr[:, h])
    models.append(gbr)

# === 6) FORECAST FROM t₀ = (j_va0 + WINDOW_REC - 1) ===
# The “anchor index” for first validation sample is j_va0.
# The corresponding “current week” t₀ = j_va0 + WINDOW_REC - 1.
t0 = j_va0 + WINDOW_REC - 1

# Build the feature vector exactly as above, but only for j = j_va0
t_cur = t0

recent_block = df["I_proc"].values[t_cur - WINDOW_REC + 1 : t_cur + 1]
t_year       = t_cur - LAG_YEAR
lag_year_log = float(df["I_proc"].iat[t_year])
sin_cur      = float(df["sin_woy"].iat[t_cur])
cos_cur      = float(df["cos_woy"].iat[t_cur])
roll4        = float(df["roll4_log"].iat[t_cur])
delta_year   = float(df["I_proc"].iat[t_cur] - df["I_proc"].iat[t_year])
delta_wk     = float(df["I_proc"].iat[t_cur] - df["I_proc"].iat[t_cur - 1])

feat_t0 = np.concatenate([
    recent_block,
    [lag_year_log, sin_cur, cos_cur, roll4, delta_year, delta_wk]
], axis=0).astype(np.float32)  # (14,)

feat_t0_s = feat_scaler.transform(feat_t0.reshape(1, -1))  # shape (1, 14)

# Predict log‐counts for horizons 1..8
Y_pred_log = np.zeros((1, HORIZON), dtype=np.float32)
for h in range(HORIZON):
    Y_pred_log[0, h] = models[h].predict(feat_t0_s)[0]

# === 7) INVERT LOG → COUNTS ===
if USE_LOG:
    I_pred_cnt = np.round(np.exp(Y_pred_log) - 1.0).astype(int).flatten()
else:
    I_pred_cnt = np.round(Y_pred_log).astype(int).flatten()

# Gather the actual counts for t₀+1 ... t₀+8
I_actual = I_raw[t0 + 1 : t0 + 1 + HORIZON].astype(int)  # shape (8,)

# Build the 8-week forecast table
rows = []
for h in range(HORIZON):
    week_idx    = t0 + 1 + h
    wk_date     = df["date"].iat[week_idx]
    wk_str      = wk_date.strftime("%Y-%m-%d")
    actual_cnt  = int(I_actual[h])
    predicted   = int(I_pred_cnt[h])
    pct_err     = abs(actual_cnt - predicted) / (actual_cnt if actual_cnt != 0 else 1) * 100

    rows.append({
        "prediction_week": wk_str,
        "horizon_week":    h + 1,
        "actual_count":    actual_cnt,
        "predicted_count": predicted,
        "pct_error":       f"{pct_err:.2f}%"
    })

single_forecast = pd.DataFrame(rows)
print(f"\nBoosted‐trees direct 8-week forecast (t₀={t0}):")
print(single_forecast.to_string(index=False))

# === 8) MAX ERROR & OVERALL ACCURACY ===
pct_vals = single_forecast["pct_error"].str.rstrip("%").astype(float)
idx_max  = pct_vals.idxmax()
max_wk   = single_forecast.loc[idx_max, "prediction_week"]
max_err  = abs(single_forecast.loc[idx_max, "actual_count"] -
               single_forecast.loc[idx_max, "predicted_count"])
accuracy = 100 - pct_vals.mean()

print(f"\nMaximum absolute error: {max_err} on {max_wk}")
print(f"Average accuracy over {HORIZON} weeks: {accuracy:.2f}%")






