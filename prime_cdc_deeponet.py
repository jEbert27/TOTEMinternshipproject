#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === 0) Settings ===
# We will keep a 4-week forecast horizon (weekly predictions).
WINDOW        = 12       # use past 14 weeks of data to predict the next 4 weeks
HORIZON       = 4           # forecast 4 weeks ahead
ENSEMBLE_SZ   = 3           # use a small ensemble
LEARNING_RATE = 1e-3
BATCH_SIZE    = 16
EPOCHS        = 25

os.makedirs("results", exist_ok=True)

# === 1) Load & preprocess data ===

df_all = pd.read_csv("CDC_weekly_flu_US.csv")

# Filter to the state of interest (e.g. New Mexico).
location_of_interest = "New Mexico"
df = df_all[df_all['location_name'] == location_of_interest].copy()

# Parse dates and drop NaN “infected” rows
df['date'] = pd.to_datetime(df['week_ending_date'], format="%Y-%m-%d")
df.rename(columns={'infected': 'I'}, inplace=True)
df = df.dropna(subset=['I']).copy()

# Sort chronologically
df.sort_values('date', inplace=True)
df.reset_index(drop=True, inplace=True)

# === 1a) Feature engineering ===
# 1. Log-transform I: y_target = log1p(I_future)
df['I_log1p'] = np.log1p(df['I'])

# 2. Compute weekly change dI and log-transform it (keeping the sign)
df['dI'] = df['I'].diff().fillna(0)
df['dI_log1p'] = np.log1p(df['dI'].abs()) * np.sign(df['dI'])

# 3. Day-of-year seasonality (sine and cosine)
doy = df['date'].dt.dayofyear.astype(float)
df['sin_doy'] = np.sin(2 * np.pi * doy / 365.0)
df['cos_doy'] = np.cos(2 * np.pi * doy / 365.0)

# Final feature list: ['I_log1p', 'dI_log1p', 'sin_doy', 'cos_doy']
features = ['I_log1p', 'dI_log1p', 'sin_doy', 'cos_doy']

# === 1b) Build sliding windows for X and y ===
X_list, y_list = [], []
for j in range(len(df) - WINDOW - HORIZON + 1):
    # X: shape = (WINDOW, n_features)
    X_block = df.loc[j : j + WINDOW - 1, features].values
    X_list.append(X_block)
    # y: next HORIZON weeks of raw I, but we store log1p(I) for each of those 4 weeks
    future_I = df.loc[j + WINDOW : j + WINDOW + HORIZON - 1, 'I'].values
    y_list.append(np.log1p(future_I))  # shape = (HORIZON,)

X = np.stack(X_list)     # shape = (n_samples, WINDOW, 4)
y = np.stack(y_list)     # shape = (n_samples, HORIZON)

# Train/validation split (80/20)
n_samples = X.shape[0]
n_train = int(0.8 * n_samples)
X_tr, X_va = X[:n_train], X[n_train:]
y_tr, y_va = y[:n_train], y[n_train:]

print(f"Total samples: {n_samples}")
print(f"Training samples: {len(X_tr)}  —  Validation samples: {len(X_va)}")

# === 2) Define a small LSTM model that outputs HORIZON weeks ahead ===
class WeeklyLSTM(Model):
    def __init__(self):
        super().__init__()
        # Encoder: LSTM with 16 units
        self.encoder = LSTM(16, activation='tanh', dropout=0.2, recurrent_dropout=0.2)
        # Output layer: Dense(HORIZON) to produce log1p(I) for each of the next 4 weeks
        self.out_fc = Dense(HORIZON, activation='linear')

    def call(self, inputs, training=None):
        # inputs.shape = (batch, WINDOW, n_features)
        h = self.encoder(inputs, training=training)    # (batch, 16)
        return self.out_fc(h)                          # (batch, HORIZON)

# === 3) Train an ensemble of models ===
all_val_preds = []
trained_models = []

for seed in range(ENSEMBLE_SZ):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = WeeklyLSTM()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='mae',           # mean absolute error on log1p(I)
        metrics=['mae']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    trained_models.append(model)
    val_pred = model.predict(X_va)  # shape = (n_val, HORIZON)
    all_val_preds.append(val_pred)

# Ensemble-average validation predictions in log1p-space
y_va_pred_log = np.mean(np.stack(all_val_preds, axis=0), axis=0)  # (n_val, HORIZON)

# Invert log1p to raw I for both predictions and ground truth
y_va_pred = np.expm1(y_va_pred_log)     # (n_val, HORIZON)
y_va_true = np.expm1(y_va)              # (n_val, HORIZON)

# Compute validation metrics (MAE and MAPE) on raw I
mae_val = np.mean(np.abs(y_va_true - y_va_pred))
mape_val = np.mean(
    np.abs((y_va_true - y_va_pred) / np.maximum(y_va_true, 1))
) * 100

print(f"\nValidation MAE (raw I): {mae_val:.2f} cases")
print(f"Validation MAPE (raw I): {mape_val:.2f}%\n")

# === 4) Single 4-week forecast from a chosen origin ===
# Make sure j_origin <= len(df) - WINDOW - HORIZON
j_origin = 50

# 4a) Grab the last WINDOW weeks to form the “latest block”
latest_block = df.loc[j_origin : j_origin + WINDOW - 1, features].values  # (WINDOW, 4)
latest_block = latest_block.reshape((1, WINDOW, len(features)))           # (1, WINDOW, 4)

# 4b) Have each model predict log1p(I) for the next 4 weeks, then average
all_future_logs = []
for model in trained_models:
    pred_log = model.predict(latest_block)[0]   # shape = (HORIZON,)
    all_future_logs.append(pred_log)

pred_log_avg = np.mean(np.stack(all_future_logs, axis=0), axis=0)  # (HORIZON,)
pred_I_future = np.expm1(pred_log_avg)                            # (HORIZON,) in raw I

# 4c) Compare against actual I for those 4 weeks
forecast_rows = []
for h in range(HORIZON):
    week_idx    = j_origin + WINDOW + h
    pred_date   = df.loc[week_idx, 'date']
    actual_I    = df.loc[week_idx, 'I']
    pred_I      = pred_I_future[h]
    if actual_I != 0:
        pct_err = abs(actual_I - pred_I) / actual_I * 100
        pct_err_str = f"{pct_err:.2f}%"
    else:
        pct_err_str = None

    forecast_rows.append({
        'horizon_week':    h + 1,
        'prediction_date': pred_date.strftime("%Y-%m-%d"),
        'actual_count':    int(actual_I),
        'predicted_count': float(np.round(pred_I, 1)),
        'pct_error':       pct_err_str
    })

forecast_df = pd.DataFrame(forecast_rows)

origin_date_str = df.loc[j_origin + WINDOW - 1, 'date'].strftime("%Y-%m-%d")
last_pred_str   = df.loc[j_origin + WINDOW + HORIZON - 1, 'date'].strftime("%Y-%m-%d")
print(f"Single 4-week forecast (from origin {origin_date_str} → {last_pred_str}):")
print(forecast_df.to_string(index=False))

# 4d) Save to CSV
forecast_df.to_csv("results/4-week_single_forecast.csv", index=False)
print(f"\nState of {location_of_interest} - 4-week forecast saved → results/4-week_single_forecast.csv")


