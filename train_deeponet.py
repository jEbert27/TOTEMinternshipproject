#!/usr/bin/env python3 
#average acuracy over 28 days: 74.52%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam

# === 0) Settings ===
TOTAL_POP    = 3_000_000   # total population
WINDOW       = 14          # history window in days
HORIZON      = 28          # forecast 28 days ahead
ENSEMBLE_SZ  = 5           # number of models in ensemble
GAMMA        = 0.1
LEARNING_RATE= 1e-3
BATCH_SIZE   = 32
EPOCHS       = 50

os.makedirs("results", exist_ok=True)

# === 1) Load & rescale data ===
df = pd.read_csv("sir_data.csv", parse_dates=['date'])
df[['S','I','R']] *= TOTAL_POP  # convert fractions → raw counts

# build sliding windows
X, y, idxs = [], [], []
for j in range(len(df) - WINDOW - HORIZON + 1):
    X.append(df[['S','I','R']].values[j:j+WINDOW])
    y.append(df['I'].values[j+WINDOW:j+WINDOW+HORIZON])
    idxs.append(j)
X = np.stack(X)  # shape (n_samples, WINDOW, 3)
y = np.stack(y)  # shape (n_samples, HORIZON)

# === 2) Scale features & targets to [0,1] ===
feat_scaler = MinMaxScaler().fit(X.reshape(-1,3))
Xs = feat_scaler.transform(X.reshape(-1,3)).reshape(X.shape)
tgt_scaler = MinMaxScaler().fit(y)
ys = tgt_scaler.transform(y)

# === 3) Train/validation split ===
n_train = int(0.8 * len(Xs))
X_tr, X_va = Xs[:n_train], Xs[n_train:]
y_tr, y_va = ys[:n_train], ys[n_train:]
idx_va     = idxs[n_train:]

# === 4) Weighted‐horizon loss ===
def weighted_mse(y_true, y_pred):
    # weights ramp from 1.0 on day1 up to 2.0 on day28
    w = tf.linspace(1.0, 2.0, HORIZON)
    return tf.reduce_mean(w * tf.square(y_true - y_pred))

# === 5) Define the Neural-β SIR model ===
class SIRBetaNet(Model):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        self.beta_net = tf.keras.Sequential([
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1,  activation='softplus'),
        ])
    def call(self, inputs, training=None, true_I=None):
        s = inputs[:, -1, 0]
        i = inputs[:, -1, 1]
        r = inputs[:, -1, 2]
        preds = []
        for _ in range(HORIZON):
            x      = tf.stack([s, i, r], axis=1)    # shape (batch,3)
            beta_t = tf.squeeze(self.beta_net(x), 1)
            N      = s + i + r
            new_inf = beta_t * s * i / N
            new_rec = self.gamma * i
            s = s - new_inf
            i_pred = i + new_inf - new_rec
             # if training and true_I is provided, use it
            if training and true_I is not None:
                i = true_I[:, t]
            else:
                i = i_pred
            r = r + new_rec
            preds.append(i)
        return tf.stack(preds, axis=1)  # shape (batch, HORIZON)

# === 6) Ensemble training & prediction ===
all_preds = []
for seed in range(ENSEMBLE_SZ):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = SIRBetaNet(GAMMA)
    model.compile(optimizer=Adam(LEARNING_RATE), loss=weighted_mse)
    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        verbose=0
    )
    pred_s = model.predict(X_va)            # scaled preds, shape (n_val, HORIZON)
    all_preds.append(pred_s)

# average ensemble predictions
y_pred_s = np.mean(all_preds, axis=0)       # shape (n_val, HORIZON)
y_pred   = tgt_scaler.inverse_transform(y_pred_s)
y_true   = tgt_scaler.inverse_transform(y_va)

# === 7) Build & display single-sample forecast ===
# choose sample_idx = 0 from validation
j0 = idx_va[0]
rows = []
for h in range(HORIZON):
    date_pred = df['date'].iloc[j0 + WINDOW + h]
    actual    = y_true[0, h]
    pred      = y_pred[0, h]
    pct_err   = abs(actual - pred) / actual * 100 if actual != 0 else 0
    rows.append({
        'prediction_date': date_pred.strftime("%Y-%m-%d"),
        'horizon_day':     h+1,
        'actual_count':    int(actual),
        'predicted_count': int(pred),
        'pct_error':       f"{pct_err:.2f}%"
    })
single = pd.DataFrame(rows)

print(f"\nEnsemble forecast for sample_idx=0 over {HORIZON} days:")
print(single.to_string(index=False))

# === 8) Max error & overall accuracy ===
pct_vals = single['pct_error'].str.rstrip('%').astype(float)
idx_max  = pct_vals.idxmax()
max_date = single.loc[idx_max, 'prediction_date']
max_err  = abs(single.loc[idx_max, 'actual_count'] - single.loc[idx_max, 'predicted_count'])
accuracy = 100 - pct_vals.mean()

print(f"\nMaximum absolute error: {max_err:,} on {max_date}")
print(f"Average accuracy over {HORIZON} days: {accuracy:.2f}%")

# === 9) Save results ===
single.to_csv("results/sample0_28day_ensemble_forecast.csv", index=False)
print("\nSaved → results/sample0_28day_ensemble_forecast.csv")


