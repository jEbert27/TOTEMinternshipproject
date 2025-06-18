##!/usr/bin/env python3
#average accuracy over 28 days: ~91.71%

import os
import numpy as np
import pandas as pd
import tensorflow as tf

# === 0) Settings ===
TOTAL_POP     = 3_000_000   # total population
WINDOW        = 14          # history window in days
HORIZON       = 28          # forecast 28 days ahead
ENSEMBLE_SZ   = 5           # number of models in ensemble
GAMMA         = 0.1         # we’ll keep gamma fixed for now
LEARNING_RATE = 1e-3
BATCH_SIZE    = 32
EPOCHS        = 50

os.makedirs("results", exist_ok=True)

# === 1) Load & rescale data ===
df = pd.read_csv("sir_data.csv", parse_dates=['date'])
df[['S','I','R']] *= TOTAL_POP  # convert fractions → raw counts

# ===== Compute and append seasonal sin/cos features =====
doy = df['date'].dt.dayofyear.astype(float)
df['sin_doy'] = np.sin(2 * np.pi * doy / 365.0)
df['cos_doy'] = np.cos(2 * np.pi * doy / 365.0)

# === 1a) Compute delta‐I as a new column ===
df['dI'] = df['I'].diff().fillna(0)

# === 1b) Build sliding windows with (S, I, R, dI) ===
X, y, idxs = [], [], []
for j in range(len(df) - WINDOW - HORIZON + 1):
    # Now include the new 'dI' feature as the 4th column
    X.append(df[['S','I','R','dI']].values[j:j+WINDOW])      # shape (WINDOW, 4)
    y.append(df['I'].values[j+WINDOW : j+WINDOW+HORIZON])   # unchanged
    idxs.append(j)
X = np.stack(X)  # shape (n_samples, WINDOW, 4)
y = np.stack(y)  # shape (n_samples, HORIZON)

# === 2) Scale 4 features now ===
from sklearn.preprocessing import MinMaxScaler
feat_scaler = MinMaxScaler().fit(X.reshape(-1, 4))
Xs = feat_scaler.transform(X.reshape(-1, 4)).reshape(X.shape)

tgt_scaler = MinMaxScaler().fit(y)
ys = tgt_scaler.transform(y)

# (The rest—train/val split, etc.—stays the same.)


# build sliding windows exactly as before 
X, y, idxs = [], [], []
for j in range(len(df) - WINDOW - HORIZON + 1):
    X.append(df[['S','I','R']].values[j:j+WINDOW])      # shape (WINDOW, 3)
    y.append(df['I'].values[j+WINDOW:j+WINDOW+HORIZON])  # shape (HORIZON,)
    idxs.append(j)
X = np.stack(X)  # shape (n_samples, WINDOW, 3)
y = np.stack(y)  # shape (n_samples, HORIZON)

# === 2) Scale features & targets to [0,1] ===
from sklearn.preprocessing import MinMaxScaler

# Scale each of S, I, R separately over the entire dataset
feat_scaler = MinMaxScaler().fit(X.reshape(-1, 3))
Xs = feat_scaler.transform(X.reshape(-1, 3)).reshape(X.shape)

tgt_scaler = MinMaxScaler().fit(y)
ys = tgt_scaler.transform(y)

# === 3) Train/validation split ===
n_train = int(0.8 * len(Xs))
X_tr, X_va = Xs[:n_train], Xs[n_train:]
y_tr, y_va = ys[:n_train], ys[n_train:]
idx_va     = idxs[n_train:]

# === 4) Weighted‐horizon loss (unchanged) ===
def weighted_mse(y_true, y_pred):
    # weights ramp from 1.0 on day1 up to 2.0 on day28
    w = tf.linspace(1.0, 2.0, HORIZON)
    return tf.reduce_mean(w * tf.square(y_true - y_pred))

# === 5) Define the new SIR‐LSTM model ===
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout

class SIRBetaLSTM(Model):
    def __init__(self, gamma):
        super().__init__()
        # ================================================
        # Make log‐gamma trainable instead of fixed:
        self.log_gamma = tf.Variable(tf.math.log(gamma),
                                     trainable=True,
                                     name='log_gamma')
        # ================================================
        self.encoder = LSTM(32, activation='tanh', name='encoder_lstm')
        self.beta_net = tf.keras.Sequential([
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='softplus'),
        ], name='beta_mlp')

    @property
    def gamma(self):
        # Convert log‐gamma back to positive γ
        return tf.exp(self.log_gamma)

    def call(self, inputs, training=None, true_I=None):
        h_enc = self.encoder(inputs, training=training)
        s = inputs[:, -1, 0]
        i = inputs[:, -1, 1]
        r = inputs[:, -1, 2]

        preds = []
        for t in range(HORIZON):
            x_net = tf.concat([h_enc, tf.stack([s, i, r], axis=1)], axis=1)
            beta_t = tf.squeeze(self.beta_net(x_net, training=training), 1)

            N = s + i + r + 1e-6
            new_inf = beta_t * s * i / N

            # Use trainable gamma now (instead of a fixed 0.1)
            new_rec = self.gamma * i

            s = s - new_inf
            i_pred = i + new_inf - new_rec
            r = r + new_rec

            s = tf.maximum(s, 0.0)
            i = tf.maximum(i_pred, 0.0)
            r = tf.maximum(r, 0.0)

            preds.append(i)

        return tf.stack(preds, axis=1)

# === 6) Ensemble training & prediction ===
all_preds = []
for seed in range(ENSEMBLE_SZ):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Instantiate a fresh model for each seed
    model = SIRBetaLSTM(GAMMA)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=weighted_mse,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
    )

    # Fit for up to EPOCHS epochs
    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Predict on the *scaled* validation set
    pred_s = model.predict(X_va)  # shape (n_val, HORIZON)
    all_preds.append(pred_s)

# === 7) Average ensemble predictions & invert scaling ===
y_pred_s = np.mean(all_preds, axis=0)        # shape (n_val, HORIZON)
y_pred   = tgt_scaler.inverse_transform(y_pred_s)
y_true   = tgt_scaler.inverse_transform(y_va)

# === 8) Build & display single‐sample forecast (unchanged) ===
# choose sample_idx = 0 from validation
j0 = idx_va[0]
rows = []
for h in range(HORIZON):
    date_pred = df['date'].iloc[j0 + WINDOW + h]
    actual    = y_true[0, h]
    pred      = y_pred[0, h]
    pct_err   = abs(actual - pred) / (actual if actual != 0 else 1) * 100
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

# === 9) Max error & overall accuracy ===
pct_vals = single['pct_error'].str.rstrip('%').astype(float)
idx_max  = pct_vals.idxmax()
max_date = single.loc[idx_max, 'prediction_date']
max_err  = abs(single.loc[idx_max, 'actual_count'] - single.loc[idx_max, 'predicted_count'])
accuracy = 100 - pct_vals.mean()

print(f"\nMaximum absolute error: {max_err:,} on {max_date}")
print(f"Average accuracy over {HORIZON} days: {accuracy:.2f}%")

# === 10) Save results ===
single.to_csv("results/sample0_28day_ensemble_forecast.csv", index=False)
print("\nSaved → results/sample0_28day_ensemble_forecast.csv")



