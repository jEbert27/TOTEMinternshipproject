#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SIR + Neural‐Net β(t) → 8‐Week Flu Forecast for New Mexico

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# ========================
# 0) SETTINGS
# ========================
USE_LOG    = True           # Train on log(I+1)
TRAIN_FRAC  = 0.8           # Train/validation split fraction
HORIZON     = 4             # Forecast 4 weeks ahead
POP_NM      = 7_446_000     # Approximate population of New Mexico
GAMMA       = 1.0 / 2.0     # Recovery rate (per week)
LR          = 1e-4          # Learning rate for NN
BATCH_SIZE  = 32
EPOCHS      = 100
HIDDEN_DIM  = 16            # Hidden units in β‐network
BETA_MAX    = 1.0           # Upper bound on β_t
PATIENCE    = 10            # EarlyStopping patience

os.makedirs("results", exist_ok=True)
tf.random.set_seed(42)
np.random.seed(42)

# ========================
# 1) LOAD & PREPROCESS DATA
# ========================
df = pd.read_csv("CDC_weekly_flu_US.csv", parse_dates=["week_ending_date"])
df = df.rename(columns={"week_ending_date": "date", "infected": "I"})
df = df[df["location_name"] == "New Mexico"].sort_values("date").reset_index(drop=True)

# Extract raw I counts
I_raw = df["I"].astype(np.float32).values
T     = len(df)

# Optional log‐transform
if USE_LOG:
    df["I_proc"] = np.log(I_raw + 1.0).astype(np.float32)
else:
    df["I_proc"] = I_raw.copy()

# Compute weekly seasonality features
woy = df["date"].dt.isocalendar().week.astype(np.float32)
df["sin_woy"] = np.sin(2.0 * np.pi * woy / 52.0).astype(np.float32)
df["cos_woy"] = np.cos(2.0 * np.pi * woy / 52.0).astype(np.float32)

# ========================
# 2) RECONSTRUCT SIR STATES FROM OBSERVED I
# ========================
S = np.zeros(T, dtype=np.float32)
I = I_raw.copy()
R = np.zeros(T, dtype=np.float32)

# Initialize at t=0
R[0] = 0.0
S[0] = POP_NM - I[0]

# Use discrete‐time approximation:
for t in range(1, T):
    R[t] = R[t - 1] + GAMMA * I[t - 1]
    S[t] = POP_NM - I[t] - R[t]

# ========================
# 3) BUILD (FEATURE, TARGET) PAIRS FOR β‐NETWORK
#    We train the NN to predict I_proc(t+1) from features at time t
# ========================
feature_list = []
target_list  = []
S_list       = []
I_list       = []

# We can only start at t=1 (so that I[t-1] exists) and end at t=T-2 (so that I[t+1] exists)
for t in range(1, T - 1):
    # Build feature vector at time t:
    #   x_t = [log I_proc(t), log I_proc(t-1), sin_woy(t), cos_woy(t)]
    x_t = np.array([
        df["I_proc"].iat[t],
        df["I_proc"].iat[t - 1],
        df["sin_woy"].iat[t],
        df["cos_woy"].iat[t]
    ], dtype=np.float32)

    # Target: log(I_proc(t+1)) if USE_LOG, else I_proc(t+1)
    y_t1 = df["I_proc"].iat[t + 1]

    feature_list.append(x_t)
    target_list.append(y_t1)
    S_list.append(S[t])
    I_list.append(I[t])

features = np.stack(feature_list, axis=0)  # shape = (T-2, 4)
targets  = np.stack(target_list,  axis=0)  # shape = (T-2,)
S_arr    = np.stack(S_list,       axis=0)  # shape = (T-2,)
I_arr    = np.stack(I_list,       axis=0)  # shape = (T-2,)

# ========================
# 4) TRAIN/VALID SPLIT
# ========================
n_samples = features.shape[0]
n_train   = int(TRAIN_FRAC * n_samples)

X_tr, X_va = features[:n_train], features[n_train:]
Y_tr, Y_va = targets[:n_train],  targets[n_train:]
S_tr, S_va = S_arr[:n_train],     S_arr[n_train:]
I_tr, I_va = I_arr[:n_train],     I_arr[n_train:]

# ========================
# 5) DEFINE β‐NETWORK
# ========================
class BetaNet(Model):
    def __init__(self, hidden_dim):
        super().__init__()
        self.d1  = Dense(hidden_dim, activation="relu")
        self.d2  = Dense(hidden_dim, activation="relu")
        self.out = Dense(1, activation="softplus")  # ensures β_t ≥ 0

    def call(self, x, training=False):
        h = self.d1(x)
        h = self.d2(h)
        return tf.squeeze(self.out(h), axis=-1)  # shape = (batch,)

beta_model = BetaNet(hidden_dim=HIDDEN_DIM)
optimizer  = Adam(learning_rate=LR)
mse_loss   = tf.keras.losses.MeanSquaredError()

# ========================
# 6) TRAINING STEP (ONE STEP‐AHEAD SIR)
# ========================
@tf.function
def train_step(x_b, s_b, i_b, y_b):
    with tf.GradientTape() as tape:
        # 1) Predict β_t
        beta_pred = beta_model(x_b, training=True)  # shape = (batch,)

        # 2) Clip β_t to [0, BETA_MAX]
        beta_clipped = tf.clip_by_value(beta_pred, 0.0, BETA_MAX)

        # 3) One‐step SIR Euler in count space
        N = tf.cast(POP_NM, tf.float32)
        s = s_b
        i = i_b
        b = beta_clipped
        infection = b * s * i / N
        i_next = i + infection - GAMMA * i
        i_next_clipped = tf.maximum(i_next, 0.0)

        # 4) Convert to log‐space for loss
        if USE_LOG:
            y_pred = tf.math.log(i_next_clipped + 1.0)
        else:
            y_pred = i_next_clipped

        loss_val = mse_loss(y_b, y_pred)

    grads = tape.gradient(loss_val, beta_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, beta_model.trainable_variables))
    return loss_val

# ========================
# 7) TRAINING LOOP
# ========================
train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_tr, S_tr, I_tr, Y_tr))
    .shuffle(1024, seed=42)
    .batch(BATCH_SIZE)
)
val_dataset = tf.data.Dataset.from_tensor_slices((X_va, S_va, I_va, Y_va)).batch(BATCH_SIZE)

best_val_loss = np.inf
patience_cnt  = 0

for epoch in range(1, EPOCHS + 1):
    train_losses = []
    for x_b, s_b, i_b, y_b in train_dataset:
        l = train_step(x_b, s_b, i_b, y_b)
        train_losses.append(l.numpy())
    train_loss = np.mean(train_losses)

    # Validation pass
    val_losses = []
    for x_b, s_b, i_b, y_b in val_dataset:
        beta_pred = beta_model(x_b, training=False)
        beta_clipped = tf.clip_by_value(beta_pred, 0.0, BETA_MAX)
        N = tf.cast(POP_NM, tf.float32)
        s = s_b
        i = i_b
        b = beta_clipped
        infection = b * s * i / N
        i_next = i + infection - GAMMA * i
        i_next_clipped = tf.maximum(i_next, 0.0)
        if USE_LOG:
            y_pred = tf.math.log(i_next_clipped + 1.0)
        else:
            y_pred = i_next_clipped
        val_losses.append(mse_loss(y_b, y_pred).numpy())
    val_loss = np.mean(val_losses)

    # EarlyStopping
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        patience_cnt = 0
        beta_model.save_weights("results/beta_model_best.ckpt")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"Stopping at epoch {epoch}, val_loss plateaued.")
            break

    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:03d} → train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")

# Restore best weights
beta_model.load_weights("results/beta_model_best.ckpt")

# ========================
# 8) RECURSIVE 8‐WEEK FORECAST
# ========================
# Choose t0 so that t0 + HORIZON <= T - 1
t0 = T - HORIZON - 1
if t0 < 1:
    raise ValueError("Not enough data to forecast out HORIZON weeks.")

# Initialize S, I, R at t0
S_curr = S[t0]
I_curr = I_raw[t0]
R_curr = R[t0]

# Also need “previous log” and seasonal at t0
I_log_prev = float(df["I_proc"].iat[t0 - 1])
sin_prev   = float(df["sin_woy"].iat[t0])
cos_prev   = float(df["cos_woy"].iat[t0])

I_preds_cnt = np.zeros(HORIZON, dtype=np.float32)
I_preds_log = np.zeros(HORIZON, dtype=np.float32)

for h in range(HORIZON):
    # Build feature vector at week t = t0 + h
    if USE_LOG:
        x_h = np.array([float(np.log(I_curr + 1.0)), I_log_prev, sin_prev, cos_prev], dtype=np.float32)
    else:
        x_h = np.array([I_curr, I_log_prev, sin_prev, cos_prev], dtype=np.float32)
    x_h = x_h.reshape(1, -1)  # (1, 4)

    # Predict β_t and clip
    beta_h = float(beta_model(x_h, training=False).numpy()[0])
    beta_h = min(max(beta_h, 0.0), BETA_MAX)

    # SIR step using true POP_NM
    N = float(POP_NM)
    infection = beta_h * S_curr * I_curr / N
    I_next    = I_curr + infection - GAMMA * I_curr
    I_next    = max(I_next, 0.0)

    I_preds_cnt[h] = I_next
    if USE_LOG:
        I_preds_log[h] = float(np.log(I_next + 1.0))
    else:
        I_preds_log[h] = I_next

    R_next = R_curr + GAMMA * I_curr
    S_next = N - I_next - R_next

    # Update for next iteration
    I_log_prev = float(np.log(I_curr + 1.0)) if USE_LOG else I_curr
    woy_new    = float((int(woy.iat[t0]) + (h + 1) - 1) % 52 + 1)
    sin_prev   = float(np.sin(2.0 * np.pi * woy_new / 52.0))
    cos_prev   = float(np.cos(2.0 * np.pi * woy_new / 52.0))

    S_curr = S_next
    I_curr = I_next
    R_curr = R_next

# ========================
# 9) PRINT & SAVE RESULTS
# ========================
rows = []
for h in range(HORIZON):
    wk        = pd.to_datetime(df["date"].iat[t0]) + pd.Timedelta(weeks=h + 1)
    wk_str    = wk.strftime("%Y-%m-%d")
    actual    = None
    idx       = t0 + 1 + h
    if idx < T:
        actual = int(I_raw[idx])
    pred      = int(round(I_preds_cnt[h]))
    pct_str   = "N/A"
    if actual is not None:
        pct       = abs(actual - pred) / (actual if actual != 0 else 1) * 100
        pct_str   = f"{pct:.2f}%"
    rows.append({
        "prediction_week": wk_str,
        "horizon_week":    h + 1,
        "actual_count":    actual if actual is not None else "-",
        "predicted_count": pred,
        "pct_error":       pct_str
    })

df_forecast = pd.DataFrame(rows)
print(f"\nRecursive SIR+NN‐β forecast (t₀={t0}, HORIZON={HORIZON}):")
print(df_forecast.to_string(index=False))

df_forecast.to_csv("results/sample0_8week_sir_nn_beta.csv", index=False)
print("\nSaved → results/sample0_8week_sir_nn_beta.csv")




