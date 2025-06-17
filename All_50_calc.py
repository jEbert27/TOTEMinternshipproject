#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 4-Week Flu Forecast with Optional Exogenous Signals, TCN, Quantile Outputs, Ensembling & State Ranking
# + Blend with naïve baseline to tighten accuracy spread

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv1D, GlobalAveragePooling1D, Dense, Dropout, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Loss

# ========================
# 0) SETTINGS & POPULATIONS
# ========================
USE_LOG     = True
TRAIN_FRAC  = 0.8
HORIZON     = 4
GAMMA       = 1.0/2.0
BATCH_SIZE  = 32
EPOCHS      = 100
ENSEMBLE_N  = 5
SEQ_LEN     = 8
PATIENCE    = 20
LOW_THRESH  = 5

# **blend factor**: 0 = pure naive, 1 = pure model
BLEND_ALPHA = 0.6  

STATE_POP = {
    "Alabama":4903185,   "Alaska":731545,   "Arizona":7278717,  "Arkansas":3017804,
    "California":39512223,"Colorado":5758736,"Connecticut":3565287,"Delaware":973764,
    "Florida":21477737,  "Georgia":10617423, "Hawaii":1415872,   "Idaho":1787065,
    "Illinois":12671821,"Indiana":6732219,  "Iowa":3155070,     "Kansas":2913314,
    "Kentucky":4467673, "Louisiana":4648794, "Maine":1344212,    "Maryland":6045680,
    "Massachusetts":6892503,"Michigan":9986857,"Minnesota":5639632,"Mississippi":2976149,
    "Missouri":6137428, "Montana":1068778,  "Nebraska":1934408,  "Nevada":3080156,
    "New Hampshire":1359711,"New Jersey":8882190,"New Mexico":2096829,"New York":19453561,
    "North Carolina":10488084,"North Dakota":762062,"Ohio":11689100,"Oklahoma":3956971,
    "Oregon":4217737,   "Pennsylvania":12801989,"Rhode Island":1059361,
    "South Carolina":5148714,"South Dakota":884659,"Tennessee":6833174,
    "Texas":28995881,   "Utah":3205958,    "Vermont":623989,    "Virginia":8535519,
    "Washington":7614893,"West Virginia":1792147,"Wisconsin":5822434,"Wyoming":578759,
    "Puerto Rico":3193694, "Guam":168775,
    "American Samoa":55197, "Northern Mariana Islands":57557,
    "U.S. Virgin Islands":106235
}
max_pop = max(STATE_POP.values())

# ========================
# 1) LOAD & PREPROCESS
# ========================
df_flu = pd.read_csv("CDC_weekly_flu_US.csv", parse_dates=["week_ending_date"])
df_flu.rename(columns={"week_ending_date":"date","infected":"I"}, inplace=True)

# exogenous fallback
for fname in ("weather_US.csv","mobility_US.csv"):
    if not os.path.exists(fname):
        print(f"⚠️  Warning: '{fname}' not found; filling exogenous with zeros.")

df_weather = pd.read_csv("weather_US.csv", parse_dates=["date"]) \
    if os.path.exists("weather_US.csv") else pd.DataFrame(columns=["location_name","date","temp_avg","hum_avg"])
df_mob     = pd.read_csv("mobility_US.csv", parse_dates=["date"]) \
    if os.path.exists("mobility_US.csv") else pd.DataFrame(columns=["location_name","date","mobility_index"])

df_all = (df_flu
    .merge(df_weather, on=["location_name","date"], how="left")
    .merge(df_mob,     on=["location_name","date"], how="left")
    .fillna({"temp_avg":0.0,"hum_avg":0.0,"mobility_index":0.0})
)

# ========================
# 2) QUANTILE LOSS
# ========================
class QuantileLoss(Loss):
    def __init__(self, q):
        super().__init__()
        self.q = q
    def call(self, y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(self.q*e, (self.q-1)*e))

# ========================
# 3) BUILD FEATURES & SEQUENCES
# ========================
def build_sequences(df, pop):
    df = df.sort_values("date").reset_index(drop=True)
    I = df["I"].astype(np.float32).values
    df["I_log"]   = np.log(I+1.0) if USE_LOG else I
    df["roll4"]   = df["I_log"].rolling(4,1).mean()
    df["low_inc"] = (I<LOW_THRESH).astype(np.float32)
    woy = df["date"].dt.isocalendar().week.astype(np.float32)
    df["sin_woy"] = np.sin(2*np.pi * woy/52.0)
    df["cos_woy"] = np.cos(2*np.pi * woy/52.0)
    for col in ("temp_avg","hum_avg","mobility_index"):
        m,s = df[col].mean(), df[col].std() or 1.0
        df[col+"_n"] = (df[col]-m)/s
    S = np.zeros_like(I); R = np.zeros_like(I)
    S[0] = pop - I[0]
    for t in range(1,len(I)):
        R[t] = R[t-1] + GAMMA*I[t-1]
        S[t] = pop - I[t] - R[t]
    X, y = [], []
    for t in range(SEQ_LEN, len(df)):
        win = []
        for k in range(SEQ_LEN,0,-1):
            i = t-k
            win.append([
                df.I_log.iat[i], df.roll4.iat[i], df.low_inc.iat[i],
                df.sin_woy.iat[i], df.cos_woy.iat[i],
                df.temp_avg_n.iat[i], df.hum_avg_n.iat[i], df.mobility_index_n.iat[i],
                S[i]/pop, I[i]/pop, R[i]/pop
            ])
        X.append(win); y.append(df.I_log.iat[t])
    return np.array(X,np.float32), np.array(y,np.float32)

# prepare train/val
Xs, ys = [], []
for loc,pop in STATE_POP.items():
    df_loc = df_all[df_all.location_name==loc]
    if len(df_loc) < SEQ_LEN+HORIZON: continue
    Xi, yi = build_sequences(df_loc,pop)
    Xs.append(Xi); ys.append(yi)
X = np.concatenate(Xs,axis=0); y = np.concatenate(ys,axis=0)
idx = np.random.permutation(len(X))
split = int(TRAIN_FRAC*len(X))
tr, va = idx[:split], idx[split:]
X_tr, y_tr = X[tr], y[tr]
X_va, y_va = X[va], y[va]

# ========================
# 4) MODEL DEFINITION
# ========================
def build_tcn_quantile_model(input_shape, hidden_dim=64, dropout=0.1, l2reg=1e-5):
    inp = Input(shape=input_shape)
    x   = Conv1D(hidden_dim,3,padding="causal",dilation_rate=1,kernel_regularizer=l2(l2reg))(inp)
    x   = Conv1D(hidden_dim,3,padding="causal",dilation_rate=2,kernel_regularizer=l2(l2reg))(x)
    x   = LayerNormalization()(x)
    x   = GlobalAveragePooling1D()(x)
    x   = Dropout(dropout)(x)
    x   = Dense(hidden_dim,activation="relu",kernel_regularizer=l2(l2reg))(x)
    return Model(inp, [
        Dense(1,name="q10")(x),
        Dense(1,name="q50")(x),
        Dense(1,name="q90")(x)
    ])

# ========================
# 5) TRAIN & ENSEMBLE
# ========================
ensemble_models = []
for seed in range(ENSEMBLE_N):
    tf.random.set_seed(42+seed); np.random.seed(42+seed)
    m = build_tcn_quantile_model(input_shape=X_tr.shape[1:])
    m.compile(optimizer=Adam(1e-3),
              loss={"q10":QuantileLoss(0.1),
                    "q50":QuantileLoss(0.5),
                    "q90":QuantileLoss(0.9)},
              loss_weights={"q10":1,"q50":1,"q90":1})
    cb = EarlyStopping(monitor="val_loss",patience=PATIENCE,restore_best_weights=True)
    m.fit(X_tr,{"q10":y_tr,"q50":y_tr,"q90":y_tr},
          validation_data=(X_va,{"q10":y_va,"q50":y_va,"q90":y_va}),
          epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=[cb],verbose=1)
    ensemble_models.append(m)

# ========================
# 6) FORECAST & RANK ALL 50 STATES + TERRITORIES
# ========================
def forecast_for_loc(df_loc, pop):
    df = df_loc.sort_values("date").reset_index(drop=True)
    I_raw = df["I"].values.astype(np.float32)
    X_loc,_ = build_sequences(df_loc,pop)
    window = [row.copy() for row in X_loc[-1]]
    preds = []
    for h in range(HORIZON):
        x_in = np.array([window],dtype=np.float32)
        qs = [m.predict(x_in,verbose=0)[1].ravel()[0] for m in ensemble_models]
        q50 = np.mean(qs)
        I_pred = np.exp(q50)-1.0
        # blend with last observed incidence
        naive = I_raw[-1]
        blended = BLEND_ALPHA*I_pred + (1-BLEND_ALPHA)*naive
        preds.append(blended)
        # roll forward window as before...
        window.pop(0)
        new = window[-1].copy()
        new_I_log = np.log(blended+1.0)
        new[0], new[1], new[2] = new_I_log, \
            np.mean([w[0] for w in window[-3:]]+[new_I_log]), \
            1.0 if blended<LOW_THRESH else 0.0
        Sprev = window[-1][8]*pop; Iprev = window[-1][9]*pop
        Rprev = pop-Sprev-Iprev
        Rnew  = Rprev+GAMMA*Iprev; Snew = pop-blended-Rnew
        new[8],new[9],new[10] = Snew/pop, blended/pop, Rnew/pop
        window.append(new)
    actuals = I_raw[-HORIZON:]
    errs = [0.0 if a==0 else min(abs(a-p)/a*100,100)
            for a,p in zip(actuals,preds)]
    return 100.0-np.mean(errs)

scores = []
for loc,pop in STATE_POP.items():
    df_loc = df_all[df_all.location_name==loc]
    if len(df_loc)<SEQ_LEN+HORIZON: continue
    scores.append((loc,forecast_for_loc(df_loc,pop)))

scores.sort(key=lambda x:x[1],reverse=True)
print("\nOverall Ranking (States + Territories) by 4-Week Forecast Accuracy:")
for i,(loc,acc) in enumerate(scores,1):
    print(f"{i:2d}. {loc:<30s} → {acc:6.2f}%")

