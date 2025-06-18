#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generate synthetic mobility_US.csv and weather_US.csv

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1) Settings
states = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming", "Puerto Rico", "Guam",
    "American Samoa", "Northern Mariana Islands", "U.S. Virgin Islands"
]

start_date = datetime(2018, 1, 1)
end_date   = datetime(2025, 6, 1)

# build list of weekly dates
dates = []
d = start_date
while d <= end_date:
    dates.append(d)
    d += timedelta(days=7)

# 2) Generate mobility data
mobility_rows = []
for state in states:
    # a linear trend plus seasonality plus noise
    trend    = np.linspace(-20, 20, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 52.0)
    noise    = np.random.normal(0, 5, len(dates))
    values   = trend + seasonal + noise
    for dt, v in zip(dates, values):
        mobility_rows.append({
            "location_name": state,
            "date":           dt.strftime("%Y-%m-%d"),
            "mobility_index": round(v, 2)
        })

df_mob = pd.DataFrame(mobility_rows)
df_mob.to_csv("mobility_US.csv", index=False)
print("Generated mobility_US.csv with", len(df_mob), "rows.")

# 3) Generate weather data
weather_rows = []
for state in states:
    # temp: seasonally between ~5–35°C; hum: ~30–70%
    seasonal_temp = 20 + 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 52.0)
    seasonal_hum  = 50 + 20 * np.cos(2 * np.pi * np.arange(len(dates)) / 52.0)
    noise_t = np.random.normal(0, 3, len(dates))
    noise_h = np.random.normal(0, 5, len(dates))
    temps = seasonal_temp + noise_t
    hums  = seasonal_hum  + noise_h
    for dt, t, h in zip(dates, temps, hums):
        weather_rows.append({
            "location_name": state,
            "date":          dt.strftime("%Y-%m-%d"),
            "temp_avg":      round(t, 1),
            "hum_avg":       round(h, 1)
        })

df_weather = pd.DataFrame(weather_rows)
df_weather.to_csv("weather_US.csv", index=False)
print("Generated weather_US.csv with", len(df_weather), "rows.")

