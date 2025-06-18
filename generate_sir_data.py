#!/usr/bin/env python3
import numpy as np
import pandas as pd

def simulate_sir(beta, gamma, S0, I0, R0, days, dt=1.0):
    N = S0 + I0 + R0
    S, I, R = S0, I0, R0
    out = []
    for t in range(int(days/dt)):
        out.append((t*dt, S, I, R))
        # SIR differential equations (Euler)
        dS = -beta * S * I / N * dt
        dI = (beta * S * I / N - gamma * I) * dt
        dR = gamma * I * dt
        S += dS
        I += dI
        R += dR
    df = pd.DataFrame(out, columns=['day','S','I','R'])
    return df

def add_random_noise(df, noise_fraction=0.2, noise_scale=0.05, random_seed=None):
    """
    Introduce multiplicative noise to (S,I,R) at random intervals.
    
    Parameters:
    - df: DataFrame with columns ['day','S','I','R']
    - noise_fraction: fraction of days to perturb (e.g., 0.2 = 20%)
    - noise_scale: relative standard deviation for noise (e.g., 0.05 = 5% noise)
    - random_seed: int or None for reproducibility
    
    For each randomly chosen day, we multiply each of S, I, R by (1 + epsilon),
    where epsilon ~ Normal(0, noise_scale), then re‐normalize so S+I+R remains constant.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = len(df)
    num_noisy = int(np.floor(noise_fraction * n))
    noisy_indices = np.random.choice(n, size=num_noisy, replace=False)
    
    for idx in noisy_indices:
        s, i, r = df.loc[idx, ['S','I','R']].values
        
        # Draw independent relative noise for each compartment
        eps_s = np.random.normal(loc=0.0, scale=noise_scale)
        eps_i = np.random.normal(loc=0.0, scale=noise_scale)
        eps_r = np.random.normal(loc=0.0, scale=noise_scale)
        
        s_noisy = s * (1 + eps_s)
        i_noisy = i * (1 + eps_i)
        r_noisy = r * (1 + eps_r)
        
        # Prevent negative values
        s_noisy = max(s_noisy, 0.0)
        i_noisy = max(i_noisy, 0.0)
        r_noisy = max(r_noisy, 0.0)
        
        # Re‐normalize so that s_noisy + i_noisy + r_noisy = s + i + r
        original_sum = s + i + r
        noisy_sum = s_noisy + i_noisy + r_noisy
        if noisy_sum > 0:
            factor = original_sum / noisy_sum
            s_noisy *= factor
            i_noisy *= factor
            r_noisy *= factor
        else:
            s_noisy, i_noisy, r_noisy = s, i, r
        
        df.at[idx, 'S'] = s_noisy
        df.at[idx, 'I'] = i_noisy
        df.at[idx, 'R'] = r_noisy

if __name__ == "__main__":
    # parameters
    beta, gamma = 0.3, 0.1       # infection / recovery rates
    S0, I0, R0 = 0.99, 0.01, 0.0  # initial fractions
    days = 200
    
    # 1) Generate the “true” SIR trajectory
    df = simulate_sir(beta, gamma, S0, I0, R0, days)
    
    # 2) Inject stronger noise at random intervals (20% of days, 5% relative noise)
    add_random_noise(df, noise_fraction=0.2, noise_scale=0.05, random_seed=42)
    
    # 3) Convert day index to actual dates and reorder columns
    df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    df = df[['date','S','I','R']]
    
    # 4) Save to CSV
    df.to_csv("sir_data.csv", index=False)
    print("Saved more-noisy synthetic SIR data to sir_data.csv")

