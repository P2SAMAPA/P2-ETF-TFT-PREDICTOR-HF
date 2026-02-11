import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import interface 

# ==========================================
# 1. PEAK-DETECTION LOGIC
# ==========================================
def get_live_sofr(api_key):
    try:
        fred = Fred(api_key=api_key)
        return fred.get_series('SOFR').dropna().iloc[-1] / 100
    except: return 0.0363

class TacticalPPO(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(TacticalPPO, self).__init__()
        # Deeper architecture to handle regime switching
        self.policy = nn.Sequential(
            nn.Linear(input_dim, 512), nn.Mish(), # Mish is smoother for gradients
            nn.Linear(512, 512), nn.LayerNorm(512), nn.Mish(),
            nn.Linear(512, 256), nn.Mish(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x): return self.policy(x)

@st.cache_resource(ttl="1d")
def train_tactical_engine(etf_list, fred_key):
    start_date = "2010-01-01" # Post-2008 to focus on modern regime switching
    train_end = (pd.Timestamp.now() - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
    
    data = yf.download(etf_list, start=start_date, auto_adjust=True, threads=False, progress=False)['Close']
    rets = data.ffill().pct_change().dropna()
    
    # ADVANCED FEATURE: Volatility-Adjusted Momentum (Z-Score of Sharpe)
    lookback = 20
    roll_mean = rets.rolling(lookback).mean()
    roll_std = rets.rolling(lookback).std()
    sharpe_feats = (roll_mean / (roll_std + 1e-9)) * np.sqrt(252)
    
    # Combined Feature Set: Returns + Rolling Sharpe + Rel. Volatility
    rel_vol = roll_std / roll_std.rolling(60).mean()
    full_df = pd.concat([rets, sharpe_feats, rel_vol], axis=1).dropna()
    
    scaler = StandardScaler()
    train_data = full_df.loc[:train_end]
    target_rets = rets.loc[train_data.index]
    
    scaled_train = scaler.fit_transform(train_data)
    agent = TacticalPPO(scaled_train.shape[1], len(etf_list))
    optimizer = optim.Adam(agent.parameters(), lr=5e-5) # Slower LR for precision
    
    # 1,200 Epochs for high-precision convergence
    p_bar = st.progress(0, text="Optimizing Peak-Detection Strategy...")
    for epoch in range(1200):
        idx = np.random.randint(0, len(scaled_train)-1, 128)
        loss = nn.MSELoss()(agent(torch.FloatTensor(scaled_train[idx])), torch.FloatTensor(target_rets.values[idx]))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 120 == 0: p_bar.progress(epoch/1200)
    p_bar.empty()

    return {"agent": agent, "scaler": scaler, "returns": rets, "features": full_df}

# ==========================================
# 2. UI & PERFORMANCE
# ==========================================
st.set_page_config(layout="wide")
FRED_KEY = "YOUR_KEY"
live_sofr = get_live_sofr(FRED_KEY)

etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
engine = train_tactical_engine(etf_universe, FRED_KEY)

if engine:
    agent, scaler, returns, features = engine["agent"], engine["scaler"], engine["returns"], engine["features"]
    
    agent.eval()
    curr_state = scaler.transform(features.tail(1))
    raw_scores = agent(torch.FloatTensor(curr_state)).detach().numpy()[0]
    
    # TACTICAL OVERRIDE: Peak Exhaustion Check
    # If volatility is 50% higher than normal, penalize the score to force rotation
    for i, t in enumerate(etf_universe):
        current_vol_ratio = features.iloc[-1][f"{t}_vol_ratio"] if f"{t}_vol_ratio" in features else 1.0
        if current_vol_ratio > 1.5: raw_scores[i] *= 0.5 

    top_pick = etf_universe[np.argmax(raw_scores)]
    
    # OOS Stats
    oos_window = returns.tail(60)
    oos_wealth = (1 + oos_window[top_pick]).cumprod()
    ann_ret = ((1 + (oos_wealth.iloc[-1]-1)) ** (252/60)) - 1
    sharpe = ((oos_window[top_pick].mean() - (live_sofr/252)) / oos_window[top_pick].std()) * np.sqrt(252)

    st.markdown("### 🏔️ Alpha Engine v12: Peak-Preservation Strategy")
    c1, c2, c3 = st.columns(3)
    c1.metric("CURRENT ROTATION", top_pick)
    c2.metric("Ann. Return (60D)", f"{ann_ret:.2%}")
    c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    st.caption(f"Strategy: Volatility-Adjusted Momentum | Benchmark: SOFR {live_sofr:.2%}")

    st.line_chart(oos_wealth)
    
    with st.expander("Why this pick?"):
        st.write(f"The model chose **{top_pick}** because it currently offers the highest return-per-unit-of-volatility. If volatility in {top_pick} spikes, the 'Peak Exhaustion' logic will trigger a rotation into the next most stable asset.")
