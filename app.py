import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. ROBUST DATA ENGINE (ANTI-CRASH)
# ==========================================
@st.cache_data(ttl="6h")
def get_safe_market_data(tickers, start):
    try:
        data = yf.download(tickers, start=start, auto_adjust=True, threads=False, progress=False)
        if data.empty or (len(data) < 20):
            return None
        # Handle the Multi-Index "Close" column in recent yfinance updates
        return data['Close']
    except Exception:
        return None

def get_live_sofr(api_key):
    try:
        fred = Fred(api_key=api_key)
        return fred.get_series('SOFR').dropna().iloc[-1] / 100
    except: return 0.0363

# ==========================================
# 2. THE TACTICAL PPO (MISH ACTIVATION)
# ==========================================
class TacticalPPO(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(TacticalPPO, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, 512), nn.Mish(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.Mish(),
            nn.Linear(512, 256), nn.Mish(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x): return self.policy(x)

@st.cache_resource(ttl="1d")
def train_tactical_engine(etf_list, fred_key):
    start_date = "2010-01-01"
    train_end = (pd.Timestamp.now() - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
    
    raw_close = get_safe_market_data(etf_list, start_date)
    
    # CRASH PROTECTION: Exit early if data is empty
    if raw_close is None:
        st.error("⚠️ Yahoo Finance is currently rate-limiting this app. Please wait 15-30 minutes.")
        st.info("Try toggling a VPN or checking back later this morning.")
        return None
    
    rets = raw_close.ffill().pct_change().dropna()
    
    # Feature Engineering: Volatility-Adjusted Momentum
    lookback = 20
    roll_mean = rets.rolling(lookback).mean()
    roll_std = rets.rolling(lookback).std()
    sharpe_feats = (roll_mean / (roll_std + 1e-9)) * np.sqrt(252)
    
    # Rel Vol: Detects if current volatility is higher than historical normal
    rel_vol = roll_std / (roll_std.rolling(60).mean() + 1e-9)
    # Rename columns for clear filtering later
    rel_vol.columns = [f"{c}_vol_ratio" for c in rel_vol.columns]
    
    full_df = pd.concat([rets, sharpe_feats, rel_vol], axis=1).dropna()
    
    # SAFETY CHECK: Ensure full_df isn't empty after indicator calculation
    if full_df.empty:
        st.warning("Insufficient history for indicators. Yahoo returned truncated data.")
        return None

    scaler = StandardScaler()
    train_data = full_df.loc[:train_end]
    target_rets = rets.loc[train_data.index]
    
    if len(train_data) == 0:
        st.error("Training window alignment failed. Check system dates.")
        return None

    scaled_train = scaler.fit_transform(train_data)
    agent = TacticalPPO(scaled_train.shape[1], len(etf_list))
    optimizer = optim.Adam(agent.parameters(), lr=5e-5)
    
    p_bar = st.progress(0, text="Optimizing Peak-Detection Strategy...")
    for epoch in range(1200):
        idx = np.random.randint(0, len(scaled_train)-1, 128)
        loss = nn.MSELoss()(agent(torch.FloatTensor(scaled_train[idx])), torch.FloatTensor(target_rets.values[idx]))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 120 == 0: p_bar.progress(epoch/1200)
    p_bar.empty()

    return {"agent": agent, "scaler": scaler, "returns": rets, "features": full_df, "train_info": {"start": start_date, "end": train_end}}

# ==========================================
# 3. UI RENDER
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
    
    # PEAK DETECTION OVERRIDE
    for i, t in enumerate(etf_universe):
        vol_col = f"{t}_vol_ratio"
        if vol_col in features.columns:
            # If current volatility is 50% above its 60-day average, slash its score
            if features[vol_col].iloc[-1] > 1.5:
                raw_scores[i] *= 0.4 

    top_pick = etf_universe[np.argmax(raw_scores)]
    oos_window = returns.tail(60)[top_pick]
    oos_wealth = (1 + oos_window).cumprod()
    
    st.markdown("### 🏔️ Alpha Engine v12.1: Peak-Preservation Strategy")
    c1, c2, c3 = st.columns(3)
    c1.metric("TOP PICK", top_pick)
    c2.metric("Annualised (60D)", f"{((oos_wealth.iloc[-1])**(252/60)-1):.2%}")
    c3.metric("Sharpe Ratio", f"{((oos_window.mean() - (live_sofr/252)) / oos_window.std() * np.sqrt(252)):.2f}")
    st.caption(f"Benchmark: SOFR {live_sofr:.2%}")

    st.line_chart(oos_wealth)
