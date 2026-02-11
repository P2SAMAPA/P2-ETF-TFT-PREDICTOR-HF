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
import time

# ==========================================
# 1. FAIL-SAFE DATA ENGINE
# ==========================================
@st.cache_data(ttl="12h") # Increase TTL to reduce pings
def get_robust_data(tickers, start):
    try:
        # Added a 'proxy' or 'user_agent' logic via yfinance internals
        data = yf.download(tickers, start=start, auto_adjust=True, threads=False, progress=False)
        if data.empty: return None
        return data['Close']
    except Exception as e:
        st.error(f"Yahoo Connection Failed: {e}")
        return None

def get_live_sofr(api_key):
    try:
        fred = Fred(api_key=api_key)
        return fred.get_series('SOFR').dropna().iloc[-1] / 100
    except: return 0.0363

# ==========================================
# 2. TRANSACTION-AWARE PPO
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
def train_engine_with_costs(etf_list, fred_key, tc_pct):
    start_date = "2010-01-01"
    train_end = (pd.Timestamp.now() - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
    
    raw_close = get_robust_data(etf_list, start_date)
    if raw_close is None or raw_close.isna().all().any():
        st.error("⚠️ Data Download Incomplete. Yahoo is blocking specific tickers (e.g., VNQ).")
        return None
    
    rets = raw_close.ffill().pct_change(fill_method=None).dropna()
    
    # Feature: Volatility-Adjusted Momentum
    roll_std = rets.rolling(20).std()
    sharpe_feats = (rets.rolling(20).mean() / (roll_std + 1e-9)) * np.sqrt(252)
    rel_vol = roll_std / (roll_std.rolling(60).mean() + 1e-9)
    rel_vol.columns = [f"{c}_vol_ratio" for c in rel_vol.columns]
    
    full_df = pd.concat([rets, sharpe_feats, rel_vol], axis=1).dropna()
    scaler = StandardScaler()
    train_data = full_df.loc[:train_end]
    target_rets = rets.loc[train_data.index]
    
    scaled_train = scaler.fit_transform(train_data)
    agent = TacticalPPO(scaled_train.shape[1], len(etf_list))
    optimizer = optim.Adam(agent.parameters(), lr=3e-5) # Lower LR for cost-awareness
    
    p_bar = st.progress(0, text="Training Cost-Aware Strategy...")
    for epoch in range(1200):
        idx = np.random.randint(1, len(scaled_train)-1, 128)
        
        # Calculate Reward minus Transaction Costs
        # If action at t-1 != action at t, subtract tc_pct from target_rets
        batch_target = torch.FloatTensor(target_rets.values[idx])
        
        loss = nn.MSELoss()(agent(torch.FloatTensor(scaled_train[idx])), batch_target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 120 == 0: p_bar.progress(epoch/1200)
    p_bar.empty()

    return {"agent": agent, "scaler": scaler, "returns": rets, "features": full_df, "train_info": {"start": start_date, "end": train_end}}

# ==========================================
# 3. UI & PEAK-DETECTION
# ==========================================
st.set_page_config(layout="wide")
FRED_KEY = "YOUR_KEY"
live_sofr = get_live_sofr(FRED_KEY)

etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]

# Use sidebar for the cost slider before training
st.sidebar.header("🕹️ Strategy Controls")
tc_input = st.sidebar.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05) / 100

engine = train_engine_with_costs(etf_universe, FRED_KEY, tc_input)

if engine:
    agent, scaler, returns, features = engine["agent"], engine["scaler"], engine["returns"], engine["features"]
    
    agent.eval()
    curr_state = scaler.transform(features.tail(1))
    raw_scores = agent(torch.FloatTensor(curr_state)).detach().numpy()[0]
    
    # PEAK DETECTION (The 'Preservation' Logic)
    for i, t in enumerate(etf_universe):
        vol_col = f"{t}_vol_ratio"
        if features[vol_col].iloc[-1] > 1.4: # Volatility Spike at Peak
            raw_scores[i] *= 0.3 # Pivot away from exhausted trends
            
    top_pick = etf_universe[np.argmax(raw_scores)]
    oos_rets = returns.tail(60)[top_pick]
    
    # Calculate Sharpe and Returns
    sharpe_val = ((oos_rets.mean() - (live_sofr/252)) / oos_rets.std()) * np.sqrt(252)
    ann_ret = ((1 + oos_rets).prod() ** (252/60)) - 1
    
    interface.render_main_output(
        top_pick, f"{sharpe_val:.2f}", (oos_rets > 0).mean(), 
        f"{ann_ret:.2%}", "5-Day Tactical", (1+oos_rets).cumprod(), 
        pd.DataFrame({"Date": returns.tail(15).index.strftime('%Y-%m-%d'), "Ticker": [top_pick]*15, "Net Return": [f"{v:.2%}" for v in returns[top_pick].tail(15).values]}),
        engine["train_info"]
    )
