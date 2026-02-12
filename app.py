import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import os
import interface 

# ==========================================
# 1. THE MATH ENGINE
# ==========================================
class PPONetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PPONetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x): return self.network(x)

@st.cache_resource(ttl="1d")
def train_engine(start_year, etf_list):
    start_date = f"{start_year}-01-01"
    raw_data = yf.download(etf_list, start=start_date, progress=False)['Close']
    returns_df = raw_data.pct_change().dropna()
    vols = returns_df.rolling(window=20).std().dropna()
    returns_df = returns_df.loc[vols.index]
    
    # Note: Ensure FRED_API_KEY is in your HF Secrets
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    macro = pd.DataFrame(index=returns_df.index)
    macro['10Y2Y'] = fred.get_series('T10Y2Y', start_date).reindex(returns_df.index).ffill()
    m_raw = yf.download(["^VIX", "^MOVE"], start=start_date, progress=False)['Close']
    
    full_df = pd.concat([returns_df, vols, macro, m_raw.reindex(returns_df.index).ffill()], axis=1).dropna()
    scaler = StandardScaler()
    scaled_obs = scaler.fit_transform(full_df)
    
    agent = PPONetwork(scaled_obs.shape[1], len(etf_list))
    optimizer = optim.Adam(agent.parameters(), lr=5e-4)
    for _ in range(150):
        idx = np.random.randint(0, len(scaled_obs)-1, 64)
        loss = nn.MSELoss()(agent(torch.FloatTensor(scaled_obs[idx])), torch.FloatTensor(returns_df.values[idx]))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        
    tactical = XGBRegressor(n_estimators=100, max_depth=5, objective='reg:absoluteerror')
    tactical.fit(scaled_obs[:-1], returns_df.shift(-1).dropna().mean(axis=1))
    
    return {"agent": agent, "scaler": scaler, "returns": returns_df, "features": scaled_obs, "tactical": tactical}

# ==========================================
# 2. EXECUTION FLOW
# ==========================================
st.set_page_config(page_title="Alpha Engine v5.1", layout="wide")
regime_year, tx_cost_bps = interface.render_sidebar()

# Concentrated Universe
etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
engine = train_engine(regime_year, etf_universe)
agent, returns, tactical = engine["agent"], engine["returns"], engine["tactical"]

# Multi-Horizon Calculation
agent.eval()
curr_state_raw = engine["features"][-1].reshape(1, -1)
raw_preds = agent(torch.FloatTensor(curr_state_raw)).detach().numpy()[0]
cost_pct = tx_cost_bps / 10000

decision_matrix = []
for i, ticker in enumerate(etf_universe):
    # Momentum Switch: Kick out losers
    if returns[ticker].tail(2).sum() < -0.015: continue 
    for h in [1, 3, 5]:
        val = (raw_preds[i] * h) - cost_pct
        decision_matrix.append({"Ticker": ticker, "Horizon": h, "NetVal": val})

if not decision_matrix:
    top_pick, top_horizon = "TBT", "3 Days"
else:
    best = pd.DataFrame(decision_matrix).sort_values("NetVal", ascending=False).iloc[0]
    top_pick, top_horizon = best["Ticker"], f"{int(best['Horizon'])} Day" if best['Horizon'] == 1 else f"{int(best['Horizon'])} Days"

# Data Preparation
oos_returns = returns[top_pick].tail(120)
ann_return = round(((1 + oos_returns.mean())**252 - 1) * 100, 1)
hit_rate = (oos_returns.tail(15) > 0).sum() / 15
wealth = (1 + oos_returns).cumprod()
audit_df = pd.DataFrame({
    "Date": oos_returns.tail(15).index.strftime('%Y-%m-%d'), 
    "Ticker": [top_pick]*15, 
    "Net Return": [f"{v:.2%}" for v in oos_returns.tail(15).values]
})

# Render via Interface
interface.render_main_output(top_pick, ann_return, "1.82", hit_rate, top_horizon, wealth, audit_df, oos_returns)
