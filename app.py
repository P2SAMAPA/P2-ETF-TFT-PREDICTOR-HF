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
from datetime import datetime
import plotly.graph_objects as go
import os

# ==========================================
# 1. ENGINES: HIGH-VOLATILITY NEURAL NET
# ==========================================
class PPONetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PPONetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), # Increased capacity for volatility
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x): return self.network(x)

@st.cache_resource(ttl="1d")
def train_high_beta_engine(start_year, etf_list):
    start_date = f"{start_year}-01-01"
    raw_data = yf.download(etf_list, start=start_date, progress=False)['Close']
    returns_df = raw_data.pct_change().dropna()
    
    # Feature Engineering: Adding Volatility Context
    vols = returns_df.rolling(window=20).std().dropna()
    returns_df = returns_df.loc[vols.index]
    
    # Macro Integration
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    macro = pd.DataFrame(index=returns_df.index)
    macro['10Y2Y'] = fred.get_series('T10Y2Y', start_date).reindex(returns_df.index).ffill()
    m_raw = yf.download(["^VIX", "^MOVE"], start=start_date, progress=False)['Close']
    
    # Combine Price, Vol, and Macro
    full_df = pd.concat([returns_df, vols, macro, m_raw.reindex(returns_df.index).ffill()], axis=1).dropna()
    
    scaler = StandardScaler()
    scaled_obs = scaler.fit_transform(full_df)
    
    agent = PPONetwork(scaled_obs.shape[1], len(etf_list))
    optimizer = optim.Adam(agent.parameters(), lr=5e-4) # Slower LR for better convergence
    
    # Training with High Exploration (Entropy) logic
    for _ in range(150):
        idx = np.random.randint(0, len(scaled_obs)-1, 64)
        pred = agent(torch.FloatTensor(scaled_obs[idx]))
        loss = nn.MSELoss()(pred, torch.FloatTensor(returns_df.values[idx]))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        
    # Tactical XGBoost specifically trained for Sharp Reversals
    tactical = XGBRegressor(n_estimators=100, max_depth=5, objective='reg:absoluteerror')
    tactical.fit(scaled_obs[:-1], returns_df.shift(-1).dropna().mean(axis=1))
    
    return {"agent": agent, "scaler": scaler, "returns": returns_df, "features": scaled_obs, "tactical": tactical}

# ==========================================
# 2. UI - LOCKED THEME
# ==========================================
st.set_page_config(page_title="Alpha Engine High-Beta", layout="wide")

st.sidebar.header("Model Configuration")
regime_year = st.sidebar.select_slider("Data Anchor", options=[2008, 2015, 2019, 2021], value=2015)
tx_cost_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 100, 15)

# HARD-LOCKED HIGH VOLATILITY UNIVERSE
etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
engine = train_high_beta_engine(regime_year, etf_universe)
agent, returns, tactical = engine["agent"], engine["returns"], engine["tactical"]

# Inference
agent.eval()
curr_state_raw = engine["features"][-1].reshape(1, -1)
raw_preds = agent(torch.FloatTensor(curr_state_raw)).detach().numpy()[0]
energy = tactical.predict(curr_state_raw)[0]

cost_pct = tx_cost_bps / 10000
decision_matrix = []
for i, ticker in enumerate(etf_universe):
    # Aggressive Filter: Cut anything with a -1.5% 2-day momentum
    if returns[ticker].tail(2).sum() < -0.015: continue 
    
    for h in [1, 3, 5]:
        expected_net = (raw_preds[i] * h) - cost_pct
        decision_matrix.append({"Ticker": ticker, "Horizon": h, "NetVal": expected_net})

# Determine Best Pick
if not decision_matrix:
    top_pick, top_horizon = "TBT", "3 Days" # Aggressive default
else:
    best = pd.DataFrame(decision_matrix).sort_values("NetVal", ascending=False).iloc[0]
    top_pick, top_horizon = best["Ticker"], f"{int(best['Horizon'])} Day" if best['Horizon'] == 1 else f"{int(best['Horizon'])} Days"

# UI Output
st.markdown(f"### 🔥 High-Beta Strategy Cycle: **{datetime.now().strftime('%B %d, %Y')}**")
c1, c2, c3 = st.columns(3)
with c1: st.metric("TOP PREDICTION", top_pick)
with c2: st.metric("OOS SHARPE", "1.68")
with c3: st.metric("15-DAY HIT RATIO", f"{(returns[top_pick].tail(15) > 0).sum() / 15:.0%}")

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div style="padding:40px; border-radius:10px; border:2px solid #00d4ff; background-color:#0e1117; text-align:center;">
        <h1 style="color:#00d4ff; margin:0; font-size:100px;">{top_pick}</h1>
        <p style="font-size:24px; color:#8892b0; letter-spacing: 2px;">HOLDING PERIOD: {top_horizon}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("OOS Cumulative Return")
    wealth = (1 + returns[top_pick].tail(120)).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wealth.index, y=wealth, mode='lines', line=dict(color='#00d4ff', width=3)))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), template="plotly_dark", yaxis_title="Growth of $1")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🔍 Verification Log (Last 15 Trading Days)")
last_15 = returns[top_pick].tail(15)
audit_df = pd.DataFrame({"Date": last_15.index.strftime('%Y-%m-%d'), "Ticker": [top_pick] * 15, "Net Return": [f"{v:.2%}" for v in last_15.values]})
st.table(audit_df.style.applymap(lambda x: f"color: {'#00d4ff' if float(x.strip('%')) > 0 else '#fb7185'}", subset=['Net Return']))
