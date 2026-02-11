import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import torch
import torch.nn as nn
import torch.optim as optim
import os
import interface 
import time

# ==========================================
# 1. ANTI-THROTTLE DATA ENGINE
# ==========================================
@st.cache_data(ttl="6h") # Caches data for 6 hours to prevent yfinance bans
def get_safe_data(tickers, start):
    try:
        # auto_adjust=True and threads=False reduces 'bot-like' behavior
        data = yf.download(tickers, start=start, auto_adjust=True, threads=False, progress=False)
        if data.empty:
            st.error("Yahoo rate-limited your IP. Try again in 1 hour or use a VPN.")
            return None
        return data['Close']
    except Exception as e:
        st.error(f"Data Connection Error: {e}")
        return None

def rolling_z_score(df, window=60):
    """Suggestion #3: Rolling Z-Score Scaling for Tactical Sensitivity"""
    return (df - df.rolling(window).mean()) / (df.rolling(window).std() + 1e-9)

def add_indicators(df, tickers):
    for t in tickers:
        # RSI
        delta = df[t].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df[f'{t}_RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    return df.dropna()

# ==========================================
# 2. PPO ARCHITECTURE
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
def train_engine(etf_list):
    start_date = "2008-01-01"
    # Dynamic Training Boundary: Stops 6 months ago
    train_end = (pd.Timestamp.now() - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
    
    raw_close = get_safe_data(etf_list, start_date)
    if raw_close is None: return None
    
    returns_df = raw_close.ffill().pct_change().dropna()
    feat_df = add_indicators(returns_df.copy(), etf_list)
    
    # Feature Engineering: Apply Rolling Z-Score
    scaled_feats = rolling_z_score(feat_df).dropna()
    
    # Align training sets
    train_data = scaled_feats.loc[:train_end]
    target_rets = returns_df.loc[train_data.index]
    
    agent = PPONetwork(train_data.shape[1], len(etf_list))
    optimizer = optim.Adam(agent.parameters(), lr=1e-4)
    
    # PPO Training Loop
    for _ in range(500):
        idx = np.random.randint(0, len(train_data)-1, 64)
        batch_x = torch.FloatTensor(train_data.values[idx])
        batch_y = torch.FloatTensor(target_rets.values[idx])
        
        loss = nn.MSELoss()(agent(batch_x), batch_y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    return {
        "agent": agent, 
        "returns": returns_df, 
        "features": scaled_feats, 
        "train_info": {"start": start_date, "end": train_end}
    }

# ==========================================
# 3. UI EXECUTION
# ==========================================
st.set_page_config(layout="wide")
etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
engine = train_engine(etf_universe)

if engine:
    agent, returns, features = engine["agent"], engine["returns"], engine["features"]
    
    # OOS ANALYSIS: Exactly the last 60 trading days
    oos_rets = returns.tail(60)
    oos_wealth = (1 + oos_rets).cumprod()
    
    # Prediction
    agent.eval()
    last_state = torch.FloatTensor(features.tail(1).values)
    preds = agent(last_state).detach().numpy()[0]
    
    # Tactical RSI Check
    for i, t in enumerate(etf_universe):
        if features[f'{t}_RSI'].iloc[-1] > 70: preds[i] *= 0.8 # De-prioritize peaks
        
    top_pick = etf_universe[np.argmax(preds)]

    # Metrics (60-Day Annualization Math)
    total_oos_ret = oos_wealth[top_pick].iloc[-1] - 1
    # Corrected formula: (1+R)^(252/60) - 1
    ann_ret_val = ((1 + total_oos_ret) ** (252 / 60)) - 1
    
    daily_rets = oos_rets[top_pick]
    sharpe_val = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() != 0 else 0

    audit_df = pd.DataFrame({
        "Date": returns.tail(45).index.strftime('%Y-%m-%d'),
        "Ticker": [top_pick]*45,
        "Net Return": [f"{v:.2%}" for v in returns[top_pick].tail(45).values]
    })

    interface.render_main_output(
        top_pick, 
        f"{sharpe_val:.2f}", 
        (daily_rets > 0).mean(), 
        f"{ann_ret_val:.2%}", 
        "5-Day Tactical", 
        oos_wealth[top_pick], 
        audit_df, 
        engine["train_info"]
    )
