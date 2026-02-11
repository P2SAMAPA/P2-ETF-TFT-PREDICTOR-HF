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

# [PPONetwork remains same]
class PPONetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PPONetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x): return self.network(x)

def rolling_z_score(df, window=60):
    return (df - df.rolling(window).mean()) / df.rolling(window).std()

def add_indicators(df, tickers):
    for t in tickers:
        delta = df[t].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df[f'{t}_RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    return df.dropna()

@st.cache_resource(ttl="1d")
def train_engine(etf_list):
    start_date = "2004-01-01"
    raw_data = yf.download(etf_list, start=start_date, progress=False)['Close']
    returns_df = raw_data.ffill().pct_change().dropna()
    feat_df = add_indicators(returns_df.copy(), etf_list)
    
    # Feature Engineering: Rolling Z-Score
    scaled_feats = rolling_z_score(feat_df).dropna()
    
    # TRAINING BOUNDARIES
    train_start = "2008-01-01"
    train_end = (pd.Timestamp.now() - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
    
    train_data = scaled_feats.loc[train_start:train_end]
    target_rets = returns_df.loc[train_data.index]
    
    agent = PPONetwork(train_data.shape[1], len(etf_list))
    optimizer = optim.Adam(agent.parameters(), lr=1e-4)
    for _ in range(500):
        idx = np.random.randint(0, len(train_data)-1, 64)
        loss = nn.MSELoss()(agent(torch.FloatTensor(train_data.values[idx])), torch.FloatTensor(target_rets.values[idx]))
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    return {"agent": agent, "returns": returns_df, "features": scaled_feats, "train_info": {"start": train_start, "end": train_end}}

st.set_page_config(layout="wide")
etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
engine = train_engine(etf_universe)

if engine:
    agent, returns, features = engine["agent"], engine["returns"], engine["features"]
    
    # OOS WINDOW: Last 60 Days
    oos_window_rets = returns.tail(60)
    oos_wealth = (1 + oos_window_rets).cumprod()
    
    # Prediction
    agent.eval()
    last_state = torch.FloatTensor(features.tail(1).values)
    preds = agent(last_state).detach().numpy()[0]
    top_pick = etf_universe[np.argmax(preds)]

    # Metrics (Locked to last 60 days)
    total_oos_ret = oos_wealth[top_pick].iloc[-1] - 1
    ann_ret = ((1 + total_oos_ret) ** (252 / 60)) - 1
    
    daily_rets = oos_window_rets[top_pick]
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() != 0 else 0

    audit_df = pd.DataFrame({
        "Date": returns.tail(45).index.strftime('%Y-%m-%d'),
        "Ticker": [top_pick]*45,
        "Net Return": [f"{v:.2%}" for v in returns[top_pick].tail(45).values]
    })

    interface.render_main_output(top_pick, f"{sharpe:.2f}", (daily_rets > 0).mean(), f"{ann_ret:.2%}", "5 Days", oos_wealth[top_pick], audit_df, engine["train_info"])
