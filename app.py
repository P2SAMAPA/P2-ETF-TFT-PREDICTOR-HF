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
import time

# ==========================================
# 1. THE MATH ENGINE (CLEAN OPTION A)
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

def add_indicators(df, tickers):
    for t in tickers:
        delta = df[t].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df[f'{t}_RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        ema12 = df[t].ewm(span=12, adjust=False).mean()
        ema26 = df[t].ewm(span=26, adjust=False).mean()
        df[f'{t}_MACD'] = ema12 - ema26
    return df.dropna()

@st.cache_resource(ttl="1d")
def train_engine(etf_list):
    start_date = "2004-01-01"
    raw_data = None
    for i in range(3):
        try:
            raw_data = yf.download(etf_list, start=start_date, progress=False)['Close']
            if not raw_data.empty: break
        except:
            time.sleep(1)
            
    if raw_data is None or raw_data.empty: return None

    returns_df = raw_data.ffill().pct_change().dropna()
    feat_df = add_indicators(returns_df.copy(), etf_list)
    vols = returns_df.rolling(window=20).std().dropna()
    
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    macro = pd.DataFrame(index=feat_df.index)
    macro['10Y2Y'] = fred.get_series('T10Y2Y', start_date).reindex(feat_df.index).ffill()
    m_raw = yf.download(["^VIX", "^MOVE"], start=start_date, progress=False)['Close']
    
    full_df = pd.concat([feat_df, vols.reindex(feat_df.index), macro, m_raw.reindex(feat_df.index).ffill()], axis=1).dropna()
    scaler = StandardScaler()
    
    # --- LOCKED TRAINING & OOS BOUNDARIES ---
    train_end = "2024-12-31"
    oos_start = "2025-01-01"
    
    train_data = full_df.loc["2008-01-01":train_end]
    target_rets = returns_df.loc[train_data.index]
    
    scaled_train = scaler.fit_transform(train_data)
    agent = PPONetwork(scaled_train.shape[1], len(etf_list))
    optimizer = optim.Adam(agent.parameters(), lr=5e-4)
    
    # Training Loop
    for _ in range(500):
        idx = np.random.randint(0, len(scaled_train)-1, 64)
        loss = nn.MSELoss()(agent(torch.FloatTensor(scaled_train[idx])), torch.FloatTensor(target_rets.values[idx]))
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    return {"agent": agent, "scaler": scaler, "returns": returns_df, "full_features": full_df, "oos_start": oos_start}

# ==========================================
# 2. INTERFACE & DISPLAY
# ==========================================
st.set_page_config(page_title="Alpha Engine v7", layout="wide")

st.sidebar.title("Configuration")
st.sidebar.info("Training: 2008-2024\nOOS: 2025-Present")
tx_cost = st.sidebar.slider("Trading Cost (bps)", 0, 50, 10)

etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
engine = train_engine(etf_universe)

if engine:
    agent, returns, full_features = engine["agent"], engine["returns"], engine["full_features"]
    oos_start = engine["oos_start"]

    agent.eval()
    curr_state = engine["scaler"].transform(full_features.tail(1))
    raw_preds = agent(torch.FloatTensor(curr_state)).detach().numpy()[0]
    
    decision_matrix = []
    for i, ticker in enumerate(etf_universe):
        for h in [1, 3, 5]:
            val = (raw_preds[i] * h) - (tx_cost / 10000)
            decision_matrix.append({"Ticker": ticker, "Horizon": h, "NetVal": val})

    best = pd.DataFrame(decision_matrix).sort_values("NetVal", ascending=False).iloc[0]
    top_pick = best["Ticker"]

    # --- ACCURATE 2025-2026 PERFORMANCE MATH ---
    oos_window = returns[top_pick].loc[oos_start:]
    wealth = (1 + oos_window).cumprod()
    
    # Calculate exactly based on days in 2025/2026 window
    days_passed = len(oos_window)
    if days_passed > 0:
        total_return = wealth.iloc[-1] - 1
        ann_ret_val = (wealth.iloc[-1] ** (252 / days_passed)) - 1
        hit_rate = (oos_window > 0).sum() / days_passed
    else:
        ann_ret_val, hit_rate = 0, 0

    audit_df = pd.DataFrame({
        "Date": returns[top_pick].tail(45).index.strftime('%Y-%m-%d'),
        "Ticker": [top_pick]*45,
        "Net Return": [f"{v:.2%}" for v in returns[top_pick].tail(45).values]
    })

    interface.render_main_output(
        top_pick, 
        "2025 OOS", 
        hit_rate, 
        f"{ann_ret_val:.2%}", 
        f"{int(best['Horizon'])} Days", 
        wealth, 
        audit_df
    )
