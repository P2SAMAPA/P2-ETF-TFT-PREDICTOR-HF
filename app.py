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
# 1. THE MATH ENGINE (PPO + ENTROPY)
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
    
    # Robust Download with Retries
    raw_data = None
    for i in range(3):
        try:
            raw_data = yf.download(etf_list, start=start_date, progress=False)['Close']
            if not raw_data.empty: break
        except:
            time.sleep(2)
    
    if raw_data is None or raw_data.empty:
        st.error("Data download failed. Please refresh.")
        return None

    # Fix: Explicitly handle pct_change and dropna
    returns_df = raw_data.ffill().pct_change().dropna()
    vols = returns_df.rolling(window=20).std().dropna()
    returns_df = returns_df.loc[vols.index]
    
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    macro = pd.DataFrame(index=returns_df.index)
    macro['10Y2Y'] = fred.get_series('T10Y2Y', start_date).reindex(returns_df.index).ffill()
    
    m_raw = yf.download(["^VIX", "^MOVE"], start=start_date, progress=False)['Close']
    full_df = pd.concat([returns_df, vols, macro, m_raw.reindex(returns_df.index).ffill()], axis=1).dropna()
    
    # --- DYNAMIC OOS PROTECTION ---
    total_samples = len(full_df)
    oos_size = 126 # 6 Months
    # Ensure at least 252 days (1 year) for training
    if total_samples - oos_size < 252:
        oos_size = max(21, total_samples - 252) 
    
    train_df = full_df.iloc[:-oos_size]
    train_returns = returns_df.iloc[:-oos_size]
    
    # Validation for Empty Array
    if train_df.empty:
        st.error(f"Insufficient data for anchor {start_year}. Try an earlier year.")
        return None

    # Epoch Calculation
    years_of_data = 2026 - start_year
    n_epochs = int(np.clip(150 * (18 / max(years_of_data, 1)), 150, 1000))
    
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_df)
    
    agent = PPONetwork(scaled_train.shape[1], len(etf_list))
    optimizer = optim.Adam(agent.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    for epoch in range(n_epochs):
        idx = np.random.randint(0, len(scaled_train)-1, 64)
        preds = agent(torch.FloatTensor(scaled_train[idx]))
        targets = torch.FloatTensor(train_returns.values[idx])
        
        mse_loss = nn.MSELoss()(preds, targets)
        probs = torch.softmax(preds, dim=1)
        entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-5), dim=1))
        
        loss = mse_loss - (0.02 * entropy) 
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        
    tactical = XGBRegressor(n_estimators=100, max_depth=5, objective='reg:absoluteerror')
    tactical.fit(scaled_train[:-1], train_returns.shift(-1).dropna().mean(axis=1))
    
    return {"agent": agent, "scaler": scaler, "returns": returns_df, "full_features": full_df, "tactical": tactical, "fred": fred, "oos_size": oos_size}

# ==========================================
# 2. EXECUTION FLOW
# ==========================================
st.set_page_config(page_title="Alpha Engine v5", layout="wide")
regime_year, tx_cost_bps = interface.render_sidebar()

etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
engine = train_engine(regime_year, etf_universe)

if engine:
    agent, returns, full_features = engine["agent"], engine["returns"], engine["full_features"]
    oos_size = engine["oos_size"]

    # Inference
    agent.eval()
    curr_state_scaled = engine["scaler"].transform(full_features.tail(1))
    raw_preds = agent(torch.FloatTensor(curr_state_scaled)).detach().numpy()[0]
    cost_pct = tx_cost_bps / 10000

    decision_matrix = []
    for i, ticker in enumerate(etf_universe):
        if returns[ticker].tail(2).sum() < -0.015: continue 
        for h in [1, 3, 5]:
            val = (raw_preds[i] * h) - cost_pct
            decision_matrix.append({"Ticker": ticker, "Horizon": h, "NetVal": val})

    if not decision_matrix:
        top_pick, top_horizon = "TBT", "3 Days"
    else:
        best = pd.DataFrame(decision_matrix).sort_values("NetVal", ascending=False).iloc[0]
        top_pick, top_horizon = best["Ticker"], f"{int(best['Horizon'])} Day" if best['Horizon'] == 1 else f"{int(best['Horizon'])} Days"

    # --- PERFORMANCE ---
    oos_window = returns[top_pick].tail(oos_size)
    wealth = (1 + oos_window).cumprod()
    ann_return_val = (wealth.iloc[-1] ** (252 / oos_size)) - 1
    
    try:
        sofr_series = engine["fred"].get_series('SOFR', oos_window.index[0]).reindex(oos_window.index).ffill()
        rf_daily = (sofr_series.mean() / 100) / 252 
    except:
        rf_daily = 0.0525 / 252 

    excess_returns = oos_window - rf_daily
    sharpe_val = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0.0
    
    audit_df = pd.DataFrame({
        "Date": returns[top_pick].tail(15).index.strftime('%Y-%m-%d'), 
        "Ticker": [top_pick]*15, 
        "Net Return": [f"{v:.2%}" for v in returns[top_pick].tail(15).values]
    })

    interface.render_main_output(top_pick, f"{sharpe_val:.2f}", (oos_window > 0).sum() / oos_size, f"{ann_return_val:.2%}", top_horizon, wealth, audit_df)
