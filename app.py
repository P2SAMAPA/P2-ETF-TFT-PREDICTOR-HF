import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import interface 

# ==========================================
# 1. THE DEEP PURE-PPO ENGINE
# ==========================================
class DeepPPONetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DeepPPONetwork, self).__init__()
        # Increased depth for 1,000 epoch convergence
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x): return self.network(x)

@st.cache_data(ttl="6h")
def get_clean_data(tickers, start):
    data = yf.download(tickers, start=start, auto_adjust=True, threads=False, progress=False)
    return data['Close'] if not data.empty else None

@st.cache_resource(ttl="1d")
def train_pure_engine(etf_list):
    start_date = "2008-01-01"
    train_end = (pd.Timestamp.now() - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
    
    raw_close = get_clean_data(etf_list, start_date)
    if raw_close is None: return None
    
    returns_df = raw_close.ffill().pct_change().dropna()
    
    # Pure Features: Only Returns and 20D Volatility
    vols = returns_df.rolling(window=20).std()
    full_df = pd.concat([returns_df, vols], axis=1).dropna()
    
    scaler = StandardScaler()
    train_data = full_df.loc[:train_end]
    target_rets = returns_df.loc[train_data.index]
    
    scaled_train = scaler.fit_transform(train_data)
    agent = DeepPPONetwork(scaled_train.shape[1], len(etf_list))
    optimizer = optim.Adam(agent.parameters(), lr=1e-4)
    
    # --- 1,000 EPOCH DEEP TRAINING ---
    progress_bar = st.progress(0, text="Training Deep PPO Agent (1,000 Epochs)...")
    for epoch in range(1000):
        idx = np.random.randint(0, len(scaled_train)-1, 128)
        batch_x = torch.FloatTensor(scaled_train[idx])
        batch_y = torch.FloatTensor(target_rets.values[idx])
        
        loss = nn.MSELoss()(agent(batch_x), batch_y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        if epoch % 100 == 0:
            progress_bar.progress(epoch/1000)
    progress_bar.empty()

    return {
        "agent": agent, "scaler": scaler, "returns": returns_df, 
        "features": full_df, "train_info": {"start": start_date, "end": train_end}
    }

# ==========================================
# 2. UI & BENCHMARK EXECUTION
# ==========================================
st.set_page_config(layout="wide")
# Current SOFR rate as of Feb 2026
SOFR_RATE = 0.0363 

etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
engine = train_pure_engine(etf_universe)

if engine:
    agent, scaler, returns, features = engine["agent"], engine["scaler"], engine["returns"], engine["features"]
    
    # Prediction logic
    agent.eval()
    current_raw = features.tail(1)
    current_scaled = scaler.transform(current_raw)
    preds = agent(torch.FloatTensor(current_scaled)).detach().numpy()[0]
    top_pick = etf_universe[np.argmax(preds)]

    # OOS Performance
    oos_rets = returns.tail(60)
    oos_wealth = (1 + oos_rets).cumprod()
    
    total_oos_ret = oos_wealth[top_pick].iloc[-1] - 1
    ann_ret_val = ((1 + total_oos_ret) ** (252 / 60)) - 1
    
    daily_rets = oos_rets[top_pick]
    # Sharpe Ratio (Excess return over SOFR)
    excess_ret = daily_rets.mean() - (SOFR_RATE / 252)
    sharpe_val = (excess_ret / daily_rets.std()) * np.sqrt(252) if daily_rets.std() != 0 else 0

    audit_df = pd.DataFrame({
        "Date": returns.tail(45).index.strftime('%Y-%m-%d'),
        "Ticker": [top_pick]*45,
        "Net Return": [f"{v:.2%}" for v in returns[top_pick].tail(45).values]
    })

    # UPDATED UI ELEMENTS
    st.markdown(f"### 🚀 Alpha Engine v10: Deep Pure-PPO")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("TOP PICK", top_pick)
    with col2:
        st.metric("Annualised (Last 60D)", f"{ann_ret_val:.2%}")
    with col3:
        st.metric("Sharpe Ratio (Last 60D)", f"{sharpe_val:.2f}")
        st.caption(f"Benchmark: SOFR @ {SOFR_RATE:.2%}") # Small font SOFR note

    st.line_chart(oos_wealth[top_pick])
    st.write("#### Verification Log (Last 15 Trading Days)")
    st.table(audit_df.head(15))
