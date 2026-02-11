import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import os
import requests
import interface 

# ==========================================
# 1. RESILIENT DATA PROVIDERS
# ==========================================
def get_safe_sofr(api_key):
    """Prevents DNS/Network crashes by using a fallback rate."""
    fallback_rate = 0.0363 
    if not api_key: return fallback_rate
    try:
        fred = Fred(api_key=api_key)
        sofr_series = fred.get_series('SOFR')
        if not sofr_series.empty:
            return sofr_series.dropna().iloc[-1] / 100
    except Exception:
        st.sidebar.warning(f"🌐 FRED DNS Error: Using Fallback SOFR ({fallback_rate:.2%})")
    return fallback_rate

def get_polygon_data(ticker, start_date):
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key: return None
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/2026-12-31?adjusted=true&sort=asc&apiKey={api_key}"
    try:
        resp = requests.get(url, timeout=10).json()
        if "results" in resp:
            df = pd.DataFrame(resp["results"])
            df['t'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('t', inplace=True)
            return df['c']
    except: return None
    return None

@st.cache_data(ttl="6h")
def get_dual_source_data(tickers, start):
    # Attempt 1: Yahoo
    try:
        data = yf.download(tickers, start=start, auto_adjust=True, threads=False, progress=False)
        if not data.empty and data['Close'].dropna(axis=1, how='all').shape[1] == len(tickers):
            return data['Close']
    except: pass
    
    # Attempt 2: Polygon Fallback
    st.info("🔄 Fetching from Polygon.io lifeboat...")
    poly_frames = {}
    for t in tickers:
        p_data = get_polygon_data(t, start)
        if p_data is not None: poly_frames[t] = p_data
        
    if len(poly_frames) == len(tickers):
        return pd.DataFrame(poly_frames).ffill()
    return None

# ==========================================
# 2. PPO ARCHITECTURE
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
def train_high_alpha_engine(etf_list, tc_decimal):
    start_date = "2012-01-01"
    train_end = "2025-08-11"
    
    raw_close = get_dual_source_data(etf_list, start_date)
    if raw_close is None or raw_close.empty:
        st.error("❌ Data Failure: All sources (Yahoo & Polygon) are currently unreachable.")
        return None
    
    rets = raw_close.ffill().pct_change(fill_method=None).dropna()
    roll_std = rets.rolling(20).std()
    rel_vol = roll_std / (roll_std.rolling(60).mean() + 1e-9)
    rel_vol.columns = [f"{c}_vol_ratio" for c in rel_vol.columns]
    
    full_df = pd.concat([rets, rel_vol], axis=1).dropna()
    train_data = full_df.loc[:train_end]
    target_rets = rets.loc[train_data.index]
    
    if train_data.empty:
        st.error("❌ ValueError Guard: Training array is empty. Scaling aborted.")
        return None

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data)
    agent = TacticalPPO(scaled_train.shape[1], len(etf_list))
    optimizer = optim.Adam(agent.parameters(), lr=1e-4)
    
    p_bar = st.progress(0, text="Training High-Alpha Model...")
    for epoch in range(1000):
        idx = np.random.randint(0, len(scaled_train)-1, 128)
        loss = nn.MSELoss()(agent(torch.FloatTensor(scaled_train[idx])), torch.FloatTensor(target_rets.values[idx]))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 100 == 0: p_bar.progress(epoch/1000)
    p_bar.empty()

    return {"agent": agent, "scaler": scaler, "returns": rets, "features": full_df, "train_info": {"start": start_date, "end": train_end}}

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
st.set_page_config(layout="wide")

# Sidebar
st.sidebar.header("🕹️ Strategy Controls")
tc_pct = st.sidebar.slider("Transaction Cost / Slippage (%)", 0.0, 1.0, 0.1, 0.05)
tc_decimal = tc_pct / 100

# Live Metrics
fred_key = os.getenv("FRED_API_KEY")
live_sofr = get_safe_sofr(fred_key)

etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
engine = train_high_alpha_engine(etf_universe, tc_decimal)

if engine:
    agent, scaler, returns, features = engine["agent"], engine["scaler"], engine["returns"], engine["features"]
    
    # High-Return Search Logic
    agent.eval()
    curr_scaled = scaler.transform(features.tail(1))
    scores = agent(torch.FloatTensor(curr_scaled)).detach().numpy()[0]
    
    for i, t in enumerate(etf_universe):
        if features[f"{t}_vol_ratio"].iloc[-1] > 1.45: scores[i] *= 0.3
            
    top_pick = etf_universe[np.argmax(scores)]
    oos_rets = returns.tail(60)[top_pick]
    oos_wealth = (1+oos_rets).cumprod()
    
    sharpe_val = ((oos_rets.mean()-(live_sofr/252))/oos_rets.std()*np.sqrt(252))
    ann_ret = ((oos_wealth.iloc[-1])**(252/60)-1)
    
    interface.render_main_output(
        top_pick, f"{sharpe_val:.2f}", (oos_rets > 0).mean(),
        f"{ann_ret:.2%}", "5-Day Tactical", oos_wealth,
        pd.DataFrame({"Date": returns.tail(15).index.strftime('%Y-%m-%d'), "Ticker": [top_pick]*15, "Net Return": [f"{v:.2%}" for v in returns[top_pick].tail(15).values]}),
        engine["train_info"], (tc_decimal * 52)
    )
