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
# 1. SELF-HEALING DATA LOADER (YF + POLYGON)
# ==========================================
def get_polygon_data(ticker, start_date):
    """Fallback: Pulls daily bars from Polygon.io"""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        return None
    
    # Adjusting date format for Polygon API
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/2026-12-31?adjusted=true&sort=asc&apiKey={api_key}"
    try:
        resp = requests.get(url).json()
        if "results" in resp:
            df = pd.DataFrame(resp["results"])
            df['t'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('t', inplace=True)
            return df['c'] # Returns 'Close' prices
        return None
    except:
        return None

@st.cache_data(ttl="6h")
def get_dual_source_data(tickers, start):
    # ATTEMPT 1: yfinance
    try:
        data = yf.download(tickers, start=start, auto_adjust=True, threads=False, progress=False)
        if not data.empty and not data.isna().all().any():
            return data['Close']
    except Exception:
        pass
    
    # ATTEMPT 2: Polygon Fallback (Ticker by Ticker)
    st.warning("⚠️ Yahoo Rate-Limited. Switching to Polygon.io Lifeboat...")
    poly_frames = {}
    for t in tickers:
        poly_data = get_polygon_data(t, start)
        if poly_data is not None:
            poly_frames[t] = poly_data
    
    if poly_frames:
        return pd.DataFrame(poly_frames).ffill()
    
    st.error("❌ Both Data Sources Failed. Please check API Keys and Limits.")
    return None

# ==========================================
# 2. TACTICAL ENGINE LOGIC (CONSOLIDATED)
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
def train_engine_dual_source(etf_list, tc_pct):
    start_date = "2012-01-01" # Optimized lookback for regime switching
    train_end = (pd.Timestamp.now() - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
    
    raw_close = get_dual_source_data(etf_list, start_date)
    if raw_close is None: return None
    
    # Use fill_method=None to avoid the FutureWarning
    rets = raw_close.ffill().pct_change(fill_method=None).dropna()
    
    # Indicators for Peak Detection
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
    optimizer = optim.Adam(agent.parameters(), lr=4e-5)
    
    p_bar = st.progress(0, text="Deep Training via Dual-Source Data...")
    for epoch in range(1000):
        idx = np.random.randint(0, len(scaled_train)-1, 128)
        loss = nn.MSELoss()(agent(torch.FloatTensor(scaled_train[idx])), torch.FloatTensor(target_rets.values[idx]))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 100 == 0: p_bar.progress(epoch/1000)
    p_bar.empty()

    return {"agent": agent, "scaler": scaler, "returns": rets, "features": full_df, "train_info": {"start": start_date, "end": train_end}}

# ==========================================
# 3. UI INTEGRATION
# ==========================================
st.set_page_config(layout="wide")

# Fetch SOFR (Safe Fallback)
fred_key = os.getenv("FRED_API_KEY")
fred = Fred(api_key=fred_key) if fred_key else None
live_sofr = fred.get_series('SOFR').dropna().iloc[-1]/100 if fred else 0.0363

# Sidebar
st.sidebar.header("🕹️ Strategy Controls")
tc_input = st.sidebar.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05) / 100

etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
engine = train_engine_dual_source(etf_universe, tc_input)

if engine:
    agent, scaler, returns, features = engine["agent"], engine["scaler"], engine["returns"], engine["features"]
    
    # Peak Detection Override
    agent.eval()
    curr_state = scaler.transform(features.tail(1))
    raw_scores = agent(torch.FloatTensor(curr_state)).detach().numpy()[0]
    
    for i, t in enumerate(etf_universe):
        if features[f"{t}_vol_ratio"].iloc[-1] > 1.45: # Exhaustion Spike
            raw_scores[i] *= 0.35 # Forced Rotation
            
    top_pick = etf_universe[np.argmax(raw_scores)]
    oos_rets = returns.tail(60)[top_pick]
    
    interface.render_main_output(
        top_pick, 
        f"{((oos_rets.mean()-(live_sofr/252))/oos_rets.std()*np.sqrt(252)):.2f}",
        (oos_rets > 0).mean(),
        f"{(((1+oos_rets).prod()**(252/60))-1):.2%}",
        "5-Day Tactical",
        (1+oos_rets).cumprod(),
        pd.DataFrame({"Date": returns.tail(15).index.strftime('%y-%m-%d'), "Ticker": [top_pick]*15, "Net Return": [f"{v:.2%}" for v in returns[top_pick].tail(15).values]}),
        engine["train_info"]
    )
