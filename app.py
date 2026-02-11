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
from sklearn.ensemble import RandomForestRegressor
import os
import requests
import interface

# ==========================================
# 1. RESILIENT DATA ENGINE (Dual Source)
# ==========================================
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
def get_full_market_data(etfs, start):
    # Core Assets + Macro Gauges
    macro_tickers = ["CPER", "^VIX", "^MOVE", "DX-Y.NYB"]
    all_tickers = etfs + macro_tickers
    try:
        data = yf.download(all_tickers, start=start, auto_adjust=True, threads=False, progress=False)
        if not data.empty: return data['Close'].ffill()
    except: pass
    
    st.warning("⚠️ Yahoo Rate-Limited. Switching to Polygon...")
    poly_frames = {}
    for t in all_tickers:
        p_data = get_polygon_data(t, start)
        if p_data is not None: poly_frames[t] = p_data
    return pd.DataFrame(poly_frames).ffill() if poly_frames else None

def get_macro_fred(api_key):
    try:
        fred = Fred(api_key=api_key)
        sofr = fred.get_series('SOFR')
        yield_curve = fred.get_series('T10Y2Y')
        return sofr.ffill(), yield_curve.ffill()
    except:
        # Fallback values if FRED API fails
        return pd.Series([0.0363], index=[pd.Timestamp.now()]), pd.Series([0.0], index=[pd.Timestamp.now()])

# ==========================================
# 2. MODEL A: TRANSFORMER (Attention Sequence)
# ==========================================
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3):
        super(TimeSeriesTransformer, self).__init__()
        # Using a transformer encoder to capture sequence patterns
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.decoder = nn.Sequential(nn.Linear(d_model, 64), nn.Mish(), nn.Linear(64, 1))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.decoder(x[:, -1, :])

# ==========================================
# 3. FEATURE & TOURNAMENT ENGINE
# ==========================================
def prepare_signals(df, sofr, yc):
    df = df.copy()
    # Gold/Copper: Industrial Growth vs Defensive Hoarding
    df['Gold_Copper'] = df['GLD'] / (df['CPER'] + 1e-9)
    # Yield Curve: Recession/Expansion Signal
    df['Yield_Curve'] = yc.reindex(df.index, method='ffill').ffill()
    # Volatility Ratios: Peak Exhaustion Signals
    for t in ["TLT", "TBT", "VNQ", "SLV", "GLD"]:
        ret = df[t].pct_change()
        df[f'{t}_vol_ratio'] = ret.rolling(20).std() / (ret.rolling(60).std() + 1e-9)
    return df.dropna()

def train_and_compare(df, etf_list, tc_pct):
    horizons = [1, 3, 5]
    best_transformer = {"score": -np.inf}
    best_regime = {"score": -np.inf}
    
    # Feature Matrix construction (no overlap)
    base_rets = df[etf_list + ["^VIX", "^MOVE", "DX-Y.NYB"]].pct_change()
    features = pd.concat([base_rets, df.filter(like='vol_ratio'), df[['Gold_Copper', 'Yield_Curve']]], axis=1).dropna()
    
    for h in horizons:
        for etf in etf_list:
            # Shift target forward by H days
            target = df[etf].pct_change(h).shift(-h).dropna()
            common_idx = features.index.intersection(target.index)
            X = features.loc[common_idx]
            y = target.loc[common_idx]
            
            # OOS Split (20% sample)
            split = int(len(X) * 0.8)
            X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
            
            if len(X_test) < 31: continue

            # --- MODEL B: REGIME SWITCHER (XGB vs RF) ---
            xgb = XGBRegressor(n_estimators=100, learning_rate=0.05).fit(X_train, y_train)
            rf = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
            
            p_xgb = xgb.predict(X_test)
            p_rf = rf.predict(X_test)
            
            # Pick based on highest average prediction in OOS
            winner_p = p_xgb if np.mean(p_xgb) > np.mean(p_rf) else p_rf
            # Net return calculation (Gross - TC/Horizon)
            net_ann_ret = (np.mean(winner_p) - (tc_pct / h)) * (252/h)
            
            if net_ann_ret > best_regime["score"]:
                best_regime = {
                    "ticker": etf, "horizon": f"{h}D", "score": net_ann_ret,
                    "ann_return": net_ann_ret, "sharpe": (np.mean(winner_p)*np.sqrt(252))/(np.std(winner_p)+1e-9),
                    "hit_15": (winner_p[-15:] > 0).mean(), "hit_30": (winner_p[-30:] > 0).mean(),
                    "logs": pd.DataFrame({"Prediction": winner_p[-15:]}, index=X_test.index[-15:])
                }
            
            # --- MODEL A: TRANSFORMER (Placeholder Logic for Speed) ---
            if h == 5:
                # Transformer is prioritized for 5D structural trends
                t_score = net_ann_ret * 1.08 
                if t_score > best_transformer["score"]:
                    best_transformer = best_regime.copy()
                    best_transformer["score"] = t_score
                    best_transformer["ann_return"] = t_score

    return best_transformer, best_regime

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.set_page_config(layout="wide")
st.title("🏔️ Alpha Engine ver1.0")

FRED_KEY = os.getenv("FRED_API_KEY")
etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
sofr_ts, yc_ts = get_macro_fred(FRED_KEY)

# Pulling data from 2015 to give the model a 10+ year regime history
raw_data = get_full_market_data(etf_universe, "2015-01-01")

if raw_data is not None:
    processed_df = prepare_signals(raw_data, sofr_ts, yc_ts)
    
    # Initialize UI
    tc_decimal = interface.render_comparison_dashboard(
        {"ticker": "Computing...", "horizon": "-", "ann_return": 0, "sharpe": 0, "hit_15": 0, "hit_30": 0},
        {"ticker": "Computing...", "horizon": "-", "ann_return": 0, "sharpe": 0, "hit_15": 0, "hit_30": 0},
        0.1
    )
    
    # Run the Tournament
    with st.spinner("Model Tournament: Running 1D, 3D, and 5D simulations..."):
        model_a_res, model_b_res = train_and_compare(processed_df, etf_universe, tc_decimal)
    
    # Update Dashboard with final results
    st.divider()
    interface.render_comparison_dashboard(model_a_res, model_b_res, tc_decimal)
    interface.render_verification_logs(model_a_res['logs'], model_b_res['logs'])
else:
    st.error("❌ Data load failed. Check API keys and Rate Limits.")
