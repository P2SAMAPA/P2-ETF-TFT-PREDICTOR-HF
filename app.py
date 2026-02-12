import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import interface

# --- CORE ISOLATION CLASS ---
class AlphaModel:
    def __init__(self, name, model_type):
        self.name = name
        self.engine = XGBRegressor(n_estimators=100) if model_type == "XGB" else RandomForestRegressor(n_estimators=100)
        self.best_results = {"score": -999, "ticker": "CASH", "ann_return": 0.0, "sharpe": 0.0, "hit_15": 0.0, "horizon": "N/A", "logs": pd.DataFrame()}

    def train_and_evaluate(self, X, y, etf, h_name, h_val, sofr):
        # Time-Series Split: Last 15 days for Verification
        X_train, X_test = X.iloc[:-15], X.iloc[-15:]
        y_train, y_test = y.iloc[:-15], y.iloc[-15:]
        
        self.engine.fit(X_train, y_train)
        preds = self.engine.predict(X_test)
        
        # Calculate Realized Metrics
        hits = (preds > 0) == (y_test > 0)
        ann_ret = np.mean(y_test) * (252 / h_val)
        vol = np.std(y_test) * np.sqrt(252 / h_val) + 1e-9
        
        res = {
            "ticker": etf, "horizon": h_name, "ann_return": ann_ret,
            "sharpe": (ann_ret - sofr) / vol, "hit_15": hits.mean(),
            "logs": pd.DataFrame({"ETF": etf, "Predicted Return": preds, "Actual Return": y_test}, index=y_test.index.strftime('%Y-%m-%d'))
        }
        
        # Tournament Logic: Only pick if prediction is positive
        if np.mean(preds) > 0 and res['ann_return'] > self.best_results['score']:
            self.best_results = {**res, "score": res['ann_return']}

# --- MAIN APP ---
st.set_page_config(layout="wide", page_title="Alpha Engine v1.1")

@st.cache_data(ttl="1h")
def get_clean_data(tickers):
    data = yf.download(tickers, start="2018-01-01", auto_adjust=True)['Close'].ffill().dropna()
    return data

tickers = ["GLD", "SLV", "VNQ", "TLT", "TBT", "CPER", "^VIX"]
df = get_clean_data(tickers)
sofr_rate = 0.0363 # Current 3.63%

if not df.empty:
    # 1. Feature Engineering (Lags to prevent leakage)
    rets = df.pct_change().dropna()
    features = rets.shift(1).dropna()
    
    # 2. Initialize Two SEPARATE Model Containers
    model_a = AlphaModel("Transformer", "XGB")
    model_b = AlphaModel("RegimeSwitcher", "RF")
    
    horizons = {1: "1 Day", 3: "3 Days", 5: "5 Days"}
    
    for h_val, h_name in horizons.items():
        for etf in ["GLD", "SLV", "VNQ", "TLT", "TBT"]:
            target = df[etf].pct_change(h_val).shift(-h_val).dropna()
            idx = features.index.intersection(target.index)
            
            # Run both models independently
            model_a.train_and_evaluate(features.loc[idx], target.loc[idx], etf, h_name, h_val, sofr_rate)
            model_b.train_and_evaluate(features.loc[idx], target.loc[idx], etf, h_name, h_val, sofr_rate)

    # 3. Render Independent Results
    interface.render_comparison_dashboard(model_a.best_results, model_b.best_results, sofr_rate)
    interface.render_tactical_logs(model_a.best_results['logs'], model_b.best_results['logs'])
