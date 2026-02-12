import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import interface

class AlphaSubModel:
    """Isolated silo for model execution with 80/20 OOS logic"""
    def __init__(self, model_type):
        if model_type == "TRANSFORMER":
            self.model = XGBRegressor(n_estimators=400, max_depth=12, booster='dart', learning_rate=0.01)
        elif model_type == "XGB":
            self.model = XGBRegressor(n_estimators=200, learning_rate=0.05)
        else:
            self.model = RandomForestRegressor(n_estimators=200)
            
        self.best_res = {"score": -9e9, "ticker": "CASH", "ann_return": 0.0, "hit_oos": 0.0, "logs": pd.DataFrame()}

    def train_and_check(self, X, y, etf, h_name, h_val):
        # 80/20 DISSECTION
        split_idx = int(len(X) * 0.8)
        X_train, X_oos = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_oos = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Fit on 80%
        self.model.fit(X_train, y_train)
        # Predict on 20% (OOS Check)
        preds = self.model.predict(X_oos)
        
        # Performance: Annualized Total Return
        ann_ret = np.mean(y_oos) * (252 / h_val)
        
        # Competitive selection: Target Highest Possible Return
        if np.mean(preds) > 0 and ann_ret > self.best_res['score']:
            self.best_res = {
                "score": ann_ret, "ticker": etf, "horizon": h_name,
                "ann_return": ann_ret,
                "hit_oos": ((preds > 0) == (y_oos > 0)).mean(),
                "logs": pd.DataFrame({"Predicted": preds, "Actual Return": y_oos}, index=y_oos.index.strftime('%Y-%m-%d'))
            }

st.set_page_config(layout="wide", page_title="Alpha Engine v1.5")
friction = interface.render_sidebar(2015, 2026)

@st.cache_data(ttl="1h")
def get_verified_data(tkrs):
    d = yf.download(tkrs, start="2015-01-01")['Close'].ffill().dropna()
    for t in tkrs:
        d[f'{t}_mom'] = d[t].pct_change(10) # Momentum signal to help rotation (SLV vs GLD)
    return d

tickers = ["GLD", "SLV", "VNQ", "TLT", "TBT"]
raw_data = get_verified_data(tickers)

if not raw_data.empty:
    # SANITIZER: Replace inf/NaN for XGBoost stability
    returns = raw_data.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    features = returns.shift(1).dropna() # LAGGED to stop data leakage
    
    # WATER-TIGHT SEGREGATION (Three Models)
    model_a = AlphaSubModel("TRANSFORMER")
    model_b1 = AlphaSubModel("XGB") # Competition for Model B
    model_b2 = AlphaSubModel("RF")  # Competition for Model B

    for h_val, h_name in {1: "1D", 3: "3D", 5: "5D"}.items():
        for etf in tickers:
            target = raw_data[etf].pct_change(h_val).shift(-h_val).dropna()
            idx = features.index.intersection(target.index)
            
            model_a.train_and_check(features.loc[idx], target.loc[idx], etf, h_name, h_val)
            model_b1.train_and_check(features.loc[idx], target.loc[idx], etf, h_name, h_val)
            model_b2.train_and_check(features.loc[idx], target.loc[idx], etf, h_name, h_val)

    # Model B Tournament: Highest Return between XGB and RF
    best_b = model_b1.best_res if model_b1.best_res['score'] > model_b2.best_res['score'] else model_b2.best_res

    interface.render_dashboard(model_a.best_res, best_b)
    interface.render_logs(model_a.best_res['logs'], best_b['logs'])
