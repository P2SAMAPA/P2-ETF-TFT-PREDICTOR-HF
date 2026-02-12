import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import interface

class AlphaSilo:
    def __init__(self, model_type):
        if model_type == "TRANSFORMER":
            self.model = XGBRegressor(n_estimators=300, max_depth=10, learning_rate=0.01, booster='dart')
        elif model_type == "XGB":
            self.model = XGBRegressor(n_estimators=100)
        else:
            self.model = RandomForestRegressor(n_estimators=100)
        self.best = {"score": -9e9, "ticker": "CASH", "ann_return": 0.0, "hit_oos": 0.0, "logs": pd.DataFrame()}

    def run_tournament(self, X, y, etf, h_name, h_val):
        # 80/20 Split
        split = int(len(X) * 0.8)
        X_train, X_oos = X.iloc[:split], X.iloc[split:]
        y_train, y_oos = y.iloc[:split], y.iloc[split:]

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_oos)

        # Performance Check
        actual_ann_return = np.mean(y_oos) * (252 / h_val)
        
        if np.mean(preds) > 0 and actual_ann_return > self.best['score']:
            # Log preparation with ETF name included
            log_df = pd.DataFrame({
                "Ticker": etf,
                "Predicted": preds,
                "Actual Return": y_oos
            }, index=y_oos.index)
            
            self.best = {
                "score": actual_ann_return, "ticker": etf, "horizon": h_name,
                "ann_return": actual_ann_return,
                "hit_oos": ((preds > 0) == (y_oos > 0)).mean(),
                "logs": log_df
            }

st.set_page_config(layout="wide")
tickers = ["GLD", "SLV", "VNQ", "TLT", "TBT"]
data = yf.download(tickers, start="2015-01-01")['Close'].ffill().dropna()

# Initialize Sidebar with actual data
friction = interface.render_sidebar(data)

if not data.empty:
    # Sanitation and Lagging (Firewall)
    returns = data.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    features = returns.shift(1).dropna()
    
    # 3-Way Independent Models
    m_a = AlphaSilo("TRANSFORMER")
    m_b1 = AlphaSilo("XGB")
    m_b2 = AlphaSilo("RF")

    for h_val, h_name in {1: "1D", 3: "3D", 5: "5D"}.items():
        for etf in tickers:
            target = data[etf].pct_change(h_val).shift(-h_val).dropna()
            common_idx = features.index.intersection(target.index)
            
            X_curr, y_curr = features.loc[common_idx], target.loc[common_idx]
            
            m_a.run_tournament(X_curr, y_curr, etf, h_name, h_val)
            m_b1.run_tournament(X_curr, y_curr, etf, h_name, h_val)
            m_b2.run_tournament(X_curr, y_curr, etf, h_name, h_val)

    # Tournament Winner for B
    best_b = m_b1.best if m_b1.best['score'] > m_b2.best['score'] else m_b2.best

    interface.render_dashboard(m_a.best, best_b)
    interface.render_logs(m_a.best['logs'], best_b['logs'])
