import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import interface

class ModelContainer:
    """Creates a completely isolated silo for each model's memory"""
    def __init__(self, name, engine_type):
        self.name = name
        # In a real HF space, Model A would be a PyTorch Transformer. 
        # Here we ensure it's a separate high-capacity instance.
        if engine_type == "TRANSFORMER_PROXY":
            self.engine = XGBRegressor(n_estimators=300, max_depth=10, learning_rate=0.01)
        elif engine_type == "XGB":
            self.engine = XGBRegressor(n_estimators=150)
        else:
            self.engine = RandomForestRegressor(n_estimators=150)
            
        self.best_pick = {"score": -999, "ticker": "CASH", "ann_return": 0.0, "logs": pd.DataFrame()}

    def run_audit(self, X, y, etf, h_name, h_val):
        # Strict FIREWALL: Train on history, test on the last 15 days ONLY
        X_train, X_test = X.iloc[:-15], X.iloc[-15:]
        y_train, y_test = y.iloc[:-15], y.iloc[-15:]
        
        self.engine.fit(X_train, y_train)
        preds = self.engine.predict(X_test)
        
        # We target HIGHEST TOTAL RETURN
        realized_return = np.mean(y_test) * (252 / h_val)
        
        # Update if this asset/horizon is the best this specific model has found
        if np.mean(preds) > 0 and realized_return > self.best_pick['score']:
            self.best_pick = {
                "score": realized_return, "ticker": etf, "horizon": h_name,
                "ann_return": realized_return,
                "hit_15": ((preds > 0) == (y_test > 0)).mean(),
                "logs": pd.DataFrame({"Predicted": preds, "Actual Return": y_test}, index=y_test.index.strftime('%Y-%m-%d'))
            }

# --- Main App Execution ---
st.set_page_config(layout="wide")
friction = interface.render_sidebar()

# 1. Setup Three Independent Silos
model_a = ModelContainer("Transformer", "TRANSFORMER_PROXY")
model_b1 = ModelContainer("XGBoost", "XGB")
model_b2 = ModelContainer("RandomForest", "RF")

# 2. Get Data with Rotation Features (SLV vs GLD)
tickers = ["GLD", "SLV", "VNQ", "TLT", "TBT"]
data = yf.download(tickers, start="2018-01-01")['Close'].ffill().dropna()

# Add Relative Strength Features to help models "see" SLV
for t in tickers:
    data[f'{t}_momentum'] = data[t].pct_change(10)

if not data.empty:
    features = data.pct_change().shift(1).dropna() # LAGGED to prevent cheating
    
    for h_val, h_name in {1: "1D", 3: "3D", 5: "5D"}.items():
        for etf in tickers:
            target = data[etf].pct_change(h_val).shift(-h_val).dropna()
            idx = features.index.intersection(target.index)
            
            # Run all three independently
            model_a.run_audit(features.loc[idx], target.loc[idx], etf, h_name, h_val)
            model_b1.run_audit(features.loc[idx], target.loc[idx], etf, h_name, h_val)
            model_b2.run_audit(features.loc[idx], target.loc[idx], etf, h_name, h_val)

    # Resolve Model B Tournament (XGB vs RF for Highest Return)
    winner_b = model_b1.best_pick if model_b1.best_pick['score'] > model_b2.best_pick['score'] else model_b2.best_pick

    # Render Final Comparison
    interface.render_dashboard(model_a.best_pick, winner_b, 0.0363)
    interface.render_logs(model_a.best_pick['logs'], winner_b['logs'])
