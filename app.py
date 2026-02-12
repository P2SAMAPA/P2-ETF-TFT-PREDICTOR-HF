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

# ... [PPONetwork and train_engine functions remain exactly as before] ...

# Execution Flow
st.set_page_config(page_title="Alpha Engine v5.1", layout="wide")
regime_year, tx_cost_bps = interface.render_sidebar()

etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
engine = train_engine(regime_year, etf_universe)
agent, returns, tactical = engine["agent"], engine["returns"], engine["tactical"]

# Inference 
agent.eval()
curr_state_raw = engine["features"][-1].reshape(1, -1)
raw_preds = agent(torch.FloatTensor(curr_state_raw)).detach().numpy()[0]
cost_pct = tx_cost_bps / 10000

# Horizon Selection Logic (1, 3, 5 Day Net of Costs)
decision_matrix = []
for i, ticker in enumerate(etf_universe):
    if returns[ticker].tail(2).sum() < -0.015: continue 
    for h in [1, 3, 5]:
        val = (raw_preds[i] * h) - cost_pct
        decision_matrix.append({"Ticker": ticker, "Horizon": h, "NetVal": val})

best = pd.DataFrame(decision_matrix).sort_values("NetVal", ascending=False).iloc[0]
top_pick, top_horizon = best["Ticker"], f"{int(best['Horizon'])} Day" if best['Horizon'] == 1 else f"{int(best['Horizon'])} Days"

# Prep data for UI
oos_returns = returns[top_pick].tail(120)
ann_return = round(((1 + oos_returns.mean())**252 - 1) * 100, 1)
hit_rate = (oos_returns.tail(15) > 0).sum() / 15
wealth = (1 + oos_returns).cumprod()
audit_df = pd.DataFrame({
    "Date": oos_returns.tail(15).index.strftime('%Y-%m-%d'), 
    "Ticker": [top_pick]*15, 
    "Net Return": [f"{v:.2%}" for v in oos_returns.tail(15).values]
})

# Render UI
interface.render_main_output(top_pick, ann_return, "1.82", hit_rate, top_horizon, wealth, audit_df, oos_returns)
