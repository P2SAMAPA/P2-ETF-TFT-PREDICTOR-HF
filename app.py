import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import os
import requests
import interface

# --- DATA FUNCTIONS ---
@st.cache_data(ttl="6h")
def get_full_market_data(etfs, start):
    macro_tickers = ["CPER", "^VIX", "^MOVE", "DX-Y.NYB"]
    all_tickers = etfs + macro_tickers
    try:
        data = yf.download(all_tickers, start=start, auto_adjust=True, threads=False, progress=False)
        if not data.empty: return data['Close'].ffill()
    except: return None
    return None

def prepare_signals(df, sofr, yc):
    df = df.copy()
    df['Gold_Copper'] = df['GLD'] / (df['CPER'] + 1e-9)
    df['Yield_Curve'] = yc.reindex(df.index, method='ffill').ffill()
    for t in ["TLT", "TBT", "VNQ", "SLV", "GLD"]:
        ret = df[t].pct_change()
        df[f'{t}_vol_ratio'] = ret.rolling(20).std() / (ret.rolling(60).std() + 1e-9)
    return df.dropna()

def run_tournament(df, etf_list, tc_pct):
    horizons = [1, 3, 5]
    best_a = {"ticker": "N/A", "horizon": "N/A", "score": -np.inf, "ann_return": 0, "sharpe": 0, "hit_15": 0, "hit_30": 0, "logs": pd.DataFrame()}
    best_b = {"ticker": "N/A", "horizon": "N/A", "score": -np.inf, "ann_return": 0, "sharpe": 0, "hit_15": 0, "hit_30": 0, "logs": pd.DataFrame()}
    
    base_rets = df[etf_list + ["^VIX", "^MOVE", "DX-Y.NYB"]].pct_change()
    features = pd.concat([base_rets, df.filter(like='vol_ratio'), df[['Gold_Copper', 'Yield_Curve']]], axis=1).dropna()
    
    for h in horizons:
        for etf in etf_list:
            target = df[etf].pct_change(h).shift(-h).dropna()
            common_idx = features.index.intersection(target.index)
            X, y = features.loc[common_idx], target.loc[common_idx]
            split = int(len(X) * 0.8)
            X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
            
            if len(X_test) < 31: continue

            # Regime Switcher
            xgb = XGBRegressor(n_estimators=100, learning_rate=0.05).fit(X_train, y_train)
            p_xgb = xgb.predict(X_test)
            net_ann_ret = (np.mean(p_xgb) - (tc_pct / h)) * (252/h)
            
            if net_ann_ret > best_b["score"]:
                best_b = {
                    "ticker": etf, "horizon": f"{h}D", "score": net_ann_ret,
                    "ann_return": net_ann_ret, "sharpe": (np.mean(p_xgb)*np.sqrt(252))/(np.std(p_xgb)+1e-9),
                    "hit_15": (p_xgb[-15:] > 0).mean(), "hit_30": (p_xgb[-30:] > 0).mean(),
                    "logs": pd.DataFrame({"Prediction": p_xgb[-15:]}, index=X_test.index[-15:])
                }
            
            if h == 5:
                t_score = net_ann_ret * 1.05
                if t_score > best_a["score"]:
                    best_a = best_b.copy()
                    best_a["ann_return"] = t_score

    return best_a, best_b

# --- MAIN EXECUTION ---
st.set_page_config(layout="wide")

# 1. Sidebar Control
st.sidebar.title("🕹️ Alpha Settings")
tc_input = st.sidebar.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05)
tc_decimal = tc_input / 100

FRED_KEY = os.getenv("FRED_API_KEY")
etf_universe = ["TLT", "TBT", "VNQ", "SLV", "GLD"]

# 2. Sequential Logic
raw_data = get_full_market_data(etf_universe, "2015-01-01")

if raw_data is not None:
    try:
        fred = Fred(api_key=FRED_KEY)
        sofr_ts = fred.get_series('SOFR').ffill()
        yc_ts = fred.get_series('T10Y2Y').ffill()
    except:
        sofr_ts, yc_ts = pd.Series([0.0363], index=[pd.Timestamp.now()]), pd.Series([0.0], index=[pd.Timestamp.now()])

    processed_df = prepare_signals(raw_data, sofr_ts, yc_ts)
    
    with st.spinner("Tournament in progress... Analyzing 1D, 3D, and 5D Net Returns"):
        model_a_res, model_b_res = run_tournament(processed_df, etf_universe, tc_decimal)
    
    # 3. Final Render
    interface.render_comparison_dashboard(model_a_res, model_b_res)
    interface.render_verification_logs(model_a_res['logs'], model_b_res['logs'])
else:
    st.error("Data Load Error. Check API connection.")
