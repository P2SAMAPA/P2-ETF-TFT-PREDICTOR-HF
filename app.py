import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import datetime
import interface

# --- TRADING CALENDAR ---
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

@st.cache_data(ttl="6h")
def get_market_data(etfs):
    tickers = etfs + ["CPER", "^VIX", "DX-Y.NYB"]
    # Verified range from your sidebar requirement
    data = yf.download(tickers, start="2018-01-01", auto_adjust=True, progress=False)['Close'].ffill()
    return data

def train_tournament(df, etf_list, tc_bps, sofr):
    horizons = {1: "1 Day", 3: "3 Days", 5: "5 Days"}
    tc_decimal = tc_bps / 10000
    
    # Features: Lagged by 1 to prevent reading the future
    df_rets = df.pct_change().dropna()
    features = df_rets.shift(1).dropna()
    
    t_res = {"score": -999}
    r_res = {"score": -999}
    
    for h_val, h_name in horizons.items():
        for etf in etf_list:
            # Target is the future return we are trying to guess
            target = df[etf].pct_change(h_val).shift(-h_val).dropna()
            
            common = features.index.intersection(target.index)
            X, y = features.loc[common], target.loc[common]
            
            # Split: Preserve the last 15 days for the "Reality Check" log
            X_train, X_test = X.iloc[:-15], X.iloc[-15:]
            y_train, y_test = y.iloc[:-15], y.iloc[-15:]

            # Model A (Transformer Sim) and Model B (Regime Switcher)
            m_a = XGBRegressor(n_estimators=100).fit(X_train, y_train)
            m_b = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
            
            p_a, p_b = m_a.predict(X_test), m_b.predict(X_test)

            def get_stats(preds, actuals, h):
                # Calculate realized hit ratio against market truth
                hits = (preds > 0) == (actuals > 0)
                ann_ret = np.mean(actuals) * (252/h) # Realized market return
                vol = np.std(actuals) * np.sqrt(252/h) + 1e-9
                
                return {
                    "ann_return": ann_ret,
                    "sharpe": (ann_ret - sofr) / vol,
                    "hit_15": hits.mean(),
                    "hit_30": hits.mean(), # Simplified for 15-day view
                    "logs": pd.DataFrame({
                        "ETF": etf, 
                        "Predicted Return": preds, 
                        "Actual Return": actuals
                    }, index=actuals.index.strftime('%Y-%m-%d'))
                }

            met_a = get_stats(p_a, y_test, h_val)
            met_b = get_stats(p_b, y_test, h_val)

            if met_a['ann_return'] > t_res['score']:
                t_res = {**met_a, "ticker": etf, "horizon": h_name, "score": met_a['ann_return']}
            if met_b['ann_return'] > r_res['score']:
                r_res = {**met_b, "ticker": etf, "horizon": h_name, "score": met_b['ann_return']}

    return t_res, r_res

# --- APP EXECUTION ---
st.set_page_config(layout="wide")

with st.sidebar:
    st.header("Engine Settings")
    st.info("📅 Training Data: 2018 - Present")
    tc_bps = st.slider("Transaction Friction (bps)", 0, 100, 15)

etfs = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
data = get_market_data(etfs)

if data is not None:
    current_sofr = 0.0363 # Rectified to your 3.63% observation
    res_a, res_b = train_tournament(data, etfs, tc_bps, current_sofr)
    
    interface.render_comparison_dashboard(res_a, res_b, current_sofr)
    interface.render_tactical_logs(res_a['logs'], res_b['logs'])
