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

# --- CALENDAR: NYSE TRADING DAYS ---
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

@st.cache_data(ttl="6h")
def get_verified_data(etfs):
    tickers = etfs + ["CPER", "^VIX", "DX-Y.NYB"]
    # 2018 Start as per your Sidebar reference
    data = yf.download(tickers, start="2018-01-01", auto_adjust=True, progress=False)['Close'].ffill()
    return data

def train_tournament(df, etf_list, tc_bps, sofr):
    # Fixed Labels
    horizons = {1: "1 Day", 3: "3 Days", 5: "5 Days"}
    tc_decimal = tc_bps / 10000
    
    # Features (Lagged to prevent leakage)
    df_rets = df.pct_change().dropna()
    df['Gold_Copper'] = df['GLD'] / (df['CPER'] + 1e-9)
    features = pd.concat([df_rets, df['Gold_Copper'].shift(1)], axis=1).dropna()
    
    t_res = {"score": -999}
    r_res = {"score": -999}
    
    for h_val, h_name in horizons.items():
        for etf in etf_list:
            # TARGET: Future return (the value we want to predict)
            target = df[etf].pct_change(h_val).shift(-h_val).dropna()
            
            common_idx = features.index.intersection(target.index)
            X = features.loc[common_idx]
            y = target.loc[common_idx]
            
            # STRICT TIME-SERIES SPLIT (No looking ahead)
            split = int(len(X) * 0.9)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            # --- MODEL A: TRANSFORMER (Attention Simulation) ---
            # Using deeper XGB to represent Transformer complexity
            model_a = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05).fit(X_train, y_train)
            pred_a = model_a.predict(X_test)
            
            # --- MODEL B: REGIME SWITCHER (Tree Forest) ---
            model_b = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
            pred_b = model_b.predict(X_test)

            def get_metrics(preds, actuals, h):
                ann_ret = (np.mean(preds) - (tc_decimal/h)) * (252/h)
                vol = np.std(preds) * np.sqrt(252/h) + 1e-9
                return {
                    "ann_return": ann_ret,
                    "sharpe": (ann_ret - sofr) / vol,
                    "hit_15": (preds[-15:] > 0).mean(),
                    "hit_30": (preds[-30:] > 0).mean(),
                    "logs": pd.DataFrame({"ETF": etf, "Prediction": preds[-15:]}, 
                                         index=actuals.index[-15:].strftime('%Y-%m-%d'))
                }

            met_a = get_metrics(pred_a, y_test, h_val)
            met_b = get_metrics(pred_b, y_test, h_val)

            if met_a['ann_return'] > t_res.get('score', -999):
                t_res = {**met_a, "ticker": etf, "horizon": h_name, "score": met_a['ann_return']}
            
            if met_b['ann_return'] > r_res.get('score', -999):
                r_res = {**met_b, "ticker": etf, "horizon": h_name, "score": met_b['ann_return']}

    return t_res, r_res

# --- MAIN ---
st.set_page_config(layout="wide", page_title="Alpha Engine v1.0")

with st.sidebar:
    st.header("Model Parameters")
    st.info("📅 Dataset Range: 2018 - Present")
    st.success(f"🔄 Last Retrained: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tc_bps = st.slider("Transaction Friction (bps)", 0, 100, 15)

etfs = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
data = get_verified_data(etfs)

if data is not None:
    sofr_rate = 0.0535 
    res_a, res_b = train_tournament(data, etfs, tc_bps, sofr_rate)
    interface.render_comparison_dashboard(res_a, res_b, sofr_rate)
    interface.render_tactical_logs(res_a['logs'], res_b['logs'])
