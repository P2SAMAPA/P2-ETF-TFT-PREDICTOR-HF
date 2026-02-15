import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
from huggingface_hub import HfApi
try:
    import pandas_datareader.data as web
except ImportError:
    st.error("Missing pandas_datareader. Please add it to requirements.txt")
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import pytz
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional, 
    Conv1D, GlobalMaxPooling1D, Concatenate, MultiHeadAttention, LayerNormalization
)
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# ------------------------------
# 1. CORE CONFIG & SYNC WINDOW
# ------------------------------
REPO_ID = "P2SAMAPA/my-etf-data"

def get_est_time():
    return datetime.now(pytz.timezone('US/Eastern'))

def is_sync_window():
    # Adjusted to match your 07:06 server time
    now_est = get_est_time()
    return (now_est.hour >= 7 and now_est.hour <= 9) or (now_est.hour >= 18 and now_est.hour <= 20)

st.set_page_config(page_title="P2-Transformer Pro", layout="wide")

# ------------------------------
# 2. DATA ENGINE (RAW CSV FETCH)
# ------------------------------
def get_data(start_year):
    # FORCE FETCH RAW CSV to bypass the old 'data/' folder
    raw_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/etf_data.csv"
    try:
        df = pd.read_csv(raw_url)
    except Exception as e:
        st.error(f"Raw CSV fetch failed. Ensure etf_data.csv is in root. Error: {e}")
        return pd.DataFrame()

    # Standardize Index
    date_col = 'Date' if 'Date' in df.columns else 'date'
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    df = df.set_index(date_col).sort_index()
    df = df[df.index.year >= start_year].copy()

    # Sync Missing Signals if in window
    if is_sync_window():
        with st.status("🔄 Syncing Macro Signals...", expanded=False):
            try:
                yf_inc = yf.download(["^MOVE", "^SKEW"], start="2008-01-01", progress=False)['Close']
                yf_inc = yf_inc.rename(columns={"^MOVE": "MOVE", "^SKEW": "SKEW"}).tz_localize(None)
                df = df.combine_first(yf_inc)

                fred_inc = web.DataReader(["T10Y2Y", "T10Y3M", "BAMLH0A0HYM2"], "fred", "2008-01-01", datetime.now())
                fred_inc = fred_inc.rename(columns={"BAMLH0A0HYM2": "HY_Spread"})
                df = df.combine_first(fred_inc)
                
                token = os.getenv("HF_TOKEN")
                if token:
                    df.to_csv("etf_data.csv")
                    HfApi().upload_file(path_or_fileobj="etf_data.csv", path_in_repo="etf_data.csv", repo_id=REPO_ID, repo_type="dataset", token=token)
                    st.toast("✅ Hub Pushed Successfully!")
            except Exception as e:
                st.warning(f"Sync Warning: {e}")

    # Feature Engineering (Z-Scores)
    for col in [c for c in df.columns if '_Ret' not in c and c != 'SOFR']:
        df[f"{col}_Z"] = (df[col] - df[col].rolling(20).mean()) / (df[col].rolling(20).std() + 1e-9)

    return df.ffill().dropna()

# ------------------------------
# 3. SIDEBAR & UI
# ------------------------------
with st.sidebar:
    st.header("Model Configuration")
    st.write(f"🕒 Server Time (EST): {get_est_time().strftime('%H:%M:%S')}")
    start_yr = st.slider("Start Year", 2008, 2025, 2016)
    fee_bps = st.slider("Fee (bps)", 0, 100, 15)
    epochs = st.number_input("Epochs", 5, 500, 50)
    st.info(f"Sync Engine: {'ACTIVE 🟢' if is_sync_window() else 'IDLE ⚪'}")
    run_button = st.button("Execute Transformer Alpha", type="primary")

st.title("P2-TRANSFORMER-ETF-PREDICTOR")

# ------------------------------
# 4. EXECUTION
# ------------------------------
if run_button:
    df = get_data(start_yr)
    if not df.empty:
        target_etfs = [t for t in ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret'] if t in df.columns]
        input_features = [c for c in df.columns if c not in target_etfs and c != 'SOFR']
        
        # Validate critical signals
        missing = [s for s in ['MOVE', 'SKEW', 'HY_Spread'] if s not in df.columns]
        if missing:
            st.error(f"Still missing: {missing}. Run again during Sync Window.")
        else:
            scaler = MinMaxScaler()
            scaled_input = scaler.fit_transform(df[input_features])
            lb = 30
            
            with st.spinner("Training Transformer..."):
                X, y = [], []
                for i in range(lb, len(scaled_input)):
                    X.append(scaled_input[i-lb:i])
                    y.append(df[target_etfs].iloc[i].values)
                X, y = np.array(X), np.array(y)
                split = int(len(X) * 0.8)
                
                inputs = Input(shape=(X.shape[1], X.shape[2]))
                # FIXED: Removed symbolic iteration (no underscore)
                attn_out = MultiHeadAttention(num_heads=4, key_dim=X.shape[2])(inputs, inputs)
                attn_res = LayerNormalization()(attn_out + inputs)
                pool = GlobalMaxPooling1D()(Conv1D(64, 3, activation='relu', padding='same')(attn_res))
                lstm = Bidirectional(LSTM(64))(inputs)
                x = Dense(128, activation='relu')(Concatenate()([pool, lstm]))
                model = Model(inputs, Dense(len(target_etfs))(x))
                model.compile(optimizer='adam', loss='mse')
                model.fit(X[:split], y[:split], epochs=int(epochs), batch_size=32, verbose=0)
                
                # Backtest Result
                preds = model.predict(X[split:])
                strat_rets = []
                for i in range(len(preds)):
                    idx = np.argmax(preds[i])
                    if preds[i][idx] > (np.sort(preds[i])[-2] * 1.1):
                        strat_rets.append(y[split:][i][idx] - (fee_bps/10000))
                    else:
                        strat_rets.append(0.04/252)

                cum_strat = np.cumprod(1 + np.array(strat_rets))
                st.plotly_chart(go.Figure(go.Scatter(y=cum_strat, fill='tozeroy', line=dict(color='#00d1b2'))).update_layout(template="plotly_dark"))
