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
from datetime import datetime, timedelta
import pytz
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Bidirectional, 
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
    now_est = get_est_time()
    return (now_est.hour >= 7 and now_est.hour <= 9) or (now_est.hour >= 18 and now_est.hour <= 21)

st.set_page_config(page_title="P2-Transformer Pro", layout="wide")

# ------------------------------
# 2. DATA ENGINE (ALIGNED MACRO)
# ------------------------------
def get_data(start_year):
    raw_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/etf_data.csv"
    try:
        df = pd.read_csv(raw_url)
        df.columns = df.columns.str.strip()
        # Fix the Unnamed: 0 issue seen in Dataset Viewer
        date_col = next((c for c in df.columns if c.lower() in ['date', 'unnamed: 0']), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        df = df.set_index(date_col).sort_index()
    except Exception as e:
        st.error(f"Fetch failed: {e}")
        return pd.DataFrame()

    if is_sync_window():
        with st.status("🔄 Synchronizing Aligned Macro Signals...", expanded=False):
            try:
                # Syncing MOVE/SKEW specifically to fill the 'Signal Blindness'
                yf_inc = yf.download(["^MOVE", "^SKEW"], start="2008-01-01", progress=False)['Close']
                yf_inc = yf_inc.rename(columns={"^MOVE": "MOVE", "^SKEW": "SKEW"}).tz_localize(None)
                
                # FRED Signals for regime detection
                fred_inc = web.DataReader(["T10Y2Y", "T10Y3M", "BAMLH0A0HYM2"], "fred", "2008-01-01", datetime.now())
                fred_inc = fred_inc.rename(columns={"BAMLH0A0HYM2": "HY_Spread"})
                
                # Combine with strict forward-fill to prevent NaN gaps during training
                df = df.combine_first(yf_inc).combine_first(fred_inc)
                
                token = os.getenv("HF_TOKEN")
                if token:
                    df.index.name = "Date"
                    df.to_csv("etf_data.csv", index=True)
                    HfApi().upload_file(path_or_fileobj="etf_data.csv", path_in_repo="etf_data.csv", repo_id=REPO_ID, repo_type="dataset", token=token)
            except: pass

    # Robust Feature Engineering: Using Z-Scores for all Macro Inputs
    macro_cols = ['MOVE', 'SKEW', 'HY_Spread', 'T10Y2Y', 'T10Y3M', 'VIX', 'DXY', 'COPPER', 'GOLD']
    for col in [c for c in df.columns if c in macro_cols or '_Vol' in c]:
        df[f"{col}_Z"] = (df[col] - df[col].rolling(60).mean()) / (df[col].rolling(60).std() + 1e-9)

    return df[df.index.year >= start_year].ffill().dropna()

# ------------------------------
# 3. SIDEBAR
# ------------------------------
with st.sidebar:
    st.header("Model Configuration")
    st.write(f"🕒 Server Time (EST): {get_est_time().strftime('%H:%M:%S')}")
    start_yr = st.slider("Start Year", 2008, 2025, 2016)
    fee_bps = st.slider("Fee (bps)", 0, 100, 15)
    epochs = st.number_input("Epochs", 5, 500, 100)
    run_button = st.button("Execute Transformer Alpha", type="primary")

st.title("P2-TRANSFORMER-ETF-PREDICTOR")

# ------------------------------
# 4. ANALYTICS & UI RESTORATION
# ------------------------------
if run_button:
    df = get_data(start_yr)
    if not df.empty:
        target_etfs = [t for t in ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret'] if t in df.columns]
        # Include the new Macro Z-scores in the feature set
        input_features = [c for c in df.columns if (c.endswith('_Z') or c.endswith('_Vol')) and c not in target_etfs]
        
        scaler = MinMaxScaler()
        scaled_input = scaler.fit_transform(df[input_features])
        lb = 30
        
        with st.spinner("Training Transformer Alpha Engine..."):
            X, y = [], []
            for i in range(lb, len(scaled_input)):
                X.append(scaled_input[i-lb:i])
                y.append(df[target_etfs].iloc[i].values)
            X, y = np.array(X), np.array(y)
            split = int(len(X) * 0.8)
            
            # Restored Transformer Architecture
            inputs = Input(shape=(X.shape[1], X.shape[2]))
            attn_out = MultiHeadAttention(num_heads=4, key_dim=X.shape[2])(inputs, inputs)
            attn_res = LayerNormalization()(attn_out + inputs)
            pool = GlobalMaxPooling1D()(Conv1D(64, 3, activation='relu', padding='same')(attn_res))
            lstm = Bidirectional(tf.keras.layers.LSTM(64))(inputs)
            x = Dense(128, activation='relu')(Concatenate()([pool, lstm]))
            model = Model(inputs, Dense(len(target_etfs))(x))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X[:split], y[:split], epochs=int(epochs), batch_size=32, verbose=0)
            
            preds = model.predict(X[split:])
            oos_dates = df.index[lb + split:]
            sofr = df['SOFR'].iloc[-1] if 'SOFR' in df.columns else 0.045
            
            strat_rets, audit_trail = [], []
            for i in range(len(preds)):
                idx = np.argmax(preds[i])
                signal = target_etfs[idx].split('_')[0]
                realized = y[split:][i][idx]
                net_ret = realized - (fee_bps/10000)
                strat_rets.append(net_ret)
                audit_trail.append({'Date': oos_dates[i].strftime('%Y-%m-%d'), 'Signal': signal, 'Return': net_ret})

            # --- RESTORED ANALYTICS UI ---
            next_day = (df.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
            st.success(f"🎯 **Target Allocation for Next Session ({next_day}): {audit_trail[-1]['Signal']}**")

            # Corrected Metrics
            cum_strat = np.cumprod(1 + np.array(strat_rets))
            ann_ret = (cum_strat[-1]**(252/len(strat_rets)) - 1)
            sharpe = (np.mean(strat_rets) - (sofr/252)) / (np.std(strat_rets) + 1e-9) * np.sqrt(252)
            # FIXED: Hit Ratio now correctly checks if Return > 0
            hit_ratio = np.mean([1 for r in strat_rets[-15:] if r > 0])
            max_dd = np.min(np.array(strat_rets))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ann. Return (OOS)", f"{ann_ret*100:.2f}%")
            c2.metric("Sharpe (Live SOFR)", f"{sharpe:.2f}")
            c3.metric("Hit Ratio (15d)", f"{hit_ratio*100:.0f}%")
            c4.metric("Max Daily DD", f"{max_dd*100:.2f}%")

            st.plotly_chart(go.Figure(go.Scatter(x=oos_dates, y=cum_strat, fill='tozeroy', line=dict(color='#00d1b2'))).update_layout(template="plotly_dark", title="Out-of-Sample Equity Curve"))

            # Restored Feature Importance as a % Graph
            st.subheader("Transformer Signal Strength (%)")
            weights = np.mean(np.abs(model.layers[-1].get_weights()[0]), axis=1)
            importance = (weights / weights.sum()) * 100
            feat_imp = pd.Series(importance[:len(input_features)], index=input_features).sort_values(ascending=False).head(12)
            
            fig_imp = go.Figure(go.Bar(x=feat_imp.values, y=feat_imp.index, orientation='h', marker_color='#3b82f6'))
            fig_imp.update_layout(template="plotly_dark", xaxis_title="Contribution %", height=400)
            st.plotly_chart(fig_imp)

            # Restored 15-Day Audit Trail
            st.subheader("Last 15 Days Audit Trail")
            audit_df = pd.DataFrame(audit_trail).tail(15)
            def color_ret(val):
                return 'color: #00ff00' if val > 0 else 'color: #ff4b4b'
            st.table(audit_df.style.applymap(color_ret, subset=['Return']).format({'Return': '{:.2%}'}))
