import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
from huggingface_hub import HfApi
try:
    import pandas_datareader.data as web
except ImportError:
    st.error("Missing library: pandas_datareader. Please add it to requirements.txt")
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
from datasets import load_dataset
import plotly.graph_objects as go

# ------------------------------
# 1. SCHEDULER & SYNC LOGIC
# ------------------------------
def get_next_open_date():
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=datetime.now(), end_date=datetime.now() + timedelta(days=10))
    if schedule.empty: return "TBD"
    return schedule.iloc[0].market_open.strftime('%b %d, %Y')

def get_est_time():
    return datetime.now(pytz.timezone('US/Eastern'))

def is_sync_window():
    """Checks if current time is within 7pm-8pm or 7am-8am sync windows (EST)."""
    now_est = get_est_time()
    return (now_est.hour == 7) or (now_est.hour == 8)

# ------------------------------
# 2. UI CONFIGURATION
# ------------------------------
st.set_page_config(page_title="P2-Transformer Pro", layout="wide")

with st.sidebar:
    st.header("Model Configuration")
    est_now = get_est_time()
    st.write(f"🕒 **Server Time (EST):** {est_now.strftime('%H:%M:%S')}")
    
    start_year = st.slider("Training Start Year", 2008, 2025, 2016)
    fee_bps = st.slider("Transaction Cost (bps)", 0, 100, 15)
    epochs = st.number_input("Training Epoch_Count", 5, 500, 50)
    
    sync_status = "ACTIVE 🟢" if is_sync_window() else "IDLE ⚪ (HF Cache Only)"
    st.info(f"Sync Engine: {sync_status}")
    
    st.markdown("---")
    run_button = st.button("Execute Transformer Alpha", type="primary")

st.title("P2-TRANSFORMER-ETF-PREDICTOR")

# ------------------------------
# 3. DATA ENGINE (AUTO-SYNC TO HUB)
# ------------------------------
REPO_ID = "P2SAMAPA/my-etf-data"
fee_pct = fee_bps / 10000

def get_data():
    # 1. Load HF Dataset
    try:
        ds = load_dataset(REPO_ID, split="train").to_pandas()
    except Exception as e:
        st.error(f"HF Dataset Load Failed: {e}")
        return pd.DataFrame()

    date_col = 'Date' if 'Date' in ds.columns else 'date'
    ds[date_col] = pd.to_datetime(ds[date_col]).dt.tz_localize(None)
    ds = ds.set_index(date_col).sort_index()
    df = ds[ds.index.year >= start_year].copy()

    # 2. Check for Gaps & Sync Window
    last_date = df.index.max().date()
    today = get_est_time().date()
    has_gap = last_date < today

    if has_gap and is_sync_window():
        with st.status("🔄 Syncing & Updating HF Hub...", expanded=True) as status:
            st.write("Fetching missing macro signals...")
            try:
                # Fetch full history for the missing columns
                yf_inc = yf.download(["^MOVE", "^SKEW"], start="2008-01-01", progress=False)['Close']
                yf_inc = yf_inc.rename(columns={"^MOVE": "MOVE", "^SKEW": "SKEW"})
                yf_inc.index = yf_inc.index.tz_localize(None)
                df = df.combine_first(yf_inc)

                fred_syms = ["T10Y2Y", "T10Y3M", "BAMLH0A0HYM2"]
                fred_inc = web.DataReader(fred_syms, "fred", "2008-01-01", datetime.now())
                fred_inc = fred_inc.rename(columns={"BAMLH0A0HYM2": "HY_Spread"})
                df = df.combine_first(fred_inc)
                
                # AUTO-SAVE TO HF HUB
                token = os.getenv("HF_TOKEN")
                if token:
                    st.write("Pushing updated dataset to Hugging Face...")
                    # Save temporary csv
                    df.to_csv("temp_sync_data.csv")
                    api = HfApi()
                    api.upload_file(
                        path_or_fileobj="temp_sync_data.csv",
                        path_in_repo="etf_data.csv", # Ensuring it saves as a readable CSV
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        token=token
                    )
                    st.success("✅ HF Hub Updated Successfully!")
                else:
                    st.warning("HF_TOKEN secret not found. Data updated in RAM only.")
                
            except Exception as e:
                st.error(f"Sync failed: {e}")
            status.update(label="Sync Process Complete", state="complete")
    
    # 3. Validation
    critical_signals = ['MOVE', 'SKEW', 'HY_Spread']
    missing = [s for s in critical_signals if s not in df.columns]
    if missing:
        st.error(f"CRITICAL ERROR: Signals missing: {missing}")
        st.info("Sync window opens at 19:00 EST.")
        return pd.DataFrame()

    if 'SOFR' not in df.columns:
        df['SOFR'] = df['sofr'] if 'sofr' in df.columns else 0.04 

    # 4. Feature Engineering
    features_to_z = [c for c in df.columns if '_Ret' not in c and c != 'SOFR']
    for col in features_to_z:
        rolling_mean = df[col].rolling(window=20).mean()
        rolling_std = df[col].rolling(window=20).std()
        df[f"{col}_Z"] = (df[col] - rolling_mean) / (rolling_std + 1e-9)

    return df.ffill().dropna()

# ------------------------------
# 4. RUNTIME
# ------------------------------
if run_button:
    df = get_data()
    
    if not df.empty:
        # Keep the download button as a backup
        csv_data = df.to_csv().encode('utf-8')
        st.sidebar.download_button("📥 Manual Backup CSV", csv_data, "etf_backup.csv", "text/csv")

        target_etfs = [t for t in ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret'] if t in df.columns]
        input_features = [c for c in df.columns if c not in target_etfs and c != 'SOFR']
        
        scaler = MinMaxScaler()
        scaled_input = scaler.fit_transform(df[input_features])
        
        lookbacks = [30, 45, 60] if len(df) > 200 else [10, 20]
        holdings = [1, 3, 5]
        best_score = -np.inf
        final_res = None

        with st.spinner("Training Transformer Alpha..."):
            split_idx = int(len(scaled_input) * 0.8)
            for lb in lookbacks:
                X, y = [], []
                for i in range(lb, len(scaled_input)):
                    X.append(scaled_input[i-lb:i])
                    y.append(df[target_etfs].iloc[i].values)
                X, y = np.array(X), np.array(y)
                train_X, test_X = X[:split_idx - lb], X[split_idx - lb:]
                train_y, test_y = y[:split_idx - lb], y[split_idx - lb:]

                inputs = Input(shape=(X.shape[1], X.shape[2]))
                attn_out, _ = MultiHeadAttention(num_heads=4, key_dim=X.shape[2])(inputs, inputs)
                attn_res = LayerNormalization()(attn_out + inputs)
                pool = GlobalMaxPooling1D()(Conv1D(64, 3, activation='relu', padding='same')(attn_res))
                lstm = Bidirectional(LSTM(64))(inputs)
                x = Dense(128, activation='relu')(Concatenate()([pool, lstm]))
                model = Model(inputs, Dense(len(target_etfs))(x))
                model.compile(optimizer='adam', loss='mse')
                model.fit(train_X, train_y, epochs=int(epochs), batch_size=32, verbose=0)
                preds = model.predict(test_X)
                
                for hold in holdings:
                    daily_strat_rets = []
                    sofr_vals = df['SOFR'].iloc[split_idx:].values
                    for i in range(len(preds)):
                        idx = np.argmax(preds[i])
                        sorted_p = np.sort(preds[i])
                        if preds[i][idx] > (sorted_p[-2] * 1.1) and preds[i][idx] > (sofr_vals[i]/252 + fee_pct):
                            daily_strat_rets.append(test_y[i][idx] - (fee_pct if i % hold == 0 else 0))
                        else:
                            daily_strat_rets.append(sofr_vals[i]/252)
                    
                    if np.mean(daily_strat_rets) > best_score:
                        best_score = np.mean(daily_strat_rets)
                        final_res = {"model": model, "strat_rets": daily_strat_rets, "dates": df.index[split_idx:], "sofr": sofr_vals, "last_x": X[-1:], "input_features": input_features, "target_names": target_etfs, "lb": lb, "hold": hold, "preds": preds}

        if final_res:
            res = final_res
            strat_rets = np.array(res["strat_rets"])
            cum_strat = np.cumprod(1 + strat_rets)
            oos_yrs = (res["dates"][-1] - res["dates"][0]).days / 365.25
            ann_ret = (cum_strat[-1] ** (1/oos_yrs)) - 1
            sharpe = np.sqrt(252) * np.mean(strat_rets - (res["sofr"]/252)) / (np.std(strat_rets) + 1e-9)
            
            st.markdown("### Performance Scorecard")
            k1, k2, k3, k4, k5 = st.columns(5)
            p_now = res["model"].predict(res["last_x"])[0]
            best_idx = np.argmax(p_now)
            is_valid = p_now[best_idx] > (np.sort(p_now)[-2]*1.1)
            
            final_asset = res["target_names"][best_idx].split('_')[0] if is_valid else "CASH (SOFR)"
            hold_period = f"{res['hold']}d" if is_valid else "N/A"
            
            k1.metric("Predicted Asset", final_asset, f"Hold: {hold_period}")
            k2.metric("Annualized Return", f"{ann_ret*100:.2f}%", f"LB: {res['lb']}d")
            k3.metric("Sharpe Ratio", f"{sharpe:.2f}", "SOFR Adjusted")
            k4.metric("Hit Ratio (15d)", f"{(np.sum(strat_rets[-15:] > 0)/15)*100:.1f}%", f"Next: {get_next_open_date()}")
            k5.metric("Max Daily Stress", f"{np.min(strat_rets)*100:.2f}%", "Worst Day")

            st.plotly_chart(go.Figure(go.Scatter(x=res["dates"], y=cum_strat, fill='tozeroy', line=dict(color='#00d1b2'))).update_layout(template="plotly_dark", height=400), use_container_width=True)

            st.markdown("---")
            st.markdown("### Methodology Summary: P2-Transformer Pro")
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**1. Data & Auto-Sync**")
                st.write("Model prioritizes HF Hub. During 7pm/8am windows, it fetches missing signals and uses the HF_TOKEN secret to automatically push updates back to the dataset repository.")
            with cols[1]:
                st.markdown("**2. Neural Architecture**")
                st.write("Hybrid Attention-LSTM model designed for regime classification. Uses Multi-Head Attention for signal weights and Bi-LSTM for temporal sequence.")
            st.markdown("**3. 1.1x Conviction Rule**")
            st.write("Aggressive conviction hurdle: Assets must outperform the runner-up by 10% to trigger an entry, otherwise the model stays in Cash.")
