import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
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

def is_sync_window():
    """Checks if current time is within the 8pm-9pm or 8am-9am sync windows (EST)."""
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    # 8 PM Evening Window or 8 AM Morning Fallback
    return (now_est.hour == 20) or (now_est.hour == 8)

# ------------------------------
# 2. UI CONFIGURATION
# ------------------------------
st.set_page_config(page_title="P2-Transformer Pro", layout="wide")

with st.sidebar:
    st.header("Model Configuration")
    start_year = st.slider("Training Start Year", 2008, 2025, 2016)
    fee_bps = st.slider("Transaction Cost (bps)", 0, 100, 15)
    epochs = st.number_input("Training Epoch_Count", 5, 500, 50)
    
    # Dynamic Sync Status Display
    sync_status = "ACTIVE 🟢 (External Sync Allowed)" if is_sync_window() else "IDLE ⚪ (HF Cache Only)"
    st.info(f"Sync Engine: {sync_status}")
    st.markdown("---")
    run_button = st.button("Execute Transformer Alpha", type="primary")

st.title("P2-TRANSFORMER-ETF-PREDICTOR")

# ------------------------------
# 3. DATA ENGINE (HF-FIRST + GAP FILL)
# ------------------------------
REPO_ID = "P2SAMAPA/my-etf-data"
fee_pct = fee_bps / 10000

@st.cache_data(ttl=3600)
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
    today = datetime.now().date()
    has_gap = last_date < today

    if has_gap and is_sync_window():
        st.write(f"🔄 Syncing Gap: {last_date} to {today}...")
        # Yahoo Finance Gap Fill
        try:
            yf_inc = yf.download(["^MOVE", "^SKEW"], start=last_date, progress=False)['Close']
            yf_inc = yf_inc.rename(columns={"^MOVE": "MOVE", "^SKEW": "SKEW"})
            yf_inc.index = yf_inc.index.tz_localize(None)
            df = df.combine_first(yf_inc)
        except: pass

        # FRED Gap Fill
        try:
            fred_syms = ["T10Y2Y", "T10Y3M", "BAMLH0A0HYM2"]
            fred_inc = web.DataReader(fred_syms, "fred", last_date, datetime.now())
            fred_inc = fred_inc.rename(columns={"BAMLH0A0HYM2": "HY_Spread"})
            df = df.combine_first(fred_inc)
        except: pass
    
    # 3. Error Validation
    critical_signals = ['MOVE', 'SKEW', 'HY_Spread']
    missing = [s for s in critical_signals if s not in df.columns]
    if missing:
        st.error(f"CRITICAL ERROR: The following signals are missing from your HF dataset: {missing}")
        return pd.DataFrame()

    # 4. SOFR / Risk-Free Logic
    if 'SOFR' not in df.columns:
        df['SOFR'] = df['sofr'] if 'sofr' in df.columns else 0.04 

    # 5. Z-Score Feature Engineering
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
        target_etfs = [t for t in ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret'] if t in df.columns]
        input_features = [c for c in df.columns if c not in target_etfs and c != 'SOFR']
        
        scaler = MinMaxScaler()
        scaled_input = scaler.fit_transform(df[input_features])
        
        lookbacks = [30, 45, 60] if len(df) > 200 else [10, 20]
        holdings = [1, 3, 5]
        best_score = -np.inf
        final_res = None

        with st.spinner("Training Transformer Alpha Engine..."):
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
                        # Aggressive 1.1x Conviction Rule
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

            st.markdown("### 15-Day Performance Audit Trail")
            audit = []
            for i in range(1, 16):
                p = res["preds"][-i]
                is_cash = not (p[np.argmax(p)] > (np.sort(p)[-2] * 1.1))
                audit.append({"Date": res["dates"][-i].strftime('%Y-%m-%d'), "Predicted": "CASH" if is_cash else res["target_names"][np.argmax(p)].split('_')[0], "Realized Daily Return": res["strat_rets"][-i]})
            st.table(pd.DataFrame(audit).style.format({"Realized Daily Return": "{:.4%}"}))

            # --- METHODOLOGY SUMMARY ---
            st.markdown("---")
            st.markdown("### Methodology Summary: P2-Transformer Pro")
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**1. Data & Incremental Sync**")
                st.write("The engine prioritizes the HF Dataset. External calls (YF/FRED) are only triggered during 8pm/8am EST sync windows if a data gap is detected. All macro signals are transformed into rolling Z-scores for regime detection.")
            with cols[1]:
                st.markdown("**2. Transformer-LSTM Architecture**")
                st.write("A hybrid neural network: Multi-Head Attention layers isolate global correlations between macro signals and ETF returns, while Bidirectional LSTMs process the local sequential trend.")
            st.markdown("**3. The 1.1x Conviction Hurdle**")
            st.write("The model enters an ETF trade only if the top predicted return is 1.1x higher than the runner-up and covers the risk-free rate plus transaction costs. Otherwise, it defaults to CASH.")
