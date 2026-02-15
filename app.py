import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
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
# 1. UTILS & SYNC LOGIC
# ------------------------------
def get_next_open_date():
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=datetime.now(), end_date=datetime.now() + timedelta(days=10))
    if schedule.empty: return "TBD"
    return schedule.iloc[0].market_open.strftime('%b %d, %Y')

def check_sync_status():
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    if now_est.hour >= 18:
        return f"sync_{now_est.strftime('%Y-%m-%d')}_post6pm"
    return f"sync_{now_est.strftime('%Y-%m-%d')}_pre6pm"

# ------------------------------
# 2. UI CONFIGURATION
# ------------------------------
st.set_page_config(page_title="P2-Transformer Pro", layout="wide")

with st.sidebar:
    st.header("Model Configuration")
    start_year = st.slider("Training Start Year", 2008, 2025, 2016)
    fee_bps = st.slider("Transaction Cost (bps)", 0, 100, 15)
    epochs = st.number_input("Training Epoch_Count", 5, 200, 50)
    st.info(f"Data Sync: {check_sync_status()}")
    st.markdown("---")
    run_button = st.button("Execute Transformer Alpha", type="primary")

st.title("P2-TRANSFORMER-ETF-PREDICTOR")

# ------------------------------
# 3. DATA ENGINE (FRED + YF + HF)
# ------------------------------
REPO_ID = "P2SAMAPA/my-etf-data"
fee_pct = fee_bps / 10000

@st.cache_data(ttl=3600)
def get_data(sync_key):
    try:
        ds = load_dataset(REPO_ID, split="train").to_pandas()
    except Exception as e:
        st.error(f"HF Dataset Load Failed: {e}")
        return pd.DataFrame()

    date_col = 'Date' if 'Date' in ds.columns else 'date'
    ds[date_col] = pd.to_datetime(ds[date_col]).dt.tz_localize(None)
    ds = ds.set_index(date_col).sort_index()
    df = ds[ds.index.year >= start_year].copy()
    
    # --- SPLIT FETCH LOGIC ---
    # 1. MOVE and SKEW from YFinance
    try:
        yf_indices = yf.download(["^MOVE", "^SKEW"], start=df.index.min(), progress=False)['Close']
        yf_indices = yf_indices.rename(columns={"^MOVE": "MOVE", "^SKEW": "SKEW"})
        yf_indices.index = yf_indices.index.tz_localize(None)
        df = df.join(yf_indices, how='left')
    except:
        st.warning("YFinance Volatility fetch failed.")

    # 2. 10Y2Y, 10Y3M, HY_Spread from FRED
    try:
        fred_symbols = ["T10Y2Y", "T10Y3M", "BAMLH0A0HYM2"]
        fred_data = web.DataReader(fred_symbols, "fred", df.index.min(), datetime.now())
        fred_data = fred_data.rename(columns={"BAMLH0A0HYM2": "HY_Spread"})
        df = df.join(fred_data, how='left')
    except:
        st.warning("FRED Macro fetch failed.")

    # 3. Z-SCORE ENGINEERING (Idea #1)
    features_to_z = [c for c in df.columns if '_Ret' not in c and c != 'SOFR']
    for col in features_to_z:
        rolling_mean = df[col].rolling(window=20).mean()
        rolling_std = df[col].rolling(window=20).std()
        df[f"{col}_Z"] = (df[col] - rolling_mean) / (rolling_std + 1e-9)

    return df.ffill().bfill().dropna()

# ------------------------------
# 4. RUNTIME
# ------------------------------
if run_button:
    df = get_data(check_sync_status())
    
    if df.empty:
        st.error("Error: Dataset empty.")
    else:
        target_etfs = ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret']
        target_etfs = [t for t in target_etfs if t in df.columns]
        input_features = [c for c in df.columns if c not in target_etfs and c != 'SOFR']
        
        scaler = MinMaxScaler()
        scaled_input = scaler.fit_transform(df[input_features])
        
        lookbacks = [30, 45, 60] if len(df) > 200 else [10, 20]
        holdings = [1, 3, 5]
        best_score = -np.inf
        final_res = None

        with st.spinner("Training Transformer with Macro Signals..."):
            split_idx = int(len(scaled_input) * 0.8)
            for lb in lookbacks:
                X, y = [], []
                for i in range(lb, len(scaled_input)):
                    X.append(scaled_input[i-lb:i])
                    y.append(df[target_etfs].iloc[i].values)
                X, y = np.array(X), np.array(y)
                train_X, test_X = X[:split_idx - lb], X[split_idx - lb:]
                train_y, test_y = y[:split_idx - lb], y[split_idx - lb:]

                # Architecture
                inputs = Input(shape=(X.shape[1], X.shape[2]))
                attn_out, _ = MultiHeadAttention(num_heads=4, key_dim=X.shape[2])(inputs, inputs, return_attention_scores=True)
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
                    sofr = df['SOFR'].iloc[split_idx:].values
                    for i in range(len(preds)):
                        idx = np.argmax(preds[i])
                        sorted_p = np.sort(preds[i])
                        # 4. CONVICTION GATING (Idea #2)
                        if preds[i][idx] > (sorted_p[-2] * 1.2) and preds[i][idx] > (sofr[i]/252 + fee_pct):
                            daily_strat_rets.append(test_y[i][idx] - (fee_pct if i % hold == 0 else 0))
                        else:
                            daily_strat_rets.append(sofr[i]/252)
                    
                    if np.mean(daily_strat_rets) > best_score:
                        best_score = np.mean(daily_strat_rets)
                        final_res = {"model": model, "strat_rets": daily_strat_rets, "dates": df.index[split_idx:], "sofr": sofr, "last_x": X[-1:], "input_features": input_features, "target_names": target_etfs, "lb": lb, "hold": hold, "preds": preds}

        if final_res:
            res = final_res
            strat_rets = np.array(res["strat_rets"])
            cum_strat = np.cumprod(1 + strat_rets)
            oos_yrs = (res["dates"][-1] - res["dates"][0]).days / 365.25
            ann_ret = (cum_strat[-1] ** (1/oos_yrs)) - 1
            sharpe = np.sqrt(252) * np.mean(strat_rets - (res["sofr"]/252)) / np.std(strat_rets)
            
            st.markdown("### Performance Scorecard")
            k1, k2, k3, k4, k5 = st.columns(5)
            p_now = res["model"].predict(res["last_x"])[0]
            best_idx = np.argmax(p_now)
            final_asset = res["target_names"][best_idx].split('_')[0] if p_now[best_idx] > (np.sort(p_now)[-2]*1.2) else "CASH (SOFR)"
            k1.metric("Predicted Asset", final_asset, f"Next Open: {get_next_open_date()}")
            k2.metric("Annualized Return", f"{ann_ret*100:.2f}%", f"{oos_yrs:.2f} OOS Years")
            k3.metric("Sharpe Ratio", f"{sharpe:.2f}", "SOFR Adjusted")
            k4.metric("Hit Ratio (15d)", f"{(np.sum(strat_rets[-15:] > 0)/15)*100:.1f}%", f"Lookback: {res['lb']}d")
            k5.metric("Max Daily Stress", f"{np.min(strat_rets)*100:.2f}%", "Worst Day")

            st.markdown("### OOS Equity Curve")
            fig = go.Figure(go.Scatter(x=res["dates"], y=cum_strat, fill='tozeroy', line=dict(color='#00d1b2')))
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,b=0,t=20))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Signal Contribution Analysis")
            weights = np.abs(res["model"].layers[-2].get_weights()[0]).mean(axis=1)[:len(res["input_features"])]
            norm_w = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
            st.dataframe(pd.DataFrame({'Signal': res["input_features"], 'Relative Power': norm_w}).sort_values('Relative Power', ascending=False),
                         column_config={"Relative Power": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")}, hide_index=True, use_container_width=True)

            st.markdown("### 15-Day Performance Audit Trail")
            audit = []
            for i in range(1, 16):
                p = res["preds"][-i]
                is_cash = not (p[np.argmax(p)] > (np.sort(p)[-2] * 1.2))
                audit.append({"Date": res["dates"][-i].strftime('%Y-%m-%d'), "Predicted": "CASH" if is_cash else res["target_names"][np.argmax(p)].split('_')[0], "Realized Daily Return": res["strat_rets"][-i]})
            st.table(pd.DataFrame(audit).style.format({"Realized Daily Return": "{:.4%}"}).applymap(lambda x: 'color: #00ff00' if x > 0 else 'color: #ff4b4b', subset=['Realized Daily Return']))
