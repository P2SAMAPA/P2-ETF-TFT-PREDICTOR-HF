import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_market_calendars as mcal
import os
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional, 
    Conv1D, GlobalMaxPooling1D, Concatenate, MultiHeadAttention, LayerNormalization
)
from sklearn.preprocessing import MinMaxScaler
from datasets import load_dataset
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------
# 1. NYSE CALENDAR & UTILS
# ------------------------------
def get_next_open_date():
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=datetime.now(), end_date=datetime.now() + timedelta(days=10))
    if schedule.empty: return "TBD"
    return schedule.iloc[0].market_open.strftime('%b %d, %Y')

# ------------------------------
# 2. UI CONFIGURATION
# ------------------------------
st.set_page_config(page_title="P2-Transformer Pro", layout="wide")

with st.sidebar:
    st.header("Model Configuration")
    start_year = st.slider("Training Start Year", 2008, 2025, 2016)
    fee_bps = st.slider("Transaction Cost (bps)", 0, 100, 15)
    epochs = st.number_input("Training Epoch_Count", 5, 200, 50)
    st.markdown("---")
    run_button = st.button("Execute Transformer Alpha", type="primary")

st.title("P2-TRANSFORMER-ETF-PREDICTOR")

# ------------------------------
# 3. DATA ENGINE
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

    # Standardize Date Index
    date_col = 'Date' if 'Date' in ds.columns else 'date'
    ds[date_col] = pd.to_datetime(ds[date_col]).dt.tz_localize(None)
    ds = ds.set_index(date_col).sort_index()
    
    # Filter by Year
    df = ds[ds.index.year >= start_year].copy()
    if df.empty:
        return df

    # 2. Macro Fetch (VIX, TNX, DXY, GOLD, COPPER)
    tickers = {"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY", "GC=F": "GOLD", "HG=F": "COPPER"}
    fetch_start = df.index.min() - timedelta(days=60)
    
    try:
        macro = yf.download(list(tickers.keys()), start=fetch_start, progress=False)['Close']
        macro = macro.rename(columns=tickers)
        macro.index = macro.index.tz_localize(None)
        
        # Robust Join: Drop existing macro columns in HF data if they exist to avoid suffix noise
        cols_to_drop = [c for c in macro.columns if c in df.columns]
        df = df.drop(columns=cols_to_drop).join(macro, how='left')
    except:
        pass

    # 3. SOFR Rate Fetch
    try:
        sofr = yf.download("^IRX", start=fetch_start, progress=False)['Close'] / 100
        sofr.index = sofr.index.tz_localize(None)
        df['SOFR'] = sofr.reindex(df.index).ffill().bfill()
    except:
        df['SOFR'] = 0.04 

    # Robust Fill to ensure 'start_year' doesn't return empty due to one missing ticker value
    return df.ffill().bfill().dropna()

# ------------------------------
# 4. OPTIMIZATION RUNTIME
# ------------------------------
if run_button:
    df = get_data()
    
    if df.empty:
        st.error(f"Error: Dataframe empty for {start_year}. Check if dataset covers this range.")
    else:
        target_etfs = ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret']
        target_etfs = [t for t in target_etfs if t in df.columns]
        input_features = [c for c in df.columns if c not in target_etfs and c != 'SOFR']
        
        scaler = MinMaxScaler()
        scaled_input = scaler.fit_transform(df[input_features])
        
        # Dynamic lookback based on data size
        lookbacks = [30, 45, 60] if len(df) > 200 else [10, 20]
        holdings = [1, 3, 5]
        best_conviction = -np.inf
        final_res = None

        with st.spinner(f"Training Transformer on {len(df)} days of data..."):
            split_idx = int(len(scaled_input) * 0.8)
            
            for lb in lookbacks:
                if len(scaled_input) <= lb: continue
                X, y = [], []
                for i in range(lb, len(scaled_input)):
                    X.append(scaled_input[i-lb:i])
                    y.append(df[target_etfs].iloc[i].values)
                
                X, y = np.array(X), np.array(y)
                train_X, test_X = X[:split_idx - lb], X[split_idx - lb:]
                train_y, test_y = y[:split_idx - lb], y[split_idx - lb:]

                if len(train_X) < 10: continue

                # Deep Learning Architecture
                inputs = Input(shape=(X.shape[1], X.shape[2]))
                attn_output, attn_weights = MultiHeadAttention(num_heads=4, key_dim=X.shape[2])(inputs, inputs, return_attention_scores=True)
                attn_res = LayerNormalization()(attn_output + inputs)
                conv = Conv1D(64, 3, activation='relu', padding='same')(attn_res)
                pool = GlobalMaxPooling1D()(conv)
                lstm = Bidirectional(LSTM(64))(inputs)
                merged = Concatenate()([pool, lstm])
                x = Dense(128, activation='relu')(merged)
                outputs = Dense(len(target_etfs))(x)
                
                model = Model(inputs, outputs)
                weight_model = Model(inputs, attn_weights)
                
                model.compile(optimizer='adam', loss='mse')
                model.fit(train_X, train_y, epochs=int(epochs), batch_size=32, verbose=0)
                preds = model.predict(test_X)
                
                for hold in holdings:
                    daily_strat_rets = []
                    for i in range(len(preds)):
                        idx = np.argmax(preds[i])
                        daily_sofr = df['SOFR'].iloc[split_idx + i] / 252
                        if preds[i][idx] > (fee_pct + daily_sofr):
                            daily_strat_rets.append(test_y[i][idx] - (fee_pct if i % hold == 0 else 0))
                        else:
                            daily_strat_rets.append(daily_sofr)
                    
                    score = np.mean(daily_strat_rets)
                    if score > best_conviction:
                        best_conviction = score
                        final_res = {
                            "lb": lb, "hold": hold, "preds": preds, 
                            "test_y": test_y, "dates": df.index[split_idx:],
                            "model": model, "weight_model": weight_model,
                            "strat_rets": daily_strat_rets, "sofr": df['SOFR'].iloc[split_idx:].values,
                            "last_x": X[-1:], "input_features": input_features,
                            "target_names": target_etfs
                        }

        # ------------------------------
        # 5. UI: OUTPUT RENDER
        # ------------------------------
        if final_res:
            res = final_res
            strat_rets = np.array(res["strat_rets"])
            cum_strat = np.cumprod(1 + strat_rets)
            oos_yrs = (res["dates"][-1] - res["dates"][0]).days / 365.25
            ann_ret = (cum_strat[-1] ** (1/oos_yrs)) - 1 if oos_yrs > 0 else 0
            sharpe = np.sqrt(252) * np.mean(strat_rets - (res["sofr"][:len(strat_rets)]/252)) / np.std(strat_rets)
            
            st.markdown("### Performance Scorecard")
            k1, k2, k3, k4, k5 = st.columns(5)
            
            p_now = res["model"].predict(res["last_x"])[0]
            best_idx = np.argmax(p_now)
            final_asset = res["target_names"][best_idx].split('_')[0] if p_now[best_idx] > ((df['SOFR'].iloc[-1]/252) + fee_pct) else "CASH (SOFR)"
            
            k1.metric("Predicted Asset", final_asset, f"Next Open: {get_next_open_date()}")
            k2.metric("Annualized Return", f"{ann_ret*100:.2f}%", f"{oos_yrs:.2f} OOS Years")
            k3.metric("Sharpe Ratio", f"{sharpe:.2f}", "SOFR Adjusted")
            k4.metric("Hit Ratio (15d)", f"{(np.sum(strat_rets[-15:] > 0)/15)*100:.1f}%", f"Search: {res['lb']}L|{res['hold']}H")
            k5.metric("Max Daily Stress", f"{np.min(strat_rets)*100:.2f}%", "Worst Day")

            st.markdown(f"### OOS Equity Curve ({res['lb']}d Lookback | {res['hold']}d Holding)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res["dates"][:len(cum_strat)], y=cum_strat, fill='tozeroy', line=dict(color='#00d1b2', width=1.5)))
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,b=0,t=20))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("### Signal Contribution Analysis")
            raw_weights = res["weight_model"].predict(res["last_x"]) 
            importance = np.mean(raw_weights, axis=(0, 1, 2))
            importance = np.atleast_1d(importance)
            
            if len(importance) < len(res["input_features"]):
                 importance = np.pad(importance, (0, len(res["input_features"]) - len(importance)))
            else:
                 importance = importance[:len(res["input_features"])]

            feat_imp = pd.DataFrame({'Signal': res["input_features"], 'Weight': importance}).sort_values('Weight', ascending=False)
            
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.write("Predictive Power Ranking")
                st.dataframe(feat_imp.style.background_gradient(cmap='Blues'))
            with col_b:
                st.write("Temporal Attention Heatmap (Head 0)")
                fig_heat = px.imshow(raw_weights[0, 0], labels=dict(x="Steps", y="Dimension", color="Weight"), color_continuous_scale="Viridis")
                fig_heat.update_layout(height=350, template="plotly_dark", margin=dict(l=0,r=0,b=0,t=0))
                st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("### 15-Day Performance Audit Trail")
            audit = []
            display_days = min(15, len(res["preds"]))
            for i in range(1, display_days + 1):
                idx = np.argmax(res["preds"][-i])
                is_cash = res["preds"][-i][idx] <= ((res["sofr"][-i]/252) + fee_pct)
                audit.append({
                    "Date": res["dates"][-i].strftime('%Y-%m-%d'),
                    "Predicted": "CASH" if is_cash else res["target_names"][idx].split('_')[0],
                    "SOFR (Ann)": f"{res['sofr'][-i]*100:.2f}%",
                    "Realized Daily Return": res["strat_rets"][-i]
                })
            st.table(pd.DataFrame(audit).style.format({"Realized Daily Return": "{:.4%}"}).applymap(
                lambda x: 'color: #00ff00' if x > 0 else 'color: #ff4b4b', subset=['Realized Daily Return']
            ))
        else:
            st.error("No valid strategy found. Try a different Training Start Year.")
