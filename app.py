import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
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
    """Checks if we are past 6:00 PM EST for the daily incremental update."""
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    # Returns a unique key for caching based on the date and sync hour
    if now_est.hour >= 18:
        return f"sync_{now_est.strftime('%Y-%m-%d')}_post6pm"
    return f"sync_{now_est.strftime('%Y-%m-%d')}_pre6pm"

# ------------------------------
# 2. UI CONFIGURATION
# ------------------------------
st.set_page_config(page_title="P2-Transformer Alpha", layout="wide")

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

@st.cache_data(ttl=3600) # Re-check sync every hour
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
    
    # --- ADD EXTERNAL MACRO SIGNALS ---
    # Tickers for FRED via YFinance (mirrored), MOVE, and SKEW
    ext_tickers = {
        "^MOVE": "MOVE", 
        "^SKEW": "SKEW",
        "T10Y2Y": "T10Y2Y", 
        "T10Y3M": "T10Y3M",
        "BAMLH0A0HYM2": "HY_Spread" # High Yield Option-Adjusted Spread
    }
    
    try:
        ext_data = yf.download(list(ext_tickers.keys()), start=df.index.min(), progress=False)['Close']
        ext_data = ext_data.rename(columns=ext_tickers)
        ext_data.index = ext_data.index.tz_localize(None)
        df = df.join(ext_data, how='left')
    except:
        st.warning("External signal fetch (MOVE/FRED) failed. Using HF core only.")

    # 1. Z-SCORE FEATURE ENGINEERING (Idea #1)
    # Calculates relative deviation to help Transformer see "extremes"
    features_to_z = [c for c in df.columns if '_Ret' not in c and c != 'SOFR']
    for col in features_to_z:
        rolling_mean = df[col].rolling(window=20).mean()
        rolling_std = df[col].rolling(window=20).std()
        df[f"{col}_Z"] = (df[col] - rolling_mean) / (rolling_std + 1e-9)

    return df.ffill().bfill().dropna()

# ------------------------------
# 4. OPTIMIZATION RUNTIME
# ------------------------------
if run_button:
    df = get_data(check_sync_status())
    
    if df.empty:
        st.error("Error: Dataset empty after merging signals.")
    else:
        target_etfs = ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret']
        target_etfs = [t for t in target_etfs if t in df.columns]
        # Include Z-scored features in input
        input_features = [c for c in df.columns if c not in target_etfs and c != 'SOFR']
        
        scaler = MinMaxScaler()
        scaled_input = scaler.fit_transform(df[input_features])
        
        lookbacks = [30, 45, 60] if len(df) > 200 else [10, 20]
        holdings = [1, 3, 5]
        best_conviction_score = -np.inf
        final_res = None

        with st.spinner("Training Transformer with Z-Score Signals..."):
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

                # Model Architecture
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
                
                # 2. CONVICTION THRESHOLD (Idea #2)
                for hold in holdings:
                    daily_strat_rets = []
                    sofr_series = df['SOFR'] if 'SOFR' in df.columns else pd.Series(0.04, index=df.index)
                    
                    for i in range(len(preds)):
                        idx = np.argmax(preds[i])
                        # Sort to find the second best prediction
                        sorted_p = np.sort(preds[i])
                        next_best = sorted_p[-2] if len(sorted_p) > 1 else 0
                        
                        daily_rf = sofr_series.iloc[split_idx + i] / 252
                        hurdle = daily_rf + fee_pct
                        
                        # Logic: Must beat second best by 20% (1.2x) AND clear the fee hurdle
                        if preds[i][idx] > (next_best * 1.2) and preds[i][idx] > hurdle:
                            daily_strat_rets.append(test_y[i][idx] - (fee_pct if i % hold == 0 else 0))
                        else:
                            daily_strat_rets.append(daily_rf) # Stay in Cash
                    
                    score = np.mean(daily_strat_rets)
                    if score > best_conviction_score:
                        best_conviction_score = score
                        final_res = {
                            "lb": lb, "hold": hold, "preds": preds, 
                            "test_y": test_y, "dates": df.index[split_idx:],
                            "model": model, "strat_rets": daily_strat_rets, 
                            "sofr": sofr_series.iloc[split_idx:].values,
                            "last_x": X[-1:], "input_features": input_features,
                            "target_names": target_etfs
                        }

        # ------------------------------
        # 5. UI: REFINED OUTPUT
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
            
            # Predict Next Move
            p_now = res["model"].predict(res["last_x"])[0]
            best_idx = np.argmax(p_now)
            sorted_now = np.sort(p_now)
            curr_hurdle = (res["sofr"][-1]/252) + fee_pct
            
            if p_now[best_idx] > (sorted_now[-2] * 1.2) and p_now[best_idx] > curr_hurdle:
                final_asset = res["target_names"][best_idx].split('_')[0]
            else:
                final_asset = "CASH (SOFR)"
            
            k1.metric("Predicted Asset", final_asset, f"Next Open: {get_next_open_date()}")
            k2.metric("Annualized Return", f"{ann_ret*100:.2f}%", f"{oos_yrs:.2f} OOS Years")
            k3.metric("Sharpe Ratio", f"{sharpe:.2f}", "SOFR Adjusted")
            k4.metric("Hit Ratio (15d)", f"{(np.sum(strat_rets[-15:] > 0)/15)*100:.1f}%", f"Lookback: {res['lb']}d")
            k5.metric("Max Daily Stress", f"{np.min(strat_rets)*100:.2f}%", "Worst Day")

            st.markdown("### OOS Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res["dates"][:len(cum_strat)], y=cum_strat, fill='tozeroy', line=dict(color='#00d1b2', width=1.5)))
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,b=0,t=20))
            st.plotly_chart(fig, use_container_width=True)

            # Signal Importance (Progress Bars)
            st.markdown("### Signal Contribution Analysis")
            # Simple Importance Estimate (Model Weights Mean)
            weights = np.abs(res["model"].layers[-2].get_weights()[0]).mean(axis=1)
            # Match weights to input features
            weights = weights[:len(res["input_features"])]
            norm_w = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
            
            feat_imp = pd.DataFrame({'Signal': res["input_features"], 'Weight': norm_w}).sort_values('Weight', ascending=False)
            st.dataframe(
                feat_imp.head(15),
                column_config={"Weight": st.column_config.ProgressColumn("Relative Power", min_value=0, max_value=1, format="%.2f")},
                hide_index=True, use_container_width=True
            )

            # 15-Day Audit (NO SOFR COLUMN)
            st.markdown("### 15-Day Performance Audit Trail")
            audit = []
            display_days = min(15, len(res["preds"]))
            for i in range(1, display_days + 1):
                p = res["preds"][-i]
                idx = np.argmax(p)
                sorted_p = np.sort(p)
                rf_daily = res["sofr"][-i]/252
                # Apply Conviction Check
                is_cash = not (p[idx] > (sorted_p[-2] * 1.2) and p[idx] > (rf_daily + fee_pct))
                
                audit.append({
                    "Date": res["dates"][-i].strftime('%Y-%m-%d'),
                    "Predicted": "CASH" if is_cash else res["target_names"][idx].split('_')[0],
                    "Realized Daily Return": res["strat_rets"][-i]
                })
            
            st.table(pd.DataFrame(audit).style.format({"Realized Daily Return": "{:.4%}"}).applymap(
                lambda x: 'color: #00ff00' if x > 0 else 'color: #ff4b4b', subset=['Realized Daily Return']
            ))
        else:
            st.error("No valid strategy found.")
