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
from datasets import load_dataset, Dataset
from huggingface_hub import login
import plotly.graph_objects as go

# ------------------------------
# 1. NYSE CALENDAR & UTILS
# ------------------------------
def get_next_open_date():
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=datetime.now(), end_date=datetime.now() + timedelta(days=10))
    if schedule.empty:
        return "TBD"
    next_open = schedule.iloc[0].market_open
    return next_open.strftime('%b %d, %Y')

# ------------------------------
# 2. UI CONFIGURATION
# ------------------------------
st.set_page_config(page_title="P2-Transformer Pro", layout="wide")

with st.sidebar:
    st.header("Model Configuration")
    start_year = st.slider("Training Start Year", 2008, 2024, 2016)
    fee_bps = st.slider("Transaction Cost (bps)", 0, 100, 15)
    epochs = st.number_input("Training Epochs", 5, 100, 20)
    st.markdown("---")
    update_hf = st.checkbox("Sync Updates to HF Hub?", value=True)
    run_button = st.button("Execute Transformer Alpha", type="primary")

st.title("P2-TRANSFORMER-ETF-PREDICTOR")

# ------------------------------
# 3. DATA ENGINE
# ------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "P2SAMAPA/my-etf-data"
fee_pct = fee_bps / 10000

def get_data():
    ds = load_dataset(REPO_ID, split="train").to_pandas()
    ds['Date'] = pd.to_datetime(ds['Date'])
    ds = ds.set_index('Date').sort_index()
    
    # TRUNCATION FIX: Filter by year FIRST before joining/splitting
    ds = ds[ds.index.year >= start_year]
    
    tickers = {"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY", "GC=F": "GOLD", "HG=F": "COPPER"}
    macro = yf.download(list(tickers.keys()), start=ds.index.min(), progress=False)['Close']
    macro = macro.rename(columns=tickers)
    
    # Resolve Overlap
    cols_to_drop = [c for c in macro.columns if c in ds.columns]
    ds_cleaned = ds.drop(columns=cols_to_drop)
    
    full_df = ds_cleaned.join(macro, how='left').ffill().dropna()
    
    # Fetch SOFR (Risk Free) - fallback to 3.65% if live fetch fails
    try:
        sofr_data = yf.download("^IRX", start=ds.index.min(), progress=False)['Close'] / 100
        full_df['SOFR'] = sofr_data.reindex(full_df.index).ffill().fillna(0.0365)
    except:
        full_df['SOFR'] = 0.0365
        
    return full_df

# ------------------------------
# 4. OPTIMIZATION RUNTIME
# ------------------------------
if run_button:
    df = get_data()
    target_etfs = ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret']
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    
    lookbacks = [30, 45, 60]
    holdings = [1, 3, 5]
    
    best_overall_conviction = -np.inf
    final_results = None

    with st.spinner("Analyzing Temporal Scales..."):
        # 80/20 split on the truncated data
        split_idx = int(len(scaled) * 0.8)
        
        for lb in lookbacks:
            X, y = [], []
            for i in range(lb, len(scaled) - 5): 
                X.append(scaled[i-lb:i])
                y.append(df[target_etfs].iloc[i].values)
            
            X, y = np.array(X), np.array(y)
            X_train, X_test = X[:split_idx-lb], X[split_idx-lb:]
            y_train, y_test = y[:split_idx-lb], y[split_idx-lb:]

            inputs = Input(shape=(X.shape[1], X.shape[2]))
            attn = MultiHeadAttention(num_heads=4, key_dim=X.shape[2])(inputs, inputs)
            attn = LayerNormalization()(attn + inputs)
            conv = Conv1D(64, 3, activation='relu', padding='same')(attn)
            pool = GlobalMaxPooling1D()(conv)
            lstm = Bidirectional(LSTM(64))(inputs)
            merged = Concatenate()([pool, lstm])
            x = Dense(128, activation='relu')(merged)
            outputs = Dense(5)(x)
            
            model = Model(inputs, outputs)
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=int(epochs), batch_size=32, verbose=0)
            
            preds = model.predict(X_test)
            
            for hold in holdings:
                hold_rets = []
                for i in range(len(preds) - hold):
                    idx = np.argmax(preds[i])
                    # Logic: If max pred return < fee + SOFR hurdle, choose CASH
                    daily_sofr = df['SOFR'].iloc[split_idx + i] / 360
                    
                    if preds[i][idx] > (fee_pct + daily_sofr):
                        real_cum_ret = np.sum(y_test[i:i+hold, idx]) - fee_pct
                    else:
                        # Earn SOFR interest, zero fee
                        real_cum_ret = np.sum(df['SOFR'].iloc[split_idx+i : split_idx+i+hold].values / 360)
                    
                    hold_rets.append(real_cum_ret)
                
                avg_conviction = np.mean(hold_rets)
                if avg_conviction > best_overall_conviction:
                    best_overall_conviction = avg_conviction
                    final_results = {
                        "lb": lb, "hold": hold, "preds": preds, 
                        "y_test": y_test, "dates": df.index[split_idx:],
                        "model": model, "strat_rets": hold_rets,
                        "sofr": df['SOFR'].iloc[split_idx:].values
                    }

    # ------------------------------
    # 5. OUTPUT UI
    # ------------------------------
    res = final_results
    strat_rets = np.array(res["strat_rets"])
    cum_strat = np.cumprod(1 + strat_rets)
    
    # Correct Year Calculation
    oos_days = (res["dates"][-1] - res["dates"][0]).days
    oos_years = oos_days / 365.25
    ann_ret = (cum_strat[-1] ** (1/oos_years)) - 1 if oos_years > 0 else 0
    
    # Sharpe using dynamic SOFR
    excess_rets = strat_rets - (res["sofr"][:len(strat_rets)] / 360)
    sharpe = np.sqrt(252) * np.mean(excess_rets) / np.std(strat_rets) if np.std(strat_rets) != 0 else 0
    
    st.markdown("### Performance Scorecard")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        latest_pred = res["model"].predict(scaled[-res["lb"]:].reshape(1, res["lb"], -1))[0]
        top_idx = np.argmax(latest_pred)
        daily_sofr_now = df['SOFR'].iloc[-1] / 360
        top_asset = target_etfs[top_idx].split('_')[0] if latest_pred[top_idx] > (fee_pct + daily_sofr_now) else "CASH (SOFR)"
        st.metric("Predicted Asset", top_asset, f"Next Open: {get_next_open_date()}")
    
    kpi2.metric("Annualized Return", f"{ann_ret*100:.2f}%", f"{oos_years:.2f} OOS Years")
    kpi3.metric("Sharpe Ratio", f"{sharpe:.2f}", "Dynamic SOFR Adj")
    kpi4.metric("Hit Ratio (15d)", f"{(np.sum(strat_rets[-15:] > 0)/15)*100:.1f}%", f"Hold: {res['hold']}D")
    kpi5.metric("Max Daily Stress", f"{np.min(strat_rets)*100:.2f}%", "Worst Day")

    st.markdown(f"### OOS Equity Curve ({res['lb']}d Lookback | {res['hold']}d Holding)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["dates"][:len(cum_strat)], y=cum_strat, fill='tozeroy', line=dict(color='#00d1b2')))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,b=0,t=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 15-Day Performance Audit Trail")
    audit_list = []
    for i in range(1, 16):
        p_idx = np.argmax(res["preds"][-i])
        daily_sofr_audit = res["sofr"][-i] / 360
        is_cash = res["preds"][-i][p_idx] <= (fee_pct + daily_sofr_audit)
        
        audit_list.append({
            "Date": res["dates"][-i].strftime('%Y-%m-%d'),
            "Predicted": "CASH" if is_cash else target_etfs[p_idx].split('_')[0],
            "SOFR (Ann)": f"{res['sofr'][-i]*100:.2f}%",
            "Realized Return": res["strat_rets"][-i]
        })
    st.table(pd.DataFrame(audit_list).style.format({"Realized Return": "{:.4%}"}).applymap(
        lambda x: 'color: green' if x > 0 else 'color: red', subset=['Realized Return']
    ))
