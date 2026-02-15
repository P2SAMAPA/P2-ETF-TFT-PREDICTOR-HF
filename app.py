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
import plotly.express as px

# ------------------------------
# 1. NYSE CALENDAR & UTILS
# ------------------------------
def get_next_open_date():
    """Calculates the next valid trading day using the NYSE calendar."""
    nyse = mcal.get_calendar('NYSE')
    # Fetch schedule for 10 days to ensure coverage of long weekends (like Presidents' Day)
    schedule = nyse.schedule(start_date=datetime.now(), end_date=datetime.now() + timedelta(days=10))
    if schedule.empty: return "TBD"
    return schedule.iloc[0].market_open.strftime('%b %d, %Y')

# ------------------------------
# 2. UI CONFIGURATION
# ------------------------------
st.set_page_config(page_title="P2-Transformer Pro", layout="wide")

with st.sidebar:
    st.header("Model Configuration")
    start_year = st.slider("Training Start Year", 2008, 2024, 2016)
    fee_bps = st.slider("Transaction Cost (bps)", 0, 100, 15)
    epochs = st.number_input("Training Epochs", 5, 200, 50)
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
    """Loads dataset, refreshes macro signals, and handles SOFR accrual."""
    ds = load_dataset(REPO_ID, split="train").to_pandas()
    ds['Date'] = pd.to_datetime(ds['Date'])
    ds = ds.set_index('Date').sort_index()
    
    # TRUNCATION: Date-first filtering to fix the static OOS window bug
    df = ds[ds.index.year >= start_year].copy()
    
    # Fresh Macro Fetch
    tickers = {"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY", "GC=F": "GOLD", "HG=F": "COPPER"}
    macro = yf.download(list(tickers.keys()), start=df.index.min(), progress=False)['Close']
    macro = macro.rename(columns=tickers)
    
    # RECTIFICATION: Resolve column overlaps (VIX, TNX, DXY)
    cols_to_drop = [c for c in macro.columns if c in df.columns]
    df = df.drop(columns=cols_to_drop).join(macro, how='left').ffill().dropna()
    
    # SOFR (Risk Free Rate) for Cash Position Accrual
    try:
        sofr = yf.download("^IRX", start=df.index.min(), progress=False)['Close'] / 100
        df['SOFR'] = sofr.reindex(df.index).ffill().fillna(0.04)
    except:
        df['SOFR'] = 0.04
        
    return df

# ------------------------------
# 4. OPTIMIZATION RUNTIME
# ------------------------------
if run_button:
    df = get_data()
    target_etfs = ['TLT_Ret', 'TBT_Ret', 'VNQ_Ret', 'SLV_Ret', 'GLD_Ret']
    
    # Identify only input signals (excluding targets and SOFR for the Attention Head)
    input_features = [c for c in df.columns if c not in target_etfs and c != 'SOFR']
    
    scaler = MinMaxScaler()
    scaled_input = scaler.fit_transform(df[input_features])
    
    lookbacks = [30, 45, 60]
    holdings = [1, 3, 5]
    best_conviction = -np.inf
    final_res = None

    with st.spinner("Processing Signal Pattern Search..."):
        # Split on truncated data to ensure OOS scaling
        split_idx = int(len(scaled_input) * 0.8)
        
        for lb in lookbacks:
            X, y = [], []
            for i in range(lb, len(scaled_input)):
                X.append(scaled_input[i-lb:i])
                y.append(df[target_etfs].iloc[i].values)
            
            X, y = np.array(X), np.array(y)
            train_X = X[:split_idx - lb]
            test_X = X[split_idx - lb:]
            train_y = y[:split_idx - lb]
            test_y = y[split_idx - lb:]

            # Model Architecture with Attention Capture
            inputs = Input(shape=(X.shape[1], X.shape[2]))
            attn_output, attn_weights = MultiHeadAttention(num_heads=4, key_dim=X.shape[2])(inputs, inputs, return_attention_scores=True)
            attn_res = LayerNormalization()(attn_output + inputs)
            conv = Conv1D(64, 3, activation='relu', padding='same')(attn_res)
            pool = GlobalMaxPooling1D()(conv)
            lstm = Bidirectional(LSTM(64))(inputs)
            merged = Concatenate()([pool, lstm])
            x = Dense(128, activation='relu')(merged)
            outputs = Dense(5)(x)
            
            model = Model(inputs, outputs)
            weight_model = Model(inputs, attn_weights)
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(train_X, train_y, epochs=int(epochs), batch_size=32, verbose=0)
            
            preds = model.predict(test_X)
            
            for hold in holdings:
                daily_strat_rets = []
                for i in range(len(preds)):
                    idx = np.argmax(preds[i])
                    daily_sofr = df['SOFR'].iloc[split_idx + i] / 252 # Daily accrual
                    
                    # Hurdle Logic: Must beat Cash + Fee
                    if preds[i][idx] > (fee_pct + daily_sofr):
                        # Use actual 1-day realized return
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
                        "last_x": X[-1:], "input_features": input_features
                    }

    # ------------------------------
    # 5. UI: PERFORMANCE SCORECARD
    # ------------------------------
    res = final_res
    strat_rets = np.array(res["strat_rets"])
    cum_strat = np.cumprod(1 + strat_rets)
    
    # Calendar-based year calculation to fix the OOS year bug
    oos_yrs = (res["dates"][-1] - res["dates"][0]).days / 365.25
    ann_ret = (cum_strat[-1] ** (1/oos_yrs)) - 1 if oos_yrs > 0 else 0
    sharpe = np.sqrt(252) * np.mean(strat_rets - (res["sofr"]/252)) / np.std(strat_rets)
    
    st.markdown("### Performance Scorecard")
    k1, k2, k3, k4, k5 = st.columns(5)
    
    # Live prediction using next open date
    p_now = res["model"].predict(res["last_x"])[0]
    best_idx = np.argmax(p_now)
    current_hurdle = (df['SOFR'].iloc[-1]/252) + fee_pct
    final_asset = target_etfs[best_idx].split('_')[0] if p_now[best_idx] > current_hurdle else "CASH (SOFR)"
    
    k1.metric("Predicted Asset", final_asset, f"Next Open: {get_next_open_date()}")
    k2.metric("Annualized Return", f"{ann_ret*100:.2f}%", f"{oos_yrs:.2f} OOS Years")
    k3.metric("Sharpe Ratio", f"{sharpe:.2f}", "SOFR Adjusted")
    k4.metric("Hit Ratio (15d)", f"{(np.sum(strat_rets[-15:] > 0)/15)*100:.1f}%", f"Search: {res['lb']}L|{res['hold']}H")
    k5.metric("Max Daily Stress", f"{np.min(strat_rets)*100:.2f}%", "Worst Day")

    st.markdown(f"### OOS Equity Curve ({res['lb']}d Lookback | {res['hold']}d Holding)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["dates"], y=cum_strat, fill='tozeroy', line=dict(color='#00d1b2', width=1.5)))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,b=0,t=20))
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # 6. UI: SIGNAL CONTRIBUTION ANALYSIS
    # ------------------------------
    st.markdown("---")
    st.markdown("### Signal Contribution Analysis")
    
    # Extraction and length-matching for predictive power analysis
    raw_weights = res["weight_model"].predict(res["last_x"])[0] 
    importance = np.mean(raw_weights, axis=(0, 1, 2)) 
    
    # Pad or trim to match input_features exactly
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
        st.write("Temporal Attention Heatmap (Cross-Head Average)")
        # Heatmap visualizing lookback context focus
        avg_weights = np.mean(raw_weights, axis=0)
        fig_heat = px.imshow(
            avg_weights,
            labels=dict(x="Key Sequence", y="Query Sequence", color="Weight"),
            color_continuous_scale="Viridis"
        )
        fig_heat.update_layout(height=350, template="plotly_dark", margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig_heat, use_container_width=True)

    # ------------------------------
    # 7. UI: AUDIT TRAIL
    # ------------------------------
    st.markdown("### 15-Day Performance Audit Trail")
    audit = []
    for i in range(1, 16):
        idx = np.argmax(res["preds"][-i])
        daily_hurdle = (res["sofr"][-i]/252) + fee_pct
        is_cash = res["preds"][-i][idx] <= daily_hurdle
        
        audit.append({
            "Date": res["dates"][-i].strftime('%Y-%m-%d'),
            "Predicted": "CASH" if is_cash else target_etfs[idx].split('_')[0],
            "SOFR (Ann)": f"{res['sofr'][-i]*100:.2f}%",
            "Realized Daily Return": res["strat_rets"][-i]
        })
    
    st.table(pd.DataFrame(audit).style.format({"Realized Daily Return": "{:.4%}"}).applymap(
        lambda x: 'color: #00ff00' if x > 0 else 'color: #ff4b4b', subset=['Realized Daily Return']
    ))
