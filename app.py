import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from polygon import RESTClient
import yfinance as yf
import os
import time
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# ==========================================
# 1. INITIALIZATION & MODELS
# ==========================================
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Safety check for keys
if not POLYGON_API_KEY or not FRED_API_KEY:
    st.error("🔑 API Keys Missing in Space Secrets.")
    st.stop()

polygon_client = RESTClient(POLYGON_API_KEY)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 2. DATA ENGINE (The Hybrid Function)
# ==========================================
@st.cache_data(ttl=3600)
def get_hybrid_data(ticker_list, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    combined_data = pd.DataFrame()
    
    for ticker in ticker_list:
        success = False
        try:
            aggs = polygon_client.get_aggs(ticker, 1, "day", start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            df = pd.DataFrame(aggs)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            combined_data[ticker] = df['close']
            success = True
        except:
            pass # Failover to yfinance
        
        if not success:
            try:
                yf_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not yf_data.empty:
                    combined_data[ticker] = yf_data['Close']
            except:
                continue
    return combined_data

# ==========================================
# 3. UI LAYOUT
# ==========================================
st.set_page_config(page_title="Fixed Income Alpha", layout="wide")
st.title("🏦 Fixed Income Alpha Optimizer")

# Sidebar
st.sidebar.header("Trading Constraints")
tx_cost_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 100, 15)
lookback_days = 1825 
fixed_income_etfs = ["VCLT", "TLT", "TBT", "MBB", "VNQ", "HYG", "SLV", "GLD", "PFF"]

# Execution
data = get_hybrid_data(fixed_income_etfs, lookback_days)

if not data.empty:
    with st.status("🤖 Analyzing Markets & Training LSTM...", expanded=False) as status:
        # Preprocessing
        df_clean = data.dropna()
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_clean)
        
        # Windowing
        window = 30
        X, y = [], []
        for i in range(len(scaled_data) - window):
            X.append(scaled_data[i:i+window])
            y.append(scaled_data[i+window])
        
        # Auto-Training
        model = LSTMModel(len(fixed_income_etfs), 64, 2, len(fixed_income_etfs))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(30):
            model.train()
            optimizer.zero_grad()
            out = model(torch.FloatTensor(np.array(X)))
            loss = criterion(out, torch.FloatTensor(np.array(y)))
            loss.backward()
            optimizer.step()
        status.update(label="Analysis Complete", state="complete")

    # Prediction Logic
    model.eval()
    last_window = torch.FloatTensor(scaled_data[-window:].reshape(1, window, -1))
    pred_raw = model(last_window).detach().numpy()
    pred_prices = scaler.inverse_transform(pred_raw)[0]
    current_prices = df_clean.iloc[-1].values
    
    # Calculate returns for 1d, 3d, 5d holding periods
    cost_pct = tx_cost_bps / 10000
    optimization_results = []
    
    for i, ticker in enumerate(fixed_income_etfs):
        base_expected_return = (pred_prices[i] - current_prices[i]) / current_prices[i]
        
        # Simulated holding period drifts (Example: 3-day return is expected to be ~1.4x the 1-day move)
        ret_1d = base_expected_return - cost_pct
        ret_3d = (base_expected_return * 1.4) - cost_pct
        ret_5d = (base_expected_return * 1.8) - cost_pct
        
        best_period_val = max(ret_1d, ret_3d, ret_5d)
        best_period_name = "1 Day" if best_period_val == ret_1d else "3 Days" if best_period_val == ret_3d else "5 Days"
        
        optimization_results.append({
            "ETF": ticker,
            "Net Return": best_period_val,
            "Best Holding Period": best_period_name
        })

    # Sort and Display
    res_df = pd.DataFrame(optimization_results).sort_values(by="Net Return", ascending=False)
    top_pick = res_df.iloc[0]

    # Display Top Signal
    st.subheader("🎯 Optimal Trading Signal")
    c1, c2, c3 = st.columns(3)
    c1.metric("Top Pick", top_pick['ETF'])
    c2.metric("Expected Net Return", f"{top_pick['Net Return']:.2%}")
    c3.metric("Holding Period", top_pick['Best Holding Period'])

    st.divider()
    
    # Ranking Table
    st.subheader("Market Opportunity Ranking")
    st.dataframe(res_df.style.format({"Net Return": "{:.2%}"}), use_container_width=True)

    # Chart
    fig = go.Figure()
    for t in fixed_income_etfs:
        fig.add_trace(go.Scatter(x=df_clean.index[-60:], y=df_clean[t].tail(60), name=t))
    fig.update_layout(title="Price Momentum (Last 60 Days)", template="plotly_white")
    st.plotly_chart(fig, width='stretch')
else:
    st.error("Could not load data. Check your API Keys.")
