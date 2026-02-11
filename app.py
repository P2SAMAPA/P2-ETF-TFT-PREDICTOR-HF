# app.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
from polygon import RESTClient

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────

ETFS = ['TLT', 'VCLT', 'TBT', 'VNQ', 'SLV', 'GLD', 'PFF', 'MBB']

SIGNAL_TICKERS = [
    'I:VIX',          # VIX
    'I:MOVE',         # MOVE index
    'I:SKEW',         # SKEW index
    'I:PCALL',        # Put/Call ratio
    'I:US10Y',        # 10Y Treasury
    'I:US2Y',         # 2Y Treasury
    'BAMLH0A0HYM2',   # ICE BofA US High Yield Index Option-Adjusted Spread
    'C:HG'            # Copper futures (for Gold/Copper ratio)
]

ALL_TICKERS = ETFS + SIGNAL_TICKERS

# ────────────────────────────────────────────────
# Data Fetching
# ────────────────────────────────────────────────

@st.cache_data(ttl=3600)  # cache 1 hour
def fetch_data(start_date: str, end_date: str) -> pd.DataFrame:
    client = RESTClient()  # API key should be in secrets.toml or env
    data = {}
    for ticker in ALL_TICKERS:
        try:
            aggs = client.get_aggs(ticker, 1, "day", start_date, end_date, limit=50000)
            df = pd.DataFrame(aggs)
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
            df = df.set_index('date')[['close']].rename(columns={'close': ticker})
            data[ticker] = df
        except Exception as e:
            st.warning(f"Failed to fetch {ticker}: {e}")
    combined = pd.concat(data.values(), axis=1).sort_index()
    combined.index = pd.to_datetime(combined.index)
    return combined.dropna(how='all')

# ────────────────────────────────────────────────
# Feature Engineering
# ────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['10Y_2Y_spread'] = df['I:US10Y'] - df['I:US2Y']
    df['Credit_Spread'] = df['BAMLH0A0HYM2']
    df['Gold_Copper_ratio'] = df['GLD'] / df['C:HG']

    # Lags and moving averages
    for col in df.columns:
        for lag in [1, 3, 5, 10]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        df[f'{col}_ma5']  = df[col].rolling(5).mean()
        df[f'{col}_ma20'] = df[col].rolling(20).mean()

    return df.dropna()

# ────────────────────────────────────────────────
# LSTM Dataset & Model
# ────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[idx+self.seq_len],   dtype=torch.float32)
        )

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# ────────────────────────────────────────────────
# Training helper
# ────────────────────────────────────────────────

@st.cache_resource
def train_lstm(X_train, y_train, seq_len=30, epochs=40, batch_size=64):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X_train)
    y_scaled = scaler_y.fit_transform(y_train)

    dataset = TimeSeriesDataset(X_scaled, y_scaled, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMForecaster(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for Xb, yb in loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    return model, scaler_X, scaler_y

# ────────────────────────────────────────────────
# Simple greedy backtest (long only, pick best predicted ETF)
# ────────────────────────────────────────────────

def run_backtest(df_oos, hold_days, tc_bps, retrain_every=5, seq_len=30):
    features = [c for c in df_oos.columns if c not in ETFS]
    returns = []

    current_train_end = df_oos.index[0] - pd.Timedelta(days=1)
    equity = 1.0
    equity_curve = []
    positions = []

    for i in range(0, len(df_oos) - hold_days, hold_days):
        window_end = df_oos.index[i]

        # Retrain periodically
        if (i == 0) or ((window_end - current_train_end).days >= retrain_every):
            train_slice = df_full[df_full.index < window_end]
            if len(train_slice) < 200:
                continue
            X_train = train_slice[features].values
            y_train = train_slice[ETFS].pct_change(hold_days).dropna().values
            if len(y_train) == 0:
                continue
            model, scX, scy = train_lstm(X_train, y_train, seq_len=seq_len)
            current_train_end = window_end

        # Get latest sequence for prediction
        start_idx = max(0, i - seq_len)
        seq = df_oos[features].iloc[start_idx:i].values
        if len(seq) < seq_len:
            continue
        seq_scaled = scX.transform(seq)
        seq_t = torch.tensor(seq_scaled[None], dtype=torch.float32)  # [1, seq, feat]

        model.eval()
        with torch.no_grad():
            pred_scaled = model(seq_t).numpy()[0]
        pred_returns = scy.inverse_transform([pred_scaled])[0]

        # Choose ETF with highest predicted return
        best_idx = np.argmax(pred_returns)
        best_etf = ETFS[best_idx]
        expected_ret = pred_returns[best_idx]

        # Realized return over hold period
        if i + hold_days >= len(df_oos):
            break
        start_price = df_oos[best_etf].iloc[i]
        end_price   = df_oos[best_etf].iloc[i + hold_days]
        gross_ret = (end_price / start_price) - 1

        # Transaction cost (applied on entry)
        tc = tc_bps / 10000.0
        net_ret = gross_ret - tc

        equity *= (1 + net_ret)
        equity_curve.append(equity)
        returns.append(net_ret)
        positions.append(best_etf)

    if not equity_curve:
        return 0.0, [], []

    ann_ret = (equity ** (252 / (len(df_oos) / hold_days))) - 1 if equity > 0 else -1.0
    return ann_ret, equity_curve, positions

# ────────────────────────────────────────────────
# Streamlit App
# ────────────────────────────────────────────────

st.title("ETF Momentum Forecasting Engine")
st.caption("Long-only strategy — maximizes gross return (ignores risk)")

# ── Controls ─────────────────────────────────────

col1, col2, col3 = st.columns([2,2,2])

with col1:
    tc_bps = st.slider("Transaction costs (bps per trade)", 0, 100, 10, step=5)

with col2:
    hold_period = st.radio("Holding period", [1, 3, 5], index=2, horizontal=True)

with col3:
    retrain_freq = st.radio("Retrain model every", [3, 5, 10], index=1, horizontal=True)

# ── Data ─────────────────────────────────────────

today = datetime(2026, 2, 11)           # as given
start_hist = "2010-01-01"
oos_start  = "2021-01-01"

with st.spinner("Loading market data from Polygon..."):
    df_full = fetch_data(start_hist, today.strftime("%Y-%m-%d"))
    df_eng  = engineer_features(df_full)

df_train = df_eng[df_eng.index < oos_start]
df_oos   = df_eng[df_eng.index >= oos_start]

st.caption(f"Training until {oos_start}  •  OOS period: {len(df_oos)} days")

# ── Run backtest ─────────────────────────────────

if st.button("Run Backtest", type="primary"):
    with st.spinner(f"Running backtest — hold {hold_period}d, retrain every {retrain_freq}d..."):
        ann_ret, equity_curve, _ = run_backtest(
            df_oos, hold_days=hold_period, tc_bps=tc_bps,
            retrain_every=retrain_freq, seq_len=30
        )

    st.subheader(f"Annualized Return (net of costs): **{ann_ret*100:.2f}%**")

    if equity_curve:
        eq_series = pd.Series(equity_curve, index=df_oos.index[:len(equity_curve)])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_series.index,
            y=eq_series,
            mode='lines',
            name='Equity Curve',
            line=dict(color='royalblue')
        ))
        fig.update_layout(
            title=f"Equity Curve — {hold_period}-day hold, {tc_bps} bps costs",
            yaxis_title="Cumulative Return (1 = start)",
            xaxis_title="Date"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.success("Backtest completed.")
