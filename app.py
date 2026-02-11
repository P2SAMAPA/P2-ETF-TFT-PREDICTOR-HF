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

# Updated: Use only reliably supported Polygon tickers (indices + futures + ETFs as proxies)
SIGNAL_TICKERS = [
    '^VIX',         # CBOE VIX index
    'MOVE',         # MOVE index (if available; fallback empty if not)
    'IEF',          # iShares 7-10 Year Treasury → proxy for ~10Y yield moves
    'SHY',          # iShares 1-3 Year Treasury   → proxy for ~2Y/short end
    'HYG',          # iShares High Yield Corp Bond → for credit spread proxy
    'HG=F'          # Copper futures (continuous)
]

ALL_TICKERS = ETFS + SIGNAL_TICKERS

# ────────────────────────────────────────────────
# Data Fetching
# ────────────────────────────────────────────────

@st.cache_data(ttl=3600)  # cache 1 hour
def fetch_data(start_date: str, end_date: str) -> pd.DataFrame:
    client = RESTClient()  # API key from secrets.toml or env
    data = {}
    for ticker in ALL_TICKERS:
        try:
            aggs = client.get_aggs(ticker, 1, "day", start_date, end_date, limit=50000)
            df = pd.DataFrame(aggs)
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
            df = df.set_index('date')[['close']].rename(columns={'close': ticker})
            data[ticker] = df.astype(float)  # Ensure float dtype
            st.info(f"Fetched {ticker} successfully ({len(df)} rows)")
        except Exception as e:
            st.warning(f"Failed to fetch {ticker}: {str(e)}. Using empty series.")
            data[ticker] = pd.Series(dtype=float, name=ticker)  # empty with float dtype

    if not data:
        st.error("No data fetched at all. Check Polygon API key and connectivity.")
        return pd.DataFrame()

    combined = pd.concat(data.values(), axis=1).sort_index()
    combined.index = pd.to_datetime(combined.index)
    return combined.astype(float)  # Force all to float

# ────────────────────────────────────────────────
# Feature Engineering – using proxies
# ────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().astype(float)  # Ensure numeric dtypes early

    # Proxy for 10Y-2Y spread: longer-duration ETF minus shorter-duration
    if 'IEF' in df.columns and 'SHY' in df.columns:
        df['10Y_2Y_proxy'] = df['IEF'] - df['SHY']   # level difference (price proxy for yield curve)
        # Alternative: returns diff → df['10Y_2Y_proxy_ret'] = df['IEF'].pct_change() - df['SHY'].pct_change()

    # Credit spread proxy: high-yield ETF minus Treasury ETF
    if 'HYG' in df.columns and 'IEF' in df.columns:
        df['Credit_Spread_proxy'] = df['HYG'] - df['IEF']

    # Gold / Copper ratio
    if 'GLD' in df.columns and 'HG=F' in df.columns:
        df['Gold_Copper_ratio'] = df['GLD'] / df['HG=F']

    # Lags and moving averages on all available columns
    for col in df.columns:
        if col in ETFS + ['^VIX', 'MOVE']:  # focus on key signals + ETFs
            for lag in [1, 3, 5, 10]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
            df[f'{col}_ma5']  = df[col].rolling(5).mean()
            df[f'{col}_ma20'] = df[col].rolling(20).mean()

    # Fill gaps (macro data can be sparse) → forward then backward
    df = df.ffill().bfill().infer_objects(copy=False)  # Avoid FutureWarning by inferring dtypes

    return df.dropna(how='all')  # keep rows with at least some data

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
            torch.tensor(self.X[idx:idx + self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[idx + self.seq_len], dtype=torch.float32)
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
        total_loss = 0
        for Xb, yb in loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # st.write(f"Epoch {epoch+1}/{epochs} loss: {total_loss/len(loader):.6f}")

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
        if (i == 0) or ((window_end - current_train_end).days >= retrain_every * hold_days):
            train_slice = df_full[df_full.index < window_end]
            if len(train_slice) < 200:
                continue
            X_train = train_slice[features].values
            y_train = train_slice[ETFS].pct_change(hold_days).iloc[seq_len:].values  # align
            if len(y_train) < 10:
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

        # Choose ETF with highest predicted return (if positive)
        best_idx = np.argmax(pred_returns)
        best_etf = ETFS[best_idx]
        expected_ret = pred_returns[best_idx]

        # Realized return over hold period
        if i + hold_days >= len(df_oos):
            break
        start_price = df_oos[best_etf].iloc[i]
        end_price   = df_oos[best_etf].iloc[i + hold_days]
        gross_ret = (end_price / start_price) - 1 if start_price > 0 else 0

        # Transaction cost (applied on entry – simplistic)
        tc = tc_bps / 10000.0
        net_ret = gross_ret - tc if expected_ret > 0 else 0  # only trade if predicted positive

        equity *= (1 + net_ret)
        equity_curve.append(equity)
        returns.append(net_ret)
        positions.append(best_etf if net_ret != 0 else "Cash")

    if not equity_curve:
        return 0.0, [], []

    days_in_period = (df_oos.index[-1] - df_oos.index[0]).days
    ann_ret = (equity ** (365 / days_in_period)) - 1 if equity > 0 and days_in_period > 0 else -1.0
    return ann_ret, equity_curve, positions

# ────────────────────────────────────────────────
# Streamlit App
# ────────────────────────────────────────────────

st.title("ETF Macro Momentum Engine")
st.caption("Long-only – return maximization using macro proxies & LSTM forecasts")

# ── Controls ─────────────────────────────────────

col1, col2, col3 = st.columns(3)

with col1:
    tc_bps = st.slider("Transaction costs (bps per trade)", 0, 100, 10, step=5)

with col2:
    hold_period = st.radio("Holding period (days)", [1, 3, 5], index=2, horizontal=True)

with col3:
    retrain_freq = st.radio("Retrain model every (periods)", [3, 5, 10], index=1, horizontal=True)

# ── Data ─────────────────────────────────────────

today = datetime(2026, 2, 11)
start_hist = "2010-01-01"
oos_start  = "2021-01-01"

with st.spinner("Loading market & macro proxy data from Polygon..."):
    global df_full
    df_full = fetch_data(start_hist, today.strftime("%Y-%m-%d"))
    df_eng  = engineer_features(df_full)

df_train = df_eng[df_eng.index < oos_start]
df_oos   = df_eng[df_eng.index >= oos_start]

st.caption(f"Training data until {oos_start}  •  OOS period: ~{len(df_oos)} days")

# ── Run backtest ─────────────────────────────────

if st.button("Run Backtest", type="primary"):
    with st.spinner(f"Backtesting – {hold_period}d hold, retrain every {retrain_freq} periods, {tc_bps} bps costs..."):
        ann_ret, equity_curve, positions = run_backtest(
            df_oos, hold_days=hold_period, tc_bps=tc_bps,
            retrain_every=retrain_freq, seq_len=30
        )

    st.subheader(f"Annualized Return (net): **{ann_ret*100:.2f}%**")

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
            title=f"Equity Curve – {hold_period}d hold / {tc_bps} bps",
            yaxis_title="Cumulative Wealth (1 = start)",
            xaxis_title="Date",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.success("Backtest finished.")
