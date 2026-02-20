"""
P2-ETF-PREDICTOR - Modular Version
Main Streamlit application
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our modules
from utils import get_est_time, is_sync_window
from data_manager import get_data, fetch_etf_data, fetch_macro_data_robust, smart_update_hf_dataset
from models import train_transformer, train_ensemble, predict_ensemble
from strategy import execute_strategy, calculate_metrics, calculate_benchmark_metrics

import os

st.set_page_config(page_title="P2-ETF-Predictor", layout="wide")

# ------------------------------
# SIDEBAR CONFIGURATION
# ------------------------------
with st.sidebar:
    st.header("⚙️ Model Configuration")

    current_time = get_est_time()
    st.write(f"🕒 **Server Time (EST):** {current_time.strftime('%H:%M:%S')}")

    if is_sync_window():
        st.success("✅ **Sync Window Active**")
    else:
        st.info("⏸️ Sync Window Inactive")

    st.divider()

    st.subheader("📥 Dataset Management")
    force_refresh = st.checkbox(
        "Force Dataset Refresh",
        value=False,
        help="Manually fetch fresh data and update HF dataset"
    )

    clean_dataset = st.checkbox(
        "Clean HF Dataset (Remove NaN-heavy columns)",
        value=False,
        help="Remove columns with >30% missing data"
    )

    st.divider()

    start_yr = st.slider("📅 Start Year", 2008, 2025, 2016)
    fee_bps = st.slider("💰 Transaction Fee (bps)", 0, 100, 15)

    st.divider()

    st.subheader("🧠 Model Selection")
    model_option = st.selectbox(
        "Choose Model",
        [
            "Option A: Pure Transformer",
            "Option B: Random Forest + XGBoost"
        ],
        index=1
    )

    st.divider()

    st.subheader("⚙️ Training Settings")

    if "Option A" in model_option:
        epochs = st.number_input("Epochs", 10, 500, 100, step=10)
        lookback = st.slider("Lookback Days", 20, 60, 30, step=5)
        st.caption("ℹ️ Transformer: 2 heads, 1 layer, 64 FF dim")
    else:
        epochs = 100
        lookback = 30
        st.info("ℹ️ RF+XGBoost: 500 trees/rounds with early stopping")

    st.divider()

    st.subheader("📊 Data Split Strategy")

    split_option = st.selectbox(
        "Train/Val/Test Split",
        ["70/15/15", "80/10/10"],
        index=0
    )

    split_ratios = {
        "70/15/15": (0.70, 0.15, 0.15),
        "80/10/10": (0.80, 0.10, 0.10)
    }
    train_pct, val_pct, test_pct = split_ratios[split_option]

    st.divider()

    run_button = st.button("🚀 Execute Model", type="primary", use_container_width=True)

    st.divider()

    refresh_only_button = st.button("🔄 Refresh Dataset Only", type="secondary",
                                    use_container_width=True)

# ------------------------------
# MAIN APPLICATION
# ------------------------------
st.title("🤖 P2-ETF-PREDICTOR")
st.caption("Multi-Model Ensemble: Transformer, Random Forest, XGBoost")

# Dataset refresh only
if refresh_only_button:
    st.info("🔄 Refreshing dataset...")

    with st.status("📡 Fetching fresh data...", expanded=True):
        etf_list = ["TLT", "TBT", "VNQ", "SLV", "GLD", "AGG", "SPY"]

        etf_data = fetch_etf_data(etf_list)
        macro_data = fetch_macro_data_robust()

        if not etf_data.empty and not macro_data.empty:
            new_df = pd.concat([etf_data, macro_data], axis=1)

            token = os.getenv("HF_TOKEN")

            if token:
                updated_df = smart_update_hf_dataset(new_df, token)

                st.success("✅ Dataset refresh completed!")

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Rows", len(updated_df))
                col2.metric("Total Columns", len(updated_df.columns))
                col3.metric("Date Range", f"{updated_df.index[0].strftime('%Y-%m-%d')} to {updated_df.index[-1].strftime('%Y-%m-%d')}")
            else:
                st.error("❌ HF_TOKEN not found.")
        else:
            st.error("❌ Failed to fetch data")

    st.stop()

# Main execution
if run_button:
    # Load data
    df = get_data(start_yr, force_refresh=force_refresh, clean_hf_dataset=clean_dataset)

    if df.empty:
        st.error("❌ No data available")
        st.stop()

    years_of_data = df.index[-1].year - df.index[0].year + 1

    st.write(f"📅 **Data Range:** {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} ({years_of_data} years)")

    # Identify features and targets
    target_etfs = [col for col in df.columns if col.endswith('_Ret') and
                   any(etf in col for etf in ['TLT', 'TBT', 'VNQ', 'SLV', 'GLD'])]

    input_features = [col for col in df.columns
                     if (col.endswith('_Z') or col.endswith('_Vol') or
                         'Regime' in col or 'YC_' in col or 'Credit_' in col or
                         'Rates_' in col or 'VIX_Term_' in col)
                     and col not in target_etfs]

    if not target_etfs or not input_features:
        st.error("❌ Missing required features or targets")
        st.stop()

    st.info(f"🎯 **Targets:** {len(target_etfs)} ETFs | **Features:** {len(input_features)} signals")

    # Prepare data
    last_proba = None   # will hold per-ETF probabilities for conviction

    if "Option B" in model_option:
        # Ensemble: flat features
        X = df[input_features].values
        y_raw = df[target_etfs].values
        y = np.argmax(y_raw, axis=1)

        train_size = int(len(X) * train_pct)
        val_size = int(len(X) * val_pct)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_raw_test = y_raw[train_size + val_size:]

        st.success(f"✅ Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

        # Train
        with st.spinner("🌲 Training ensemble..."):
            rf_model, xgb_model = train_ensemble(X_train, y_train, X_val, y_val)
            st.success("✅ Training completed!")

        # ── Get class probabilities for richer conviction scoring ───────────
        rf_proba = rf_model.predict_proba(X_test)
        try:
            xgb_proba = xgb_model.predict_proba(X_test)
            ensemble_proba = (rf_proba + xgb_proba) / 2
        except Exception:
            ensemble_proba = rf_proba

        last_proba = ensemble_proba[-1]   # shape: (n_etfs,) for the last prediction

        preds = predict_ensemble(rf_model, xgb_model, X_test)
        test_dates = df.index[train_size + val_size:]
        model_type = "ensemble"

    else:
        # Transformer: sequences
        scaler = MinMaxScaler()
        scaled_input = scaler.fit_transform(df[input_features])

        X, y = [], []
        for i in range(lookback, len(scaled_input)):
            X.append(scaled_input[i-lookback:i])
            y.append(df[target_etfs].iloc[i].values)

        X = np.array(X)
        y = np.array(y)

        train_size = int(len(X) * train_pct)
        val_size = int(len(X) * val_pct)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        st.success(f"✅ Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

        # Train
        with st.spinner("🧠 Training Transformer..."):
            model, history = train_transformer(X_train, y_train, X_val, y_val, epochs=int(epochs))
            st.success(f"✅ Training completed!")

        preds = model.predict(X_test, verbose=0)
        test_dates = df.index[lookback + train_size + val_size:]
        y_raw_test = y_test
        model_type = "transformer"

    # Execute strategy
    sofr = df['T10Y3M'].iloc[-1] / 100 if 'T10Y3M' in df.columns else 0.045

    (strat_rets, audit_trail, next_signal, next_trading_date,
     conviction_zscore, conviction_label, all_etf_scores) = execute_strategy(
        preds, y_raw_test, test_dates, target_etfs, fee_bps, model_type
    )

    # ── Override scores with richer RF/XGB probabilities when available ─────
    if model_type == "ensemble" and last_proba is not None:
        from strategy import compute_signal_conviction
        all_etf_scores = last_proba
        _, conviction_zscore, conviction_label = compute_signal_conviction(last_proba)

    # Calculate metrics
    metrics = calculate_metrics(strat_rets, sofr)

    # ── NEXT TRADING DAY banner ──────────────────────────────────────────────
    st.divider()

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #00d1b2 0%, #00a896 100%);
                padding: 25px; border-radius: 15px; text-align: center;
                box-shadow: 0 8px 16px rgba(0,0,0,0.3); margin: 20px 0;">
        <h1 style="color: white; font-size: 48px; margin: 0 0 10px 0;
                   font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🎯 NEXT TRADING DAY
        </h1>
        <h2 style="color: white; font-size: 36px; margin: 0; font-weight: bold;">
            {next_trading_date.strftime('%Y-%m-%d')} → {next_signal}
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # ── SIGNAL CONVICTION block ──────────────────────────────────────────────
    conviction_colors = {
        "Very High": ("#00d1b2", "🟢"),
        "High":      ("#4CAF50", "🟢"),
        "Moderate":  ("#FFA500", "🟡"),
        "Low":       ("#ff4b4b", "🔴"),
    }
    conv_color, conv_dot = conviction_colors.get(conviction_label, ("#888", "⚪"))

    # Z-score gauge: map [-3, 3] → [0%, 100%]
    z_clipped = max(-3.0, min(3.0, conviction_zscore))
    bar_pct = int((z_clipped + 3) / 6 * 100)

    # Build per-ETF horizontal bar rows
    etf_names = [e.replace('_Ret', '') for e in target_etfs]
    max_score = float(np.max(all_etf_scores)) if np.max(all_etf_scores) > 0 else 1.0
    etf_score_rows = ""
    for name, score in zip(etf_names, all_etf_scores):
        bar_w = int(score / max_score * 100)
        is_winner = name == next_signal
        highlight = "font-weight:bold; color:#00d1b2;" if is_winner else "color:#ccc;"
        star = " ★" if is_winner else ""
        bar_color = "#00d1b2" if is_winner else "#555"
        etf_score_rows += f"""
        <div style="display:flex; align-items:center; gap:10px; margin:5px 0;">
          <span style="width:42px; text-align:right; font-size:13px; {highlight}">{name}{star}</span>
          <div style="flex:1; background:#2a2a3e; border-radius:4px; height:15px; overflow:hidden;">
            <div style="width:{bar_w}%; height:100%; background:{bar_color}; border-radius:4px;"></div>
          </div>
          <span style="width:52px; font-size:12px; color:#aaa; text-align:right;">{score:.4f}</span>
        </div>"""

    st.markdown(f"""
    <div style="background:#1a1a2e; border:1px solid {conv_color}44; border-left: 4px solid {conv_color};
                border-radius:12px; padding:20px 25px; margin:10px 0 22px 0;
                box-shadow: 0 4px 16px rgba(0,0,0,0.5);">

      <!-- Header row -->
      <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
        <span style="font-size:24px;">{conv_dot}</span>
        <span style="font-size:20px; font-weight:bold; color:white; letter-spacing:0.5px;">
          Signal Conviction
        </span>
        <span style="margin-left:auto; background:{conv_color}; color:#000; font-weight:bold;
                     padding:5px 16px; border-radius:20px; font-size:14px; letter-spacing:0.5px;">
          {conviction_label}
        </span>
      </div>

      <!-- Z-score gauge -->
      <div style="margin-bottom:18px;">
        <div style="display:flex; justify-content:space-between; font-size:11px;
                    color:#666; margin-bottom:5px;">
          <span>Weak  −3σ</span>
          <span style="color:{conv_color}; font-weight:bold; font-size:14px;">
            Z-Score = {conviction_zscore:.2f} σ
          </span>
          <span>Strong  +3σ</span>
        </div>
        <div style="background:#2a2a3e; border-radius:8px; height:18px;
                    overflow:hidden; position:relative;">
          <div style="position:absolute; left:50%; top:0; width:2px;
                      height:100%; background:#444;"></div>
          <div style="width:{bar_pct}%; height:100%;
                      background:linear-gradient(90deg, #005f73, {conv_color});
                      border-radius:8px;"></div>
        </div>
      </div>

      <!-- Per-ETF breakdown -->
      <div style="font-size:12px; color:#777; margin-bottom:8px; letter-spacing:0.3px;">
        Model score breakdown across all candidate ETFs:
      </div>
      {etf_score_rows}

      <div style="margin-top:14px; font-size:11px; color:#555; line-height:1.5;">
        Z-score = how many standard deviations the selected ETF's model score sits
        above the mean of all ETF scores.  Higher → model is more decisive.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── METRICS ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "📈 Annualized Return",
        f"{metrics['ann_return'] * 100:.2f}%",
        delta=f"vs SOFR: {(metrics['ann_return'] - sofr) * 100:.2f}%"
    )

    col2.metric(
        "📊 Sharpe Ratio",
        f"{metrics['sharpe']:.2f}",
        delta="Risk-Adjusted" if metrics['sharpe'] > 1 else "Below Threshold"
    )

    col3.metric(
        "🎯 Hit Ratio (15d)",
        f"{metrics['hit_ratio'] * 100:.0f}%",
        delta="Strong" if metrics['hit_ratio'] > 0.6 else "Weak"
    )

    col4.metric(
        "📉 Max Drawdown",
        f"{metrics['max_dd'] * 100:.2f}%",
        delta="Peak to Trough"
    )

    col5.metric(
        "⚠️ Max Daily DD",
        f"{metrics['max_daily_dd'] * 100:.2f}%",
        delta="Worst Day"
    )

    # ── EQUITY CURVE ─────────────────────────────────────────────────────────
    st.subheader("📈 Out-of-Sample Equity Curve (with Benchmarks)")

    fig_equity = go.Figure()

    if model_type == "ensemble":
        plot_dates = df.index[train_size + val_size:][:len(metrics['cum_returns'])]
    else:
        plot_dates = df.index[lookback + train_size + val_size:][:len(metrics['cum_returns'])]

    fig_equity.add_trace(go.Scatter(
        x=plot_dates,
        y=metrics['cum_returns'],
        mode='lines',
        name='Strategy',
        line=dict(color='#00d1b2', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 209, 178, 0.1)'
    ))

    fig_equity.add_trace(go.Scatter(
        x=plot_dates,
        y=metrics['cum_max'],
        mode='lines',
        name='High Water Mark',
        line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash')
    ))

    if 'AGG_Ret' in df.columns:
        if model_type == "ensemble":
            agg_returns = df['AGG_Ret'].iloc[train_size + val_size:].values[:len(strat_rets)]
        else:
            agg_returns = df['AGG_Ret'].iloc[lookback + train_size + val_size:].values[:len(strat_rets)]

        agg_metrics = calculate_benchmark_metrics(agg_returns, sofr)
        fig_equity.add_trace(go.Scatter(
            x=plot_dates,
            y=agg_metrics['cum_returns'],
            mode='lines',
            name='AGG (Bond Benchmark)',
            line=dict(color='#ffa500', width=2, dash='dot')
        ))

    if 'SPY_Ret' in df.columns:
        if model_type == "ensemble":
            spy_returns = df['SPY_Ret'].iloc[train_size + val_size:].values[:len(strat_rets)]
        else:
            spy_returns = df['SPY_Ret'].iloc[lookback + train_size + val_size:].values[:len(strat_rets)]

        spy_metrics = calculate_benchmark_metrics(spy_returns, sofr)
        fig_equity.add_trace(go.Scatter(
            x=plot_dates,
            y=spy_metrics['cum_returns'],
            mode='lines',
            name='SPY (Equity Benchmark)',
            line=dict(color='#ff4b4b', width=2, dash='dot')
        ))

    fig_equity.update_layout(
        template="plotly_dark",
        height=450,
        hovermode='x unified',
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis_title="Date",
        yaxis_title="Cumulative Return"
    )

    st.plotly_chart(fig_equity, use_container_width=True)

    # ── AUDIT TRAIL ──────────────────────────────────────────────────────────
    st.subheader("📋 Last 15 Days Audit Trail")

    audit_df = pd.DataFrame(audit_trail).tail(15)[['Date', 'Signal', 'Net_Return']]

    def color_return(val):
        return 'color: #00ff00; font-weight: bold' if val > 0 else 'color: #ff4b4b; font-weight: bold'

    styled_audit = audit_df.style.applymap(
        color_return,
        subset=['Net_Return']
    ).format({
        'Net_Return': '{:.2%}'
    }).set_properties(**{
        'font-size': '18px',
        'text-align': 'center'
    }).set_table_styles([
        {'selector': 'th', 'props': [('font-size', '20px'), ('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('padding', '12px')]}
    ])

    st.dataframe(styled_audit, use_container_width=True, height=600)

else:
    st.info("👈 Configure parameters in the sidebar and click '🚀 Execute Model' to begin")

    current_time = get_est_time()
    st.write(f"🕒 Current EST Time: **{current_time.strftime('%H:%M:%S')}**")

    if is_sync_window():
        st.success("✅ **Sync Window Active** - Data will be updated")
    else:
        next_sync = "07:00-08:00" if current_time.hour < 7 else "19:00-20:00"
        st.info(f"⏸️ Next sync window: **{next_sync} EST**")
