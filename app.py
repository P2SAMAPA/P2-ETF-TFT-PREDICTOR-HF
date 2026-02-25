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

    st.subheader("🛑 Trailing Stop Loss")
    stop_loss_pct = st.slider(
        "Stop Loss (2-day cumulative return)",
        min_value=-20, max_value=-8, value=-12, step=1,
        format="%d%%",
        help="If 2-day cumulative return ≤ this value, switch to CASH."
    ) / 100.0

    z_reentry = st.slider(
        "Re-entry Conviction Z-score (σ)",
        min_value=0.75, max_value=1.50, value=1.00, step=0.05,
        format="%.2f",
        help="Return to ETF when model conviction Z-score exceeds this threshold."
    )

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

        # ── T+1 fix: features at day i predict returns at day i+1 ──────────
        # Drop last row of X (no future return known), drop first row of y_raw
        X      = X[:-1]
        y_raw  = y_raw[1:]
        # Align dates: test dates shift forward by 1 (execution is next day)
        dates_aligned = df.index[1:]

        y = np.argmax(y_raw, axis=1)

        train_size = int(len(X) * train_pct)
        val_size   = int(len(X) * val_pct)

        X_train    = X[:train_size]
        y_train    = y[:train_size]
        X_val      = X[train_size:train_size + val_size]
        y_val      = y[train_size:train_size + val_size]
        X_test     = X[train_size + val_size:]
        y_raw_test = y_raw[train_size + val_size:]

        test_dates = dates_aligned[train_size + val_size:]

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

        last_proba  = ensemble_proba[-1]
        preds       = predict_ensemble(rf_model, xgb_model, X_test)
        model_type  = "ensemble"
        all_proba   = ensemble_proba   # full per-day probabilities for stop re-entry

        # ── Diagnostic: how often does model pick the correct best ETF? ──────
        y_test_labels = np.argmax(y_raw_test, axis=1)
        accuracy = np.mean(preds == y_test_labels)
        random_baseline = 1.0 / len(target_etfs)
        st.info(f"🎯 **Test Accuracy:** {accuracy:.1%} | Random baseline: {random_baseline:.1%} | "
                f"{'✅ Above random' if accuracy > random_baseline else '❌ Below random — inverse pattern learned'}")

    else:
        # Transformer: sequences
        # Fit scaler only on training portion to avoid leakage
        n_total = len(df[input_features])
        # Approximate train boundary before sequence building
        approx_train_end = int((n_total - 1) * train_pct)
        scaler = MinMaxScaler()
        scaler.fit(df[input_features].values[:approx_train_end])
        scaled_input = scaler.transform(df[input_features].values)

        X, y, dates_seq = [], [], []
        for i in range(lookback, len(scaled_input) - 1):  # -1 for T+1 shift
            X.append(scaled_input[i-lookback:i])
            y.append(df[target_etfs].iloc[i + 1].values)  # T+1: next day's returns
            dates_seq.append(df.index[i + 1])             # execution date = next day

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

        preds      = model.predict(X_test, verbose=0)
        test_dates = np.array(dates_seq)[train_size + val_size:]
        y_raw_test = y_test
        model_type = "transformer"
        all_proba  = None   # transformer uses raw preds directly

    # Execute strategy
    # Risk-free rate: fetch DTB3 (3-Month T-Bill) live from FRED
    sofr = 0.045
    sofr_source = "fallback (4.5%)"
    try:
        import pandas_datareader.data as web
        dtb3 = web.DataReader('DTB3', 'fred', start='2024-01-01')
        dtb3 = dtb3.dropna()
        if not dtb3.empty:
            sofr = float(dtb3.iloc[-1].values[0]) / 100
            sofr_source = f"FRED DTB3 live ({dtb3.index[-1].date()})"
    except Exception as ex:
        if 'DTB3' in df.columns:
            sofr = float(df['DTB3'].dropna().iloc[-1]) / 100
            sofr_source = "dataset DTB3 column"

    st.caption(f"📊 Risk-free rate: **{sofr*100:.2f}%** — source: {sofr_source}")

    (strat_rets, audit_trail, next_signal, next_trading_date,
     conviction_zscore, conviction_label, all_etf_scores) = execute_strategy(
        preds, y_raw_test, test_dates, target_etfs, fee_bps, model_type,
        stop_loss_pct=stop_loss_pct, z_reentry=z_reentry,
        sofr=sofr, all_proba=all_proba
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
        "Very High": "#00b894",
        "High":      "#00cec9",
        "Moderate":  "#fdcb6e",
        "Low":       "#d63031",
    }
    conviction_icons = {
        "Very High": "🟢", "High": "🟢", "Moderate": "🟡", "Low": "🔴",
    }
    conv_color = conviction_colors.get(conviction_label, "#888888")
    conv_dot   = conviction_icons.get(conviction_label, "⚪")

    z_clipped = max(-3.0, min(3.0, conviction_zscore))
    bar_pct   = int((z_clipped + 3) / 6 * 100)

    etf_names = [e.replace('_Ret', '') for e in target_etfs]
    sorted_pairs = sorted(zip(etf_names, all_etf_scores), key=lambda x: x[1], reverse=True)
    max_score = float(sorted_pairs[0][1]) if sorted_pairs[0][1] > 0 else 1.0

    # ── Header + gauge ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#ffffff; border:1px solid #ddd;
                border-left:5px solid {conv_color}; border-radius:12px 12px 0 0;
                padding:20px 24px 14px 24px; margin:12px 0 0 0;
                box-shadow:0 2px 8px rgba(0,0,0,0.07);">

      <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px; flex-wrap:wrap;">
        <span style="font-size:22px;">{conv_dot}</span>
        <span style="font-size:19px; font-weight:700; color:#1a1a1a;">Signal Conviction</span>
        <span style="background:#f0f0f0; border:1px solid {conv_color};
                     color:{conv_color}; font-weight:700; font-size:15px;
                     padding:4px 14px; border-radius:8px;">
          Z = {conviction_zscore:.2f} &sigma;
        </span>
        <span style="margin-left:auto; background:{conv_color}; color:#fff;
                     font-weight:700; padding:5px 18px; border-radius:20px; font-size:14px;">
          {conviction_label}
        </span>
      </div>

      <div style="display:flex; justify-content:space-between;
                  font-size:11px; color:#999; margin-bottom:5px;">
        <span>Weak &minus;3&sigma;</span>
        <span>Neutral 0&sigma;</span>
        <span>Strong +3&sigma;</span>
      </div>
      <div style="background:#f0f0f0; border-radius:8px; height:16px;
                  overflow:hidden; position:relative; border:1px solid #e0e0e0;">
        <div style="position:absolute; left:50%; top:0; width:2px;
                    height:100%; background:#ccc;"></div>
        <div style="width:{bar_pct}%; height:100%;
                    background:linear-gradient(90deg, #fab1a0, {conv_color});
                    border-radius:8px;"></div>
      </div>

      <div style="font-size:12px; color:#999; margin-top:14px; margin-bottom:2px;">
        Model probability by ETF (ranked high &rarr; low):
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ETF bars — one st.markdown per ETF avoids Streamlit HTML escaping ───
    for i, (name, score) in enumerate(sorted_pairs):
        bar_w     = int(score / max_score * 100)
        is_winner = (name == next_signal)
        is_last    = (i == len(sorted_pairs) - 1)
        name_style = "font-weight:700; color:#00897b;" if is_winner else "color:#444;"
        bar_color  = conv_color if is_winner else "#b2dfdb" if score > max_score * 0.5 else "#e0e0e0"
        star       = " ★" if is_winner else ""
        bottom_r   = "0 0 12px 12px" if is_last else "0"
        border_bot = "border-bottom:1px solid #f0f0f0;" if not is_last else ""

        st.markdown(f"""
        <div style="background:#ffffff; border:1px solid #ddd; border-top:none;
                    border-radius:{bottom_r}; padding:8px 24px; {border_bot}
                    box-shadow:0 2px 8px rgba(0,0,0,0.07);">
          <div style="display:flex; align-items:center; gap:12px;">
            <span style="width:44px; text-align:right; font-size:13px; {name_style}">{name}{star}</span>
            <div style="flex:1; background:#f5f5f5; border-radius:4px;
                        height:15px; overflow:hidden; border:1px solid #e8e8e8;">
              <div style="width:{bar_w}%; height:100%;
                          background:{bar_color}; border-radius:4px;"></div>
            </div>
            <span style="width:56px; font-size:12px; color:#888; text-align:right;">{score:.4f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("Z-score = std deviations the top ETF sits above the mean of all ETF scores. Higher → model is more decisive.")

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
