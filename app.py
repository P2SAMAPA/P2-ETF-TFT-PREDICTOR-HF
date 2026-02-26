"""
P2-ETF-PREDICTOR — TFT Edition
Temporal Fusion Transformer for Fixed Income ETF rotation
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import plotly.graph_objects as go
import os

from utils import get_est_time, is_sync_window
from data_manager import get_data, fetch_etf_data, fetch_macro_data_robust, smart_update_hf_dataset
from models import train_tft, predict_tft
from strategy import execute_strategy, calculate_metrics, calculate_benchmark_metrics

st.set_page_config(page_title="P2-ETF-Predictor | TFT", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    current_time = get_est_time()
    st.write(f"🕒 **EST:** {current_time.strftime('%H:%M:%S')}")
    if is_sync_window():
        st.success("✅ Sync Window Active")
    else:
        st.info("⏸️ Sync Window Inactive")

    st.divider()

    st.subheader("📥 Dataset")
    force_refresh = st.checkbox("Force Dataset Refresh", value=False)
    clean_dataset = st.checkbox("Clean HF Dataset (>30% NaN columns)", value=False)

    st.divider()

    start_yr = st.slider("📅 Start Year", 2008, 2025, 2008)
    fee_bps  = st.slider("💰 Transaction Fee (bps)", 0, 100, 15)

    st.divider()

    st.subheader("🧠 TFT Training")
    epochs   = st.number_input("Max Epochs", 50, 500, 200, step=25,
                               help="Early stopping will halt before this if val_loss plateaus")
    st.caption("Architecture: VSN → GRN → 2× Multi-Head Attention (4 heads) → Softmax")
    st.caption("⚙️ Lookback auto-optimised across 20/30/40/50/60 days on validation set")

    st.divider()

    split_option = st.selectbox("Train/Val/Test Split", ["70/15/15", "80/10/10"], index=0)
    split_ratios = {"70/15/15": (0.70, 0.15, 0.15), "80/10/10": (0.80, 0.10, 0.10)}
    train_pct, val_pct, test_pct = split_ratios[split_option]

    st.divider()

    st.subheader("🛑 Risk Controls")
    stop_loss_pct = st.slider(
        "Stop Loss (2-day cumulative)", min_value=-20, max_value=-8,
        value=-12, step=1, format="%d%%",
        help="Switch to CASH if 2-day return ≤ this threshold"
    ) / 100.0

    z_reentry = st.slider(
        "Re-entry Conviction (σ)", min_value=0.75, max_value=1.50,
        value=1.00, step=0.05, format="%.2f",
        help="Return from CASH when model conviction Z-score ≥ this"
    )

    z_min_entry = st.slider(
        "Min Entry Conviction (σ)", min_value=0.0, max_value=1.5,
        value=0.5, step=0.05, format="%.2f",
        help="Only enter ETF if conviction Z-score ≥ this — below = CASH"
    )

    st.divider()

    run_button          = st.button("🚀 Run TFT Model", type="primary",  use_container_width=True)
    refresh_only_button = st.button("🔄 Refresh Dataset Only", type="secondary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🤖 P2-ETF-PREDICTOR")
st.caption("Temporal Fusion Transformer — Fixed Income ETF Rotation")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET REFRESH ONLY
# ─────────────────────────────────────────────────────────────────────────────
if refresh_only_button:
    st.info("🔄 Refreshing dataset...")
    with st.status("📡 Fetching fresh data...", expanded=True):
        etf_list   = ["TLT", "TBT", "VNQ", "SLV", "GLD", "AGG", "SPY"]
        etf_data   = fetch_etf_data(etf_list)
        macro_data = fetch_macro_data_robust()
        if not etf_data.empty and not macro_data.empty:
            new_df = pd.concat([etf_data, macro_data], axis=1)
            token  = os.getenv("HF_TOKEN")
            if token:
                updated_df = smart_update_hf_dataset(new_df, token)
                st.success("✅ Dataset refresh completed!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows",    len(updated_df))
                c2.metric("Columns", len(updated_df.columns))
                c3.metric("Range",   f"{updated_df.index[0].date()} → {updated_df.index[-1].date()}")
            else:
                st.error("❌ HF_TOKEN not found.")
        else:
            st.error("❌ Failed to fetch data")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if run_button:

    # ── Load data ─────────────────────────────────────────────────────────────
    df = get_data(start_yr, force_refresh=force_refresh, clean_hf_dataset=clean_dataset)
    if df.empty:
        st.error("❌ No data available")
        st.stop()

    years = df.index[-1].year - df.index[0].year + 1
    st.write(f"📅 **Data:** {df.index[0].date()} → {df.index[-1].date()} ({years} years)")

    # ── Identify targets and features ─────────────────────────────────────────
    TARGET_ETFS = ['TLT', 'TBT', 'VNQ', 'SLV', 'GLD']
    target_etfs = [c for c in df.columns
                   if c.endswith('_Ret') and any(e in c for e in TARGET_ETFS)]

    input_features = [c for c in df.columns
                      if (c.endswith('_Z') or c.endswith('_Vol') or
                          'Regime' in c or 'YC_' in c or 'Credit_' in c or
                          'Rates_' in c or 'VIX_Term_' in c or
                          'Rising' in c or 'Falling' in c or 'Accelerating' in c)
                      and c not in target_etfs]

    if not target_etfs or not input_features:
        st.error("❌ Missing features or targets — try Force Dataset Refresh")
        st.stop()

    st.info(f"🎯 **Targets:** {len(target_etfs)} ETFs | **Features:** {len(input_features)} signals")

    # ── Build 5-day forward return targets ────────────────────────────────────
    FORWARD_DAYS = 5
    fwd_returns = pd.DataFrame(index=df.index)
    for col in target_etfs:
        fwd_returns[col] = df[col].rolling(FORWARD_DAYS).sum().shift(-FORWARD_DAYS)

    valid_idx  = fwd_returns.dropna().index
    df_model   = df.loc[valid_idx]
    fwd_model  = fwd_returns.loc[valid_idx]

    # ── Scale features (fit on train only — approx boundary using 60d max) ──
    approx_train_end = int((len(df_model) - 60 - 1) * train_pct)
    scaler = RobustScaler()
    scaler.fit(df_model[input_features].values[:approx_train_end])
    scaled = scaler.transform(df_model[input_features].values)

    fwd_vals = fwd_model[target_etfs].values

    # ── Auto-optimise lookback (20/30/40/50/60 days) on validation loss ─────
    LOOKBACK_CANDIDATES = [20, 30, 40, 50, 60]
    best_lookback   = 30
    best_val_loss   = float('inf')
    lookback_results = {}

    with st.status("🔍 Auto-optimising lookback window...", expanded=False) as status:
        for lb in LOOKBACK_CANDIDATES:
            # Build sequences
            X_lb, y_lb = [], []
            for i in range(lb, len(scaled) - 1):
                X_lb.append(scaled[i - lb:i])
                y_lb.append(int(np.argmax(fwd_vals[i + 1])))
            X_lb = np.array(X_lb, dtype=np.float32)
            y_lb = np.array(y_lb, dtype=np.int32)

            ts = int(len(X_lb) * train_pct)
            vs = int(len(X_lb) * val_pct)

            from models import build_tft_model
            import tensorflow as tf

            probe_model = build_tft_model(
                seq_len=lb,
                num_features=X_lb.shape[2],
                num_classes=len(np.unique(y_lb)),
                units=64, num_heads=4, num_attn_layers=2, dropout_rate=0.15
            )
            probe_model.compile(
                optimizer=tf.keras.optimizers.Adam(5e-4),
                loss='sparse_categorical_crossentropy'
            )
            probe_hist = probe_model.fit(
                X_lb[:ts], y_lb[:ts],
                validation_data=(X_lb[ts:ts+vs], y_lb[ts:ts+vs]),
                epochs=30,
                batch_size=64,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=8, restore_best_weights=True)],
                verbose=0
            )
            val_loss = min(probe_hist.history['val_loss'])
            lookback_results[lb] = round(val_loss, 4)
            st.write(f"  lookback={lb}d → val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lookback = lb

        status.update(label=f"✅ Best lookback: {best_lookback}d (val_loss={best_val_loss:.4f})",
                      state="complete")

    lookback = best_lookback
    st.info(f"📐 **Lookback selected: {lookback} days** | Search results: {lookback_results}")

    # ── Build sequences with optimal lookback and T+1 shift ──────────────────
    X, y_cls, y_fwd, dates_seq = [], [], [], []
    for i in range(lookback, len(scaled) - 1):
        X.append(scaled[i - lookback:i])
        y_fwd.append(fwd_vals[i + 1])
        y_cls.append(int(np.argmax(fwd_vals[i + 1])))
        dates_seq.append(df_model.index[i + 1])

    X      = np.array(X,     dtype=np.float32)
    y_cls  = np.array(y_cls, dtype=np.int32)
    y_fwd  = np.array(y_fwd, dtype=np.float32)

    train_size = int(len(X) * train_pct)
    val_size   = int(len(X) * val_pct)

    X_train    = X[:train_size]
    y_train    = y_cls[:train_size]
    X_val      = X[train_size:train_size + val_size]
    y_val      = y_cls[train_size:train_size + val_size]
    X_test     = X[train_size + val_size:]
    y_fwd_test = y_fwd[train_size + val_size:]
    y_cls_test = y_cls[train_size + val_size:]
    test_dates = pd.DatetimeIndex(dates_seq)[train_size + val_size:]

    st.success(f"✅ Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Train TFT ─────────────────────────────────────────────────────────────
    with st.spinner(f"🧠 Training TFT (up to {epochs} epochs, early stopping active)..."):
        model, history = train_tft(X_train, y_train, X_val, y_val, epochs=int(epochs))
        actual_epochs  = len(history.history['loss'])
        st.success(f"✅ Training complete — stopped at epoch {actual_epochs}")

    # ── Predict ───────────────────────────────────────────────────────────────
    proba = predict_tft(model, X_test)   # shape (N, 5) softmax probabilities

    # ── Accuracy diagnostic ───────────────────────────────────────────────────
    pred_labels = np.argmax(proba, axis=1)
    accuracy    = np.mean(pred_labels == y_cls_test)
    random_bl   = 1.0 / len(target_etfs)
    st.info(
        f"🎯 **5-day Test Accuracy:** {accuracy:.1%} | "
        f"Random baseline: {random_bl:.1%} | "
        f"{'✅ Above random' if accuracy > random_bl else '❌ Below random'}"
    )

    # ── Daily returns for P&L (aligned to test_dates) ─────────────────────────
    daily_ret_test = df.reindex(test_dates)[target_etfs].values

    # ── Risk-free rate ─────────────────────────────────────────────────────────
    sofr        = 0.045
    sofr_source = "fallback (4.5%)"
    try:
        import pandas_datareader.data as web
        dtb3 = web.DataReader('DTB3', 'fred', start='2024-01-01').dropna()
        if not dtb3.empty:
            sofr        = float(dtb3.iloc[-1].values[0]) / 100
            sofr_source = f"FRED DTB3 live ({dtb3.index[-1].date()})"
    except Exception:
        if 'DTB3' in df.columns:
            sofr        = float(df['DTB3'].dropna().iloc[-1]) / 100
            sofr_source = "dataset DTB3"
    st.caption(f"📊 Risk-free rate: **{sofr*100:.2f}%** — {sofr_source}")

    # ── Execute strategy ──────────────────────────────────────────────────────
    (strat_rets, audit_trail, next_signal, next_trading_date,
     conviction_zscore, conviction_label, all_etf_scores) = execute_strategy(
        proba, y_fwd_test, test_dates, target_etfs, fee_bps,
        stop_loss_pct=stop_loss_pct, z_reentry=z_reentry,
        sofr=sofr, z_min_entry=z_min_entry,
        daily_ret_override=daily_ret_test
    )

    metrics = calculate_metrics(strat_rets, sofr)

    # ─────────────────────────────────────────────────────────────────────────
    # NEXT TRADING DAY BANNER
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#00d1b2,#00a896);
                padding:25px;border-radius:15px;text-align:center;
                box-shadow:0 8px 16px rgba(0,0,0,0.3);margin:20px 0;">
        <h1 style="color:white;font-size:48px;margin:0 0 10px 0;
                   font-weight:bold;text-shadow:2px 2px 4px rgba(0,0,0,0.3);">
            🎯 NEXT TRADING DAY
        </h1>
        <h2 style="color:white;font-size:36px;margin:0;font-weight:bold;">
            {next_trading_date} → {next_signal}
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL CONVICTION BLOCK
    # ─────────────────────────────────────────────────────────────────────────
    conviction_colors = {
        "Very High": "#00b894", "High": "#00cec9",
        "Moderate":  "#fdcb6e", "Low":  "#d63031",
    }
    conviction_icons = {
        "Very High": "🟢", "High": "🟢", "Moderate": "🟡", "Low": "🔴",
    }
    conv_color = conviction_colors.get(conviction_label, "#888888")
    conv_dot   = conviction_icons.get(conviction_label, "⚪")
    z_clipped  = max(-3.0, min(3.0, conviction_zscore))
    bar_pct    = int((z_clipped + 3) / 6 * 100)

    etf_names   = [e.replace('_Ret', '') for e in target_etfs]
    sorted_pairs = sorted(zip(etf_names, all_etf_scores),
                          key=lambda x: x[1], reverse=True)
    max_score   = max(float(sorted_pairs[0][1]), 1e-9)

    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #ddd;
                border-left:5px solid {conv_color};border-radius:12px 12px 0 0;
                padding:20px 24px 14px 24px;margin:12px 0 0 0;
                box-shadow:0 2px 8px rgba(0,0,0,0.07);">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap;">
        <span style="font-size:22px;">{conv_dot}</span>
        <span style="font-size:19px;font-weight:700;color:#1a1a1a;">Signal Conviction</span>
        <span style="background:#f0f0f0;border:1px solid {conv_color};
                     color:{conv_color};font-weight:700;font-size:15px;
                     padding:4px 14px;border-radius:8px;">
          Z = {conviction_zscore:.2f} &sigma;
        </span>
        <span style="margin-left:auto;background:{conv_color};color:#fff;
                     font-weight:700;padding:5px 18px;border-radius:20px;font-size:14px;">
          {conviction_label}
        </span>
      </div>
      <div style="display:flex;justify-content:space-between;
                  font-size:11px;color:#999;margin-bottom:5px;">
        <span>Weak &minus;3&sigma;</span><span>Neutral 0&sigma;</span><span>Strong +3&sigma;</span>
      </div>
      <div style="background:#f0f0f0;border-radius:8px;height:16px;
                  overflow:hidden;position:relative;border:1px solid #e0e0e0;">
        <div style="position:absolute;left:50%;top:0;width:2px;height:100%;background:#ccc;"></div>
        <div style="width:{bar_pct}%;height:100%;
                    background:linear-gradient(90deg,#fab1a0,{conv_color});
                    border-radius:8px;"></div>
      </div>
      <div style="font-size:12px;color:#999;margin-top:14px;margin-bottom:2px;">
        Model probability by ETF (ranked high &rarr; low):
      </div>
    </div>
    """, unsafe_allow_html=True)

    for i, (name, score) in enumerate(sorted_pairs):
        bar_w     = int(score / max_score * 100)
        is_winner = (name == next_signal)
        is_last   = (i == len(sorted_pairs) - 1)
        name_style = "font-weight:700;color:#00897b;" if is_winner else "color:#444;"
        bar_color  = conv_color if is_winner else "#b2dfdb" if score > max_score * 0.5 else "#e0e0e0"
        star       = " ★" if is_winner else ""
        bottom_r   = "0 0 12px 12px" if is_last else "0"
        border_bot = "border-bottom:1px solid #f0f0f0;" if not is_last else ""
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #ddd;border-top:none;
                    border-radius:{bottom_r};padding:8px 24px;{border_bot}
                    box-shadow:0 2px 8px rgba(0,0,0,0.07);">
          <div style="display:flex;align-items:center;gap:12px;">
            <span style="width:44px;text-align:right;font-size:13px;{name_style}">{name}{star}</span>
            <div style="flex:1;background:#f5f5f5;border-radius:4px;
                        height:15px;overflow:hidden;border:1px solid #e8e8e8;">
              <div style="width:{bar_w}%;height:100%;background:{bar_color};border-radius:4px;"></div>
            </div>
            <span style="width:56px;font-size:12px;color:#888;text-align:right;">{score:.4f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("Z-score = std deviations the top ETF sits above the mean of all ETF scores.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📈 Ann. Return",   f"{metrics['ann_return']*100:.2f}%",
              delta=f"vs SOFR: {(metrics['ann_return']-sofr)*100:.2f}%")
    c2.metric("📊 Sharpe",        f"{metrics['sharpe']:.2f}",
              delta="Risk-Adjusted" if metrics['sharpe'] > 1 else "Below Threshold")
    c3.metric("🎯 Hit Ratio 15d", f"{metrics['hit_ratio']*100:.0f}%",
              delta="Strong" if metrics['hit_ratio'] > 0.6 else "Weak")
    c4.metric("📉 Max Drawdown",  f"{metrics['max_dd']*100:.2f}%",
              delta="Peak to Trough")
    c5.metric("⚠️ Max Daily DD",  f"{metrics['max_daily_dd']*100:.2f}%",
              delta="Worst Day")

    # ─────────────────────────────────────────────────────────────────────────
    # EQUITY CURVE
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("📈 Out-of-Sample Equity Curve (with Benchmarks)")

    plot_dates = test_dates[:len(metrics['cum_returns'])]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_dates, y=metrics['cum_returns'], mode='lines',
        name='TFT Strategy', line=dict(color='#00d1b2', width=3),
        fill='tozeroy', fillcolor='rgba(0,209,178,0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=plot_dates, y=metrics['cum_max'], mode='lines',
        name='High Water Mark',
        line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash')
    ))

    for bm_col, bm_name, bm_color in [
        ('AGG_Ret', 'AGG (Bond)',   '#ffa500'),
        ('SPY_Ret', 'SPY (Equity)', '#ff4b4b')
    ]:
        if bm_col in df.columns:
            bm_rets = df.reindex(test_dates)[bm_col].values[:len(strat_rets)]
            bm_rets = np.nan_to_num(bm_rets, nan=0.0)
            bm_m    = calculate_benchmark_metrics(bm_rets, sofr)
            fig.add_trace(go.Scatter(
                x=plot_dates, y=bm_m['cum_returns'], mode='lines',
                name=bm_name, line=dict(color=bm_color, width=2, dash='dot')
            ))

    fig.update_layout(
        template="plotly_dark", height=450, hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis_title="Date", yaxis_title="Cumulative Return"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # AUDIT TRAIL
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("📋 Last 20 Days Audit Trail")

    audit_df = pd.DataFrame(audit_trail).tail(20)
    if not audit_df.empty:
        display_cols = ['Date', 'Signal', 'Conviction_Z', 'Net_Return', 'Stop_Active', 'Rotated']
        display_cols = [c for c in display_cols if c in audit_df.columns]
        audit_df = audit_df[display_cols]

        def color_return(val):
            return ('color:#00ff00;font-weight:bold' if val > 0
                    else 'color:#ff4b4b;font-weight:bold')

        styled = (audit_df.style
                  .applymap(color_return, subset=['Net_Return'])
                  .format({'Net_Return': '{:.2%}', 'Conviction_Z': '{:.2f}'})
                  .set_properties(**{'font-size': '16px', 'text-align': 'center'})
                  .set_table_styles([
                      {'selector': 'th', 'props': [('font-size', '17px'),
                                                   ('font-weight', 'bold'),
                                                   ('text-align', 'center')]},
                      {'selector': 'td', 'props': [('padding', '10px')]}
                  ]))
        st.dataframe(styled, use_container_width=True, height=650)

else:
    st.info("👈 Configure parameters and click **🚀 Run TFT Model** to begin")
    current_time = get_est_time()
    st.write(f"🕒 Current EST: **{current_time.strftime('%H:%M:%S')}**")
    if is_sync_window():
        st.success("✅ Sync Window Active — data will auto-update")
    else:
        next_sync = "07:00–08:00" if current_time.hour < 7 else "19:00–20:00"
        st.info(f"⏸️ Next sync window: **{next_sync} EST**")
