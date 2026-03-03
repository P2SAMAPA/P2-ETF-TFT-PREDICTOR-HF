"""
P2-ETF-PREDICTOR — TFT Edition (Display Mode)
===============================================
Reads pre-computed model outputs pushed daily by GitHub Actions.
Replays execute_strategy() live so all sliders work without retraining.

Files read from HF Space repo root:
  - model_outputs.npz   — proba, daily returns, dates, target_etfs
  - signals.json        — next signal, conviction, metadata
  - training_meta.json  — lookback, epochs, accuracy info
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
import requests
import time

from utils import get_est_time, is_sync_window
from data_manager import get_data, fetch_etf_data, fetch_macro_data_robust, smart_update_hf_dataset
from strategy import execute_strategy, calculate_metrics, calculate_benchmark_metrics

st.set_page_config(page_title="P2-ETF-Predictor | TFT", layout="wide")

# ── HF Space raw URL base ─────────────────────────────────────────────────────
HF_SPACE_RAW = "https://huggingface.co/spaces/P2SAMAPA/P2-ETF-TFT-PREDICTOR/resolve/main"


@st.cache_data(ttl=1800)   # refresh cache every 30 min
def load_model_outputs():
    try:
        url = f"{HF_SPACE_RAW}/model_outputs.npz?t={int(time.time())}"
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return {}, f"model_outputs.npz not found (HTTP {r.status_code})"
        from io import BytesIO
        npz = np.load(BytesIO(r.content), allow_pickle=True)
        # Convert NpzFile → plain dict so st.cache_data can pickle it
        data = {k: npz[k] for k in npz.files}
        return data, None
    except Exception as e:
        return {}, str(e)


@st.cache_data(ttl=1800)
def load_signals():
    """Load latest signals.json from HF Space repo."""
    try:
        url = f"{HF_SPACE_RAW}/signals.json?t={int(time.time())}"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None, f"signals.json not found (HTTP {r.status_code})"
        return r.json(), None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=1800)
def load_training_meta():
    """Load training_meta.json from HF Space repo."""
    try:
        url = f"{HF_SPACE_RAW}/training_meta.json?t={int(time.time())}"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


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
    force_refresh  = st.checkbox("Force Dataset Refresh", value=False)
    clean_dataset  = st.checkbox("Clean HF Dataset (>30% NaN columns)", value=False)
    refresh_only_button = st.button("🔄 Refresh Dataset Only",
                                     type="secondary", use_container_width=True)

    st.divider()

    start_yr = st.slider("📅 Start Year (OOS display)", 2008, 2024, 2016)
    fee_bps  = st.slider("💰 Transaction Fee (bps)", 0, 100, 15)

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
    )

    z_min_entry = st.slider(
        "Min Entry Conviction (σ)", min_value=0.0, max_value=1.5,
        value=0.5, step=0.05, format="%.2f",
    )

    st.divider()
    st.caption("🤖 Model retrained daily via GitHub Actions · Split: 80/10/10 (hardcoded)")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🤖 P2-ETF-PREDICTOR")
st.caption("Temporal Fusion Transformer — Fixed Income ETF Rotation")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET REFRESH ONLY (unchanged — still works exactly as before)
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
                c3.metric("Range",   f"{updated_df.index[0].date()} → "
                                     f"{updated_df.index[-1].date()}")
            else:
                st.error("❌ HF_TOKEN not found.")
        else:
            st.error("❌ Failed to fetch data")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD PRE-COMPUTED OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("📦 Loading pre-computed model outputs..."):
    outputs, err = load_model_outputs()
    signals, sig_err = load_signals()
    meta = load_training_meta()

if not outputs:@st.cache_data(ttl=1800)
    st.error(f"❌ Could not load model outputs: {err}")
    st.info("💡 The model may not have been trained yet. "
            "Trigger the GitHub Actions workflow manually to run the first training.")
    st.stop()

if signals is None:
    st.warning(f"⚠️ Could not load signals.json: {sig_err}")

# ── Extract arrays ────────────────────────────────────────────────────────────
proba          = outputs['proba']           # (N, 7)
daily_ret_test = outputs['daily_ret_test']  # (N, 7)
y_fwd_test     = outputs['y_fwd_test']      # (N, 7)
spy_ret_test   = outputs['spy_ret_test']    # (N,)
agg_ret_test   = outputs['agg_ret_test']    # (N,)
test_dates     = pd.DatetimeIndex(outputs['test_dates'])
target_etfs    = list(outputs['target_etfs'])
sofr           = float(outputs['sofr'][0])
etf_names      = [e.replace('_Ret', '') for e in target_etfs]

# ── Apply start year filter ───────────────────────────────────────────────────
start_mask = test_dates.year >= start_yr
if start_mask.sum() < 50:
    st.warning(f"⚠️ Less than 50 test days after {start_yr} filter. Showing all data.")
    start_mask = np.ones(len(test_dates), dtype=bool)

proba_f      = proba[start_mask]
daily_ret_f  = daily_ret_test[start_mask]
y_fwd_f      = y_fwd_test[start_mask]
spy_ret_f    = spy_ret_test[start_mask]
agg_ret_f    = agg_ret_test[start_mask]
test_dates_f = test_dates[start_mask]

# ── Show dataset info ─────────────────────────────────────────────────────────
if signals:
    st.info(f"📅 **Data:** {signals['data_start']} → {signals['data_end']} | "
            f"**OOS Test:** {test_dates_f[0].date()} → {test_dates_f[-1].date()} "
            f"({len(test_dates_f)} days) | "
            f"🕒 Last trained: {signals['run_timestamp_utc'][:10]}")
    if meta:
        st.caption(f"📐 Lookback: {meta['lookback_days']}d · "
                   f"Features: {meta['n_features']} · "
                   f"Split: {meta['split']} · "
                   f"Targets: {', '.join(etf_names)}")

# ─────────────────────────────────────────────────────────────────────────────
# LIVE STRATEGY REPLAY (all sliders applied here — no retraining needed)
# ─────────────────────────────────────────────────────────────────────────────
(strat_rets, audit_trail, next_signal, next_trading_date,
 conviction_zscore, conviction_label, all_etf_scores) = execute_strategy(
    proba_f, y_fwd_f, test_dates_f, target_etfs,
    fee_bps,
    stop_loss_pct=stop_loss_pct,
    z_reentry=z_reentry,
    sofr=sofr,
    z_min_entry=z_min_entry,
    daily_ret_override=daily_ret_f
)

metrics = calculate_metrics(strat_rets, sofr)

# ── Accuracy info from meta ───────────────────────────────────────────────────
if meta and 'accuracy_per_etf' in meta:
    st.info(f"🎯 **Binary Accuracy per ETF:** {meta['accuracy_per_etf']} | "
            f"Random baseline: 50.0%")

# ─────────────────────────────────────────────────────────────────────────────
# NEXT TRADING DAY BANNER
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL CONVICTION BLOCK
# ─────────────────────────────────────────────────────────────────────────────
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

sorted_pairs = sorted(zip(etf_names, all_etf_scores),
                      key=lambda x: x[1], reverse=True)
max_score = max(float(sorted_pairs[0][1]), 1e-9)

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

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
excess = (metrics['ann_return'] - sofr) * 100
c1.metric("📈 Ann. Return",   f"{metrics['ann_return']*100:.2f}%",
          delta=f"{excess:+.1f}pp vs T-Bill")
c2.metric("📊 Sharpe",        f"{metrics['sharpe']:.2f}",
          delta="Above 1.0 ✓" if metrics['sharpe'] > 1 else "Below 1.0")
c3.metric("🎯 Hit Ratio 15d", f"{metrics['hit_ratio']*100:.0f}%",
          delta="Strong" if metrics['hit_ratio'] > 0.6 else "Weak")
c4.metric("📉 Max Drawdown",  f"{metrics['max_dd']*100:.2f}%",
          delta="Peak to Trough")
c5.metric("⚠️ Max Daily DD",  f"{metrics['max_daily_dd']*100:.2f}%",
          delta="Worst Day")

# ─────────────────────────────────────────────────────────────────────────────
# EQUITY CURVE
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📈 Out-of-Sample Equity Curve (with Benchmarks)")

plot_dates = test_dates_f[:len(metrics['cum_returns'])]
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

# Benchmarks
spy_m = calculate_benchmark_metrics(
    np.nan_to_num(spy_ret_f[:len(strat_rets)], nan=0.0), sofr)
agg_m = calculate_benchmark_metrics(
    np.nan_to_num(agg_ret_f[:len(strat_rets)], nan=0.0), sofr)

fig.add_trace(go.Scatter(
    x=plot_dates, y=spy_m['cum_returns'], mode='lines',
    name='SPY (Equity)', line=dict(color='#ff4b4b', width=2, dash='dot')
))
fig.add_trace(go.Scatter(
    x=plot_dates, y=agg_m['cum_returns'], mode='lines',
    name='AGG (Bond)', line=dict(color='#ffa500', width=2, dash='dot')
))

fig.update_layout(
    template="plotly_dark", height=450, hovermode='x unified',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    xaxis_title="Date", yaxis_title="Cumulative Return"
)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUDIT TRAIL
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📋 Last 20 Days Audit Trail")
audit_df = pd.DataFrame(audit_trail).tail(20)
if not audit_df.empty:
    display_cols = ['Date', 'Signal', 'Top_Pick', 'Conviction_Z',
                    'Net_Return', 'Stop_Active', 'Rotated']
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

# ─────────────────────────────────────────────────────────────────────────────
# METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📖 Methodology & Model Notes")
lookback_display = meta['lookback_days'] if meta else "auto"
rf_label_display = signals['rf_label'] if signals else "4.5% fallback"

st.markdown(f"""
<div style="background:#1a1a2e;border:1px solid #2d2d4e;border-radius:12px;
            padding:28px 32px;color:#e0e0e0;font-size:14px;line-height:1.8;">

<h4 style="color:#00d1b2;margin-top:0;">🏗️ Model Architecture — 7 Binary Temporal Fusion Transformers</h4>
<p>One independent binary TFT per ETF. Each model answers:
<em>"Will this ETF beat the risk-free rate (3M T-Bill) over the next 5 trading days?"</em>
At inference time, ETFs are ranked by their confidence probability.</p>

<h4 style="color:#00d1b2;margin-top:20px;">📊 Training Methodology</h4>
<ul>
  <li><b>Split:</b> 80% train / 10% val / 10% test — strictly chronological</li>
  <li><b>Lookback auto-optimised:</b> Best window = <b>{lookback_display} days</b></li>
  <li><b>Retrained daily</b> via GitHub Actions on the latest data from HF Dataset</li>
  <li><b>Risk-free rate:</b> {sofr*100:.2f}% ({rf_label_display})</li>
</ul>

<h4 style="color:#00d1b2;margin-top:20px;">⚙️ Strategy Execution (live, applied to saved predictions)</h4>
<ul>
  <li><b>Conviction gate (σ={z_min_entry}):</b> Only enter if top ETF sits ≥ {z_min_entry}σ above mean</li>
  <li><b>Trailing stop-loss ({stop_loss_pct*100:.0f}%):</b> Switch to CASH if 2-day cumulative ≤ threshold</li>
  <li><b>Re-entry:</b> Return from CASH when conviction Z ≥ {z_reentry}σ</li>
  <li><b>5-day loss rotation:</b> Rotate to #2 ETF if top pick loses every day for 5 days</li>
  <li><b>Transaction cost:</b> {fee_bps}bps on every entry/re-entry</li>
</ul>

<h4 style="color:#00d1b2;margin-top:20px;">⚠️ Disclaimer</h4>
<p>Past performance does not guarantee future results. This tool is for research
and educational purposes only and does not constitute financial advice.</p>
</div>
""", unsafe_allow_html=True)
