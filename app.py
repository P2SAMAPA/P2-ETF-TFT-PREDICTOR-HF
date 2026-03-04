"""
P2-ETF-PREDICTOR — TFT Edition
================================
Tab 1: Single-Year Results    — user picks start year, triggers training
Tab 2: Multi-Year Consensus   — sweeps 2008/2014/2016/2019/2021, cached results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
import time
import requests as req

from utils import get_est_time, is_sync_window
from data_manager import get_data, fetch_etf_data, fetch_macro_data_robust, smart_update_hf_dataset
from strategy import execute_strategy, calculate_metrics, calculate_benchmark_metrics

st.set_page_config(page_title="P2-ETF-Predictor | TFT", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
HF_OUTPUT_REPO  = "P2SAMAPA/p2-etf-tft-outputs"
GITHUB_REPO     = "P2SAMAPA/P2-ETF-TFT-PREDICTOR-HF-DATASET"
GITHUB_WORKFLOW = "train_and_push.yml"
GITHUB_API_BASE = "https://api.github.com"
SWEEP_YEARS     = [2008, 2014, 2016, 2019, 2021]

ETF_COLORS = {
    "TLT": "#4e79a7", "VCIT": "#f28e2b", "LQD": "#59a14f",
    "HYG": "#e15759", "VNQ": "#76b7b2", "SLV": "#edc948",
    "GLD": "#b07aa1",
}


# ── HF helpers ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_model_outputs():
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_OUTPUT_REPO, filename="model_outputs.npz",
                               repo_type="dataset", force_download=True)
        npz = np.load(path, allow_pickle=True)
        return {k: npz[k] for k in npz.files}, None
    except Exception as e:
        return {}, str(e)


@st.cache_data(ttl=300)
def load_signals():
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_OUTPUT_REPO, filename="signals.json",
                               repo_type="dataset", force_download=True)
        with open(path) as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=300)
def load_training_meta():
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_OUTPUT_REPO, filename="training_meta.json",
                               repo_type="dataset", force_download=True)
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _today_est():
    from datetime import datetime as _dt, timezone, timedelta
    return (_dt.now(timezone.utc) - timedelta(hours=5)).date()


@st.cache_data(ttl=60)
def load_sweep_signals(year: int, for_date: str):
    """Load date-stamped sweep signals. Returns (data, is_today)."""
    from huggingface_hub import hf_hub_download
    date_tag = for_date.replace("-", "")

    # Try today's file
    try:
        path = hf_hub_download(repo_id=HF_OUTPUT_REPO,
                               filename=f"signals_{year}_{date_tag}.json",
                               repo_type="dataset", force_download=True)
        with open(path) as f:
            return json.load(f), True
    except Exception:
        pass

    # Fall back to yesterday's file
    try:
        from datetime import date as _date, timedelta as _td
        yesterday = (_date.fromisoformat(for_date) - _td(days=1)).strftime("%Y%m%d")
        path = hf_hub_download(repo_id=HF_OUTPUT_REPO,
                               filename=f"signals_{year}_{yesterday}.json",
                               repo_type="dataset", force_download=True)
        with open(path) as f:
            return json.load(f), False
    except Exception:
        pass

    return None, False


# ── GitHub Actions helpers ────────────────────────────────────────────────────

def trigger_github_training(start_year: int, force_refresh: bool = False,
                             sweep_mode: str = "") -> bool:
    pat = os.getenv("GITHUB_PAT")
    if not pat:
        st.error("❌ GITHUB_PAT secret not found in HF Space secrets.")
        return False
    url = (f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/actions/workflows/"
           f"{GITHUB_WORKFLOW}/dispatches")
    payload = {
        "ref": "main",
        "inputs": {
            "start_year":    str(start_year),
            "sweep_mode":    sweep_mode,
            "force_refresh": str(force_refresh).lower(),
        }
    }
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        r = req.post(url, json=payload, headers=headers, timeout=15)
        return r.status_code == 204
    except Exception as e:
        st.error(f"❌ Failed to trigger GitHub Actions: {e}")
        return False


def get_latest_workflow_run() -> dict:
    pat = os.getenv("GITHUB_PAT")
    if not pat:
        return {}
    url = (f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/actions/workflows/"
           f"{GITHUB_WORKFLOW}/runs?per_page=1")
    headers = {"Authorization": f"Bearer {pat}",
               "Accept": "application/vnd.github+json"}
    try:
        r = req.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            runs = r.json().get("workflow_runs", [])
            return runs[0] if runs else {}
    except Exception:
        pass
    return {}


# ── Consensus scoring ─────────────────────────────────────────────────────────

def compute_consensus(sweep_data: dict) -> dict:
    """
    Weighted score per ETF across all available sweep years.
    Formula: 40% Ann.Return + 20% Z-Score + 20% Sharpe + 20% (-MaxDD)
    All metrics min-max normalised across years before weighting.
    """
    etf_scores = {}   # etf → {score, years, signals}
    per_year   = []

    for year, sig in sweep_data.items():
        signal     = sig['next_signal']
        ann_ret    = sig.get('ann_return', 0.0)
        z_score    = sig.get('conviction_z', 0.0)
        sharpe     = sig.get('sharpe', 0.0)
        max_dd     = sig.get('max_dd', 0.0)
        lookback   = sig.get('lookback_days', '?')
        per_year.append({
            'year':       year,
            'signal':     signal,
            'ann_return': ann_ret,
            'z_score':    z_score,
            'sharpe':     sharpe,
            'max_dd':     max_dd,
            'lookback':   lookback,
            'conviction': sig.get('conviction_label', '?'),
        })

    if not per_year:
        return {}

    df = pd.DataFrame(per_year)

    # Min-max normalise each metric
    def minmax(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    df['n_return'] = minmax(df['ann_return'])
    df['n_z']      = minmax(df['z_score'])
    df['n_sharpe'] = minmax(df['sharpe'])
    df['n_negdd']  = minmax(-df['max_dd'])   # higher = less drawdown = better

    df['wtd_score'] = (0.40 * df['n_return'] +
                       0.20 * df['n_z']      +
                       0.20 * df['n_sharpe'] +
                       0.20 * df['n_negdd'])

    # Aggregate by ETF across years
    etf_agg = {}
    for _, row in df.iterrows():
        etf = row['signal']
        if etf not in etf_agg:
            etf_agg[etf] = {'years': [], 'scores': [], 'returns': [],
                            'z_scores': [], 'sharpes': [], 'max_dds': []}
        etf_agg[etf]['years'].append(row['year'])
        etf_agg[etf]['scores'].append(row['wtd_score'])
        etf_agg[etf]['returns'].append(row['ann_return'])
        etf_agg[etf]['z_scores'].append(row['z_score'])
        etf_agg[etf]['sharpes'].append(row['sharpe'])
        etf_agg[etf]['max_dds'].append(row['max_dd'])

    etf_summary = {}
    total_score = sum(sum(v['scores']) for v in etf_agg.values()) + 1e-9
    for etf, v in etf_agg.items():
        cum_score = sum(v['scores'])
        etf_summary[etf] = {
            'cum_score':  round(cum_score, 4),
            'score_share': round(cum_score / total_score, 3),
            'n_years':    len(v['years']),
            'years':      v['years'],
            'avg_return': round(np.mean(v['returns']), 4),
            'avg_z':      round(np.mean(v['z_scores']), 3),
            'avg_sharpe': round(np.mean(v['sharpes']), 3),
            'avg_max_dd': round(np.mean(v['max_dds']), 4),
        }

    winner = max(etf_summary, key=lambda e: etf_summary[e]['cum_score'])
    return {
        'winner':      winner,
        'etf_summary': etf_summary,
        'per_year':    df.to_dict('records'),
        'n_years':     len(per_year),
    }


# ── Load outputs at top (needed for sidebar) ──────────────────────────────────
with st.spinner("📦 Loading outputs..."):
    outputs, load_err = load_model_outputs()
    signals, sig_err  = load_signals()
    meta              = load_training_meta()

_trained_start_yr = int(signals.get('start_year', 2016)) if signals else 2016

latest_run  = get_latest_workflow_run()
is_training = latest_run.get("status") in ("queued", "in_progress")
run_started = latest_run.get("created_at", "")[:16].replace("T", " ") if latest_run else ""

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"🕒 **EST:** {get_est_time().strftime('%H:%M:%S')}")
    st.divider()

    st.subheader("📅 Training Period")
    start_yr = st.slider("Start Year", 2008, 2024, _trained_start_yr,
                         help="Single-year run uses this start year")

    st.divider()
    st.subheader("💰 Transaction Cost")
    fee_bps = st.slider("Transaction Fee (bps)", 0, 100, 15)

    st.divider()
    st.subheader("🛑 Risk Controls")
    stop_loss_pct = st.slider(
        "Stop Loss (2-day cumulative)", min_value=-20, max_value=-8,
        value=-12, step=1, format="%d%%") / 100.0
    z_reentry = st.slider("Re-entry Conviction (σ)", 0.75, 1.50, 1.00, 0.05, format="%.2f")
    z_min_entry = st.slider("Min Entry Conviction (σ)", 0.0, 1.5, 0.5, 0.05, format="%.2f")

    st.divider()
    st.subheader("📥 Dataset")
    force_refresh = st.checkbox("Force Dataset Refresh", value=False)
    refresh_only_button = st.button("🔄 Refresh Dataset Only",
                                    type="secondary", use_container_width=True)

    st.divider()
    run_button = st.button("🚀 Run TFT Model", type="primary",
                           use_container_width=True, disabled=is_training,
                           help="Trains for selected start year (~1.5hrs)")
    if is_training:
        st.warning(f"⏳ Training in progress (started {run_started} UTC)")

    st.divider()
    st.caption("🤖 Split: 80/10/10 · Trained on GitHub Actions")
    if signals:
        st.caption(f"📅 Current: start_year={signals.get('start_year','?')} · "
                   f"trained {signals.get('run_timestamp_utc','')[:10]}")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🤖 P2-ETF-PREDICTOR")
st.caption("Temporal Fusion Transformer — Fixed Income ETF Rotation")

# ── Handle refresh dataset only ───────────────────────────────────────────────
if refresh_only_button:
    with st.status("📡 Refreshing dataset...", expanded=True):
        etf_data   = fetch_etf_data(["TLT", "TBT", "VNQ", "SLV", "GLD", "AGG", "SPY"])
        macro_data = fetch_macro_data_robust()
        if not etf_data.empty and not macro_data.empty:
            token = os.getenv("HF_TOKEN")
            if token:
                updated_df = smart_update_hf_dataset(
                    pd.concat([etf_data, macro_data], axis=1), token)
                st.success("✅ Done!")
            else:
                st.error("❌ HF_TOKEN not found.")
        else:
            st.error("❌ Data fetch failed")
    st.stop()

# ── Handle run button ─────────────────────────────────────────────────────────
if run_button:
    with st.spinner(f"🚀 Triggering training for start_year={start_yr}..."):
        ok = trigger_github_training(start_year=start_yr,
                                     force_refresh=force_refresh, sweep_mode="")
    if ok:
        st.success(f"✅ Training triggered for **start_year={start_yr}**! "
                   f"Results will appear in ~90 minutes.")
        time.sleep(2)
        st.rerun()

# ── Training banner ───────────────────────────────────────────────────────────
if is_training:
    st.warning(f"⏳ **Training in progress** (started {run_started} UTC) — "
               f"showing previous results.", icon="🔄")

if signals and signals.get('start_year') and \
        int(signals.get('start_year')) != start_yr and not is_training:
    st.info(f"ℹ️ Showing results for **start_year={signals.get('start_year')}**. "
            f"Click **🚀 Run TFT Model** to train for **{start_yr}**.")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Single-Year Results", "🔄 Multi-Year Consensus Sweep"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single-Year Results
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    if not outputs:
        st.error(f"❌ No model outputs available: {load_err}")
        st.info("👈 Click **🚀 Run TFT Model** in the sidebar to trigger training.")
        st.stop()

    proba          = outputs['proba']
    daily_ret_test = outputs['daily_ret_test']
    y_fwd_test     = outputs['y_fwd_test']
    spy_ret_test   = outputs['spy_ret_test']
    agg_ret_test   = outputs['agg_ret_test']
    test_dates     = pd.DatetimeIndex(outputs['test_dates'])
    target_etfs    = list(outputs['target_etfs'])
    sofr           = float(outputs['sofr'][0])
    etf_names      = [e.replace('_Ret', '') for e in target_etfs]

    if signals:
        st.info(
            f"📅 **Trained from:** {signals.get('start_year','?')} · "
            f"**Data:** {signals['data_start']} → {signals['data_end']} | "
            f"**OOS Test:** {signals['test_start']} → {signals['test_end']} "
            f"({signals['n_test_days']} days) | "
            f"🕒 Trained: {signals['run_timestamp_utc'][:10]}"
        )
    if meta:
        st.caption(f"📐 Lookback: {meta['lookback_days']}d · "
                   f"Features: {meta['n_features']} · Split: {meta['split']} · "
                   f"Targets: {', '.join(etf_names)}")

    # ── Strategy replay ───────────────────────────────────────────────────────
    (strat_rets, audit_trail, next_signal, next_trading_date,
     conviction_zscore, conviction_label, all_etf_scores) = execute_strategy(
        proba, y_fwd_test, test_dates, target_etfs,
        fee_bps, stop_loss_pct=stop_loss_pct, z_reentry=z_reentry,
        sofr=sofr, z_min_entry=z_min_entry, daily_ret_override=daily_ret_test
    )
    metrics = calculate_metrics(strat_rets, sofr)

    if meta and 'accuracy_per_etf' in meta:
        st.info(f"🎯 **Binary Accuracy per ETF:** {meta['accuracy_per_etf']} | "
                f"Random baseline: 50.0%")

    # ── Next trading day banner ───────────────────────────────────────────────
    st.divider()
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#00d1b2,#00a896);
                padding:25px;border-radius:15px;text-align:center;
                box-shadow:0 8px 16px rgba(0,0,0,0.3);margin:20px 0;">
        <h1 style="color:white;font-size:48px;margin:0 0 10px 0;font-weight:bold;">
            🎯 NEXT TRADING DAY
        </h1>
        <h2 style="color:white;font-size:36px;margin:0;font-weight:bold;">
            {next_trading_date} → {next_signal}
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # ── Signal conviction ─────────────────────────────────────────────────────
    conviction_colors = {"Very High": "#00b894", "High": "#00cec9",
                         "Moderate": "#fdcb6e", "Low": "#d63031"}
    conviction_icons  = {"Very High": "🟢", "High": "🟢",
                         "Moderate": "🟡", "Low": "🔴"}
    conv_color  = conviction_colors.get(conviction_label, "#888")
    conv_dot    = conviction_icons.get(conviction_label, "⚪")
    z_clipped   = max(-3.0, min(3.0, conviction_zscore))
    bar_pct     = int((z_clipped + 3) / 6 * 100)
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
        bar_w      = int(score / max_score * 100)
        is_winner  = (name == next_signal)
        is_last    = (i == len(sorted_pairs) - 1)
        name_style = "font-weight:700;color:#00897b;" if is_winner else "color:#444;"
        bar_color  = conv_color if is_winner else "#b2dfdb" if score > max_score * 0.5 else "#e0e0e0"
        star       = " ★" if is_winner else ""
        bottom_r   = "0 0 12px 12px" if is_last else "0"
        border_bot = "border-bottom:1px solid #f0f0f0;" if not is_last else ""
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #ddd;border-top:none;
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

    # ── Metrics ───────────────────────────────────────────────────────────────
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

    # ── Equity curve ──────────────────────────────────────────────────────────
    st.subheader("📈 Out-of-Sample Equity Curve (with Benchmarks)")
    plot_dates = test_dates[:len(metrics['cum_returns'])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_dates, y=metrics['cum_returns'], mode='lines',
                             name='TFT Strategy', line=dict(color='#00d1b2', width=3),
                             fill='tozeroy', fillcolor='rgba(0,209,178,0.1)'))
    fig.add_trace(go.Scatter(x=plot_dates, y=metrics['cum_max'], mode='lines',
                             name='High Water Mark',
                             line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash')))
    spy_m = calculate_benchmark_metrics(
        np.nan_to_num(spy_ret_test[:len(strat_rets)], nan=0.0), sofr)
    agg_m = calculate_benchmark_metrics(
        np.nan_to_num(agg_ret_test[:len(strat_rets)], nan=0.0), sofr)
    fig.add_trace(go.Scatter(x=plot_dates, y=spy_m['cum_returns'], mode='lines',
                             name='SPY', line=dict(color='#ff4b4b', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=plot_dates, y=agg_m['cum_returns'], mode='lines',
                             name='AGG', line=dict(color='#ffa500', width=2, dash='dot')))
    fig.update_layout(template="plotly_dark", height=450, hovermode='x unified',
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                      xaxis_title="Date", yaxis_title="Cumulative Return")
    st.plotly_chart(fig, use_container_width=True)

    # ── Audit trail ───────────────────────────────────────────────────────────
    st.subheader("📋 Last 20 Days Audit Trail")
    audit_df = pd.DataFrame(audit_trail).tail(20)
    if not audit_df.empty:
        display_cols = [c for c in ['Date','Signal','Top_Pick','Conviction_Z',
                                    'Net_Return','Stop_Active','Rotated']
                        if c in audit_df.columns]
        audit_df = audit_df[display_cols]
        styled = (audit_df.style
                  .applymap(lambda v: 'color:#00ff00;font-weight:bold'
                            if v > 0 else 'color:#ff4b4b;font-weight:bold',
                            subset=['Net_Return'])
                  .format({'Net_Return': '{:.2%}', 'Conviction_Z': '{:.2f}'})
                  .set_properties(**{'font-size': '16px', 'text-align': 'center'})
                  .set_table_styles([
                      {'selector': 'th', 'props': [('font-size','17px'),
                                                   ('font-weight','bold'),
                                                   ('text-align','center')]},
                      {'selector': 'td', 'props': [('padding','10px')]}
                  ]))
        st.dataframe(styled, use_container_width=True, height=650)

    # ── Methodology ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📖 Methodology & Model Notes")
    lookback_display = meta['lookback_days'] if meta else "auto"
    rf_label_display = signals['rf_label'] if signals else "4.5% fallback"
    trained_start    = signals.get('start_year', start_yr) if signals else start_yr
    st.markdown(f"""
    <div style="background:#1a1a2e;border:1px solid #2d2d4e;border-radius:12px;
                padding:28px 32px;color:#e0e0e0;font-size:14px;line-height:1.8;">
    <h4 style="color:#00d1b2;margin-top:0;">🏗️ Architecture — 7 Binary TFTs</h4>
    <p>One binary TFT per ETF: <em>"Will this ETF beat 3M T-Bill over 5 days?"</em></p>
    <h4 style="color:#00d1b2;margin-top:16px;">📊 Training</h4>
    <ul>
      <li><b>Period:</b> {trained_start} → present · <b>Split:</b> 80/10/10 chronological</li>
      <li><b>Lookback:</b> auto-optimised → <b>{lookback_display} days</b></li>
      <li><b>Risk-free rate:</b> {sofr*100:.2f}% ({rf_label_display})</li>
    </ul>
    <h4 style="color:#00d1b2;margin-top:16px;">⚙️ Live Strategy</h4>
    <ul>
      <li>Conviction gate ≥ {z_min_entry}σ · Stop-loss {stop_loss_pct*100:.0f}% ·
          Re-entry {z_reentry}σ · Fee {fee_bps}bps</li>
    </ul>
    <h4 style="color:#00d1b2;margin-top:16px;">⚠️ Disclaimer</h4>
    <p>Research only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Multi-Year Consensus Sweep
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔄 Multi-Year Consensus Sweep")
    st.markdown(
        "Runs the TFT model across **5 start years** and aggregates signals into a "
        "consensus vote. Cached years load instantly — only untrained years trigger "
        "new GitHub Actions jobs (in parallel).\n\n"
        f"**Sweep years:** {', '.join(str(y) for y in SWEEP_YEARS)} &nbsp;·&nbsp; "
        "**Score:** 40% Return · 20% Z · 20% Sharpe · 20% (–MaxDD)"
    )

    # ── Date-aware sweep cache loading ───────────────────────────────────────
    today_str   = str(_today_est())
    sweep_cache = {}       # today's results
    prev_cache  = {}       # yesterday's results (fallback)
    stale_years = []       # years where only yesterday's data exists
    missing_years = []     # years with no data at all

    for yr in SWEEP_YEARS:
        data, is_today = load_sweep_signals(yr, today_str)
        if data and is_today:
            sweep_cache[yr] = data
        elif data and not is_today:
            prev_cache[yr]  = data
            stale_years.append(yr)
        else:
            missing_years.append(yr)

    # Display cache = today's where available, yesterday's as fallback
    display_cache = {**prev_cache, **sweep_cache}  # today overrides yesterday
    years_needing_run = [yr for yr in SWEEP_YEARS if yr not in sweep_cache]
    sweep_complete = len(sweep_cache) == len(SWEEP_YEARS)

    # ── Stale data warning banner ─────────────────────────────────────────────
    if stale_years and not sweep_complete:
        from datetime import date as _d, timedelta as _td
        yesterday = str(_d.fromisoformat(today_str) - _td(days=1))
        st.warning(
            f"⚠️ Showing **yesterday's results** ({yesterday}) for: "
            f"{', '.join(str(y) for y in stale_years)}. "
            f"Today's sweep has not run yet — auto-runs at 8pm EST or click below.",
            icon="📅"
        )
    if is_training and not sweep_complete:
        st.info(
            f"⏳ **Training in progress** — {len(sweep_cache)}/{len(SWEEP_YEARS)} years "
            f"complete today. Showing previous results where available.", icon="🔄"
        )

    # ── Status grid ──────────────────────────────────────────────────────────
    cols = st.columns(len(SWEEP_YEARS))
    for i, yr in enumerate(SWEEP_YEARS):
        with cols[i]:
            if yr in sweep_cache:
                sig = sweep_cache[yr]['next_signal']
                st.success(f"**{yr}**\n✅ {sig}")
            elif yr in prev_cache:
                sig = prev_cache[yr]['next_signal']
                st.warning(f"**{yr}**\n📅 {sig}")
            else:
                st.error(f"**{yr}**\n⏳ Not run")

    st.caption("✅ = today's result  ·  📅 = yesterday's result (stale)  ·  ⏳ = not yet run")
    st.divider()

    # ── Sweep button ──────────────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        sweep_btn = st.button(
            "🚀 Run Consensus Sweep",
            type="primary",
            use_container_width=True,
            disabled=(is_training or sweep_complete),
            help="Only runs years missing today's fresh results"
        )
    with col_info:
        if sweep_complete:
            st.success(f"✅ Today's sweep complete ({today_str}) — {len(SWEEP_YEARS)}/{len(SWEEP_YEARS)} years fresh")
        elif is_training:
            st.warning(f"⏳ Training in progress... ({len(sweep_cache)}/{len(SWEEP_YEARS)} fresh today)")
        else:
            st.info(
                f"**{len(sweep_cache)}/{len(SWEEP_YEARS)}** years fresh for today ({today_str}).  \n"
                f"Will trigger **{len(years_needing_run)}** jobs: "
                f"{', '.join(str(y) for y in years_needing_run)}"
            )

    if sweep_btn and years_needing_run:
        sweep_mode_str = ",".join(str(y) for y in years_needing_run)
        with st.spinner(f"🚀 Triggering parallel training for: {sweep_mode_str}..."):
            ok = trigger_github_training(
                start_year=years_needing_run[0],
                sweep_mode=sweep_mode_str,
                force_refresh=False
            )
        if ok:
            st.success(
                f"✅ Triggered **{len(years_needing_run)}** parallel jobs for: {sweep_mode_str}. "
                f"Each takes ~90 mins. Refresh this tab when complete."
            )
            time.sleep(2)
            st.rerun()
        else:
            st.error("❌ Failed to trigger GitHub Actions sweep.")

    # ── Consensus results ─────────────────────────────────────────────────────
    if len(display_cache) == 0:
        st.info("👆 Click **🚀 Run Consensus Sweep** to train all years.")
        st.stop()

    consensus = compute_consensus(display_cache)
    if not consensus:
        st.warning("⚠️ Could not compute consensus.")
        st.stop()

    winner      = consensus['winner']
    w_info      = consensus['etf_summary'][winner]
    win_color   = ETF_COLORS.get(winner, "#00d1b2")
    score_share = w_info['score_share'] * 100
    n_cached    = len(display_cache)

    # ── Consensus winner banner ───────────────────────────────────────────────
    split_signal = w_info['score_share'] < 0.4
    signal_label = "⚠️ Split Signal" if split_signal else "✅ Clear Signal"
    signal_note  = f"Score share {score_share:.0f}% · {w_info['n_years']}/{len(SWEEP_YEARS)} years · avg score {w_info['cum_score']:.2f}"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#2d3436,#1e272e);
                border:2px solid {win_color};border-radius:16px;
                padding:32px;text-align:center;margin:20px 0;
                box-shadow:0 8px 24px rgba(0,0,0,0.4);">
      <div style="font-size:11px;letter-spacing:3px;color:#aaa;margin-bottom:12px;">
        WEIGHTED CONSENSUS · TFT · {n_cached} START YEARS · {today_str}
      </div>
      <div style="font-size:72px;font-weight:900;color:{win_color};
                  text-shadow:0 0 30px {win_color}88;letter-spacing:2px;">
        {winner}
      </div>
      <div style="font-size:14px;color:#ccc;margin-top:8px;">{signal_label} · {signal_note}</div>
      <div style="display:flex;justify-content:center;gap:32px;margin-top:20px;flex-wrap:wrap;">
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg Return</div>
          <div style="font-size:20px;font-weight:700;color:{'#00b894' if w_info['avg_return']>0 else '#d63031'};">
            {w_info['avg_return']*100:.1f}%</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg Z</div>
          <div style="font-size:20px;font-weight:700;color:#74b9ff;">{w_info['avg_z']:.2f}σ</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg Sharpe</div>
          <div style="font-size:20px;font-weight:700;color:#a29bfe;">{w_info['avg_sharpe']:.2f}</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg MaxDD</div>
          <div style="font-size:20px;font-weight:700;color:#fd79a8;">{w_info['avg_max_dd']*100:.1f}%</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Also-ranked line
    others = sorted([(e, v) for e, v in consensus['etf_summary'].items() if e != winner],
                    key=lambda x: -x[1]['cum_score'])
    also_parts = []
    for etf, v in others:
        col = ETF_COLORS.get(etf, "#888")
        n_yrs = v['n_years']
        also_parts.append(
            f'<span style="color:{col};font-weight:600;">{etf}</span> '
            f'<span style="color:#aaa;">(score {v["cum_score"]:.2f} · {n_yrs} yr)</span>'
        )
    st.markdown(
        '<div style="text-align:center;margin-bottom:16px;font-size:13px;">'
        'Also ranked: ' + " &nbsp;|&nbsp; ".join(also_parts) + '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Weighted Score per ETF** (40% Return · 20% Z · 20% Sharpe · 20% –MaxDD)")
        etf_sum = consensus['etf_summary']
        sorted_etfs = sorted(etf_sum.keys(), key=lambda e: -etf_sum[e]['cum_score'])
        bar_colors  = [ETF_COLORS.get(e, "#888") for e in sorted_etfs]
        bar_vals    = [etf_sum[e]['cum_score'] for e in sorted_etfs]
        bar_labels  = [
            f"{etf_sum[e]['n_years']} yr · {etf_sum[e]['score_share']*100:.0f}%<br>score {etf_sum[e]['cum_score']:.2f}"
            for e in sorted_etfs
        ]
        fig_bar = go.Figure(go.Bar(
            x=sorted_etfs, y=bar_vals,
            marker_color=bar_colors,
            text=bar_labels, textposition='outside',
        ))
        fig_bar.update_layout(
            template="plotly_dark", height=380,
            yaxis_title="Cumulative Weighted Score",
            showlegend=False, margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        st.markdown("**Conviction Z-Score by Start Year**")
        per_year = consensus['per_year']
        fig_scatter = go.Figure()
        for row in per_year:
            etf = row['signal']
            col = ETF_COLORS.get(etf, "#888")
            fig_scatter.add_trace(go.Scatter(
                x=[row['year']], y=[row['z_score']],
                mode='markers+text',
                marker=dict(size=18, color=col, line=dict(color='white', width=1)),
                text=[etf], textposition='top center',
                name=etf,
                showlegend=False,
                hovertemplate=f"<b>{etf}</b><br>Year: {row['year']}<br>"
                              f"Z: {row['z_score']:.2f}σ<br>"
                              f"Return: {row['ann_return']*100:.1f}%<extra></extra>"
            ))
        # Add neutral line
        fig_scatter.add_hline(y=0, line_dash="dot",
                              line_color="rgba(255,255,255,0.3)",
                              annotation_text="Neutral")
        fig_scatter.update_layout(
            template="plotly_dark", height=380,
            xaxis_title="Start Year", yaxis_title="Z-Score (σ)",
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Per-year breakdown table ──────────────────────────────────────────────
    st.subheader("📋 Full Per-Year Breakdown")
    st.caption(
        "**Wtd Score** = 40% Ann. Return + 20% Z-Score + 20% Sharpe + 20% (–Max DD), "
        "each metric min-max normalised across all years. "
        "⚡ = loaded from cache (no retraining)."
    )

    table_rows = []
    for row in sorted(consensus['per_year'], key=lambda r: r['year']):
        etf    = row['signal']
        col    = ETF_COLORS.get(etf, "#888")
        _in_today = row['year'] in sweep_cache
        table_rows.append({
            'Start Year':   row['year'],
            'Signal':       etf,
            'Wtd Score':    round(row['wtd_score'], 3),
            'Conviction':   row['conviction'],
            'Z-Score':      f"{row['z_score']:.2f}σ",
            'Ann. Return':  f"{row['ann_return']*100:.2f}%",
            'Sharpe':       f"{row['sharpe']:.2f}",
            'Max Drawdown': f"{row['max_dd']*100:.2f}%",
            'Lookback':     f"{row['lookback']}d",
            'Cache':        "✅ Today" if row['year'] in sweep_cache else "📅 Prev",
        })

    tbl_df = pd.DataFrame(table_rows)

    def style_signal(val):
        col = ETF_COLORS.get(val, "#888")
        return f"background-color:{col}22;color:{col};font-weight:700;"

    def style_return(val):
        try:
            v = float(val.replace('%', ''))
            return 'color:#00b894;font-weight:600' if v > 0 else 'color:#d63031;font-weight:600'
        except Exception:
            return ''

    def style_wtd(val):
        try:
            v = float(val)
            intensity = min(int(v * 200), 200)
            return f'color:#00d1b2;font-weight:700'
        except Exception:
            return ''

    styled_tbl = (tbl_df.style
                  .applymap(style_signal, subset=['Signal'])
                  .applymap(style_return, subset=['Ann. Return'])
                  .applymap(style_wtd, subset=['Wtd Score'])
                  .set_properties(**{'text-align': 'center', 'font-size': '15px'})
                  .set_table_styles([
                      {'selector': 'th', 'props': [('font-size', '15px'),
                                                   ('font-weight', 'bold'),
                                                   ('text-align', 'center'),
                                                   ('background-color', '#1a1a2e'),
                                                   ('color', '#00d1b2')]},
                      {'selector': 'td', 'props': [('padding', '10px')]}
                  ]))
    st.dataframe(styled_tbl, use_container_width=True, height=300)

    # ── How to read ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📖 How to Read These Results")
    st.markdown("""
**Why does the signal change by start year?**
Each start year defines the *training regime* the model learns from.
A model trained from 2008 has seen the GFC and multiple rate cycles.
A model trained from 2019 focuses on post-COVID dynamics.
The consensus aggregates all regime perspectives into one vote.

**How is the winner chosen?**
Each year's signal scores points based on its backtested performance (Ann. Return, Sharpe, Z-Score, MaxDD).
Scores are min-max normalised so no single metric dominates.
The ETF with the highest cumulative weighted score across all years wins.

**What does ⚡ mean?**
That year's model output was loaded from cache — no retraining was needed.
Sweep cache is preserved through the daily midnight cleanup.

**Split Signal warning**
If the winning ETF has a score share below 40%, signals are fragmented across years.
Treat the result with caution — no single ETF dominates across regimes.
""")
