"""
P2-ETF-PREDICTOR — TFT Edition
================================
Two tabs: Single‑Year Results (loads pre‑computed sweep file for chosen year)
          Multi‑Year Consensus Sweep (aggregates all years)
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
from config import OPTION_A_ETFS, OPTION_B_ETFS

st.set_page_config(page_title="P2-ETF-Predictor | TFT", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
HF_OUTPUT_REPO         = "P2SAMAPA/p2-etf-tft-outputs"
GITHUB_REPO            = "P2SAMAPA/P2-ETF-TFT-PREDICTOR-HF-DATASET"
GITHUB_WORKFLOW        = "train_and_push.yml"
GITHUB_SWEEP_WORKFLOW  = "consensus_sweep.yml"
GITHUB_API_BASE        = "https://api.github.com"
SWEEP_YEARS            = list(range(2008, 2026))   # 2008–2025

# Colour maps
FI_COLOURS = {
    "TLT": "#4e79a7", "VCIT": "#f28e2b", "LQD": "#59a14f",
    "HYG": "#e15759", "VNQ": "#76b7b2", "SLV": "#edc948",
    "GLD": "#b07aa1",
}
EQ_COLOURS = {
    "QQQ": "#4e79a7", "XLK": "#f28e2b", "XLF": "#59a14f",
    "XLE": "#e15759", "XLV": "#76b7b2", "XLI": "#edc948",
    "XLY": "#b07aa1", "XLP": "#ff9da7", "XLU": "#9c755f",
    "GDX": "#86b875", "XME": "#bab0ac",
}

def get_colour_map(option: str) -> dict:
    return FI_COLOURS if option == 'a' else EQ_COLOURS


# ── Helper functions ─────────────────────────────────────────────────────────
def _today_est():
    from datetime import datetime as _dt, timezone, timedelta
    return (_dt.now(timezone.utc) - timedelta(hours=5)).date()


def _load_sweep_file(option: str, year: int) -> dict:
    """Load the most recent sweep file for a given option and year."""
    try:
        from huggingface_hub import hf_hub_download
        repo_id = HF_OUTPUT_REPO
        # We'll list files to find the latest date for that year
        from huggingface_hub import HfApi
        api = HfApi()
        prefix = f"sweep/option_{option}/signals_{year}_"
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        matches = [f for f in files if f.startswith(prefix) and f.endswith(".json")]
        if not matches:
            return None
        # Sort by date (the part after the underscore)
        matches.sort(reverse=True)
        latest = matches[0]
        path = hf_hub_download(repo_id=repo_id, filename=latest,
                               repo_type="dataset", force_download=True)
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_sweep_cache_for_date(option: str, for_date: str) -> dict:
    """Load all sweep files for a given date (used by consensus tab)."""
    cache = {}
    try:
        from huggingface_hub import hf_hub_download, HfApi
        repo_id = HF_OUTPUT_REPO
        api = HfApi()
        prefix = f"sweep/option_{option}/signals_"
        files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        for f in files:
            if f.startswith(prefix) and f.endswith(".json"):
                parts = f.replace(".json", "").split("_")
                # parts: ['sweep/option_{option}/signals', year, date]
                if len(parts) == 4:
                    yr = int(parts[2])
                    dt = parts[3]
                    if dt == for_date:
                        path = hf_hub_download(repo_id=repo_id, filename=f,
                                               repo_type="dataset", force_download=True)
                        with open(path) as fh:
                            cache[yr] = json.load(fh)
    except Exception as e:
        st.warning(f"Could not load sweep results: {e}")
    return cache


def load_sweep_cache_any(option: str) -> tuple:
    """Load most recent full set of sweep files (all years)."""
    found, best_date = {}, None
    try:
        from huggingface_hub import HfApi
        repo_id = HF_OUTPUT_REPO
        api = HfApi()
        prefix = f"sweep/option_{option}/signals_"
        files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        dates = set()
        for f in files:
            if f.startswith(prefix) and f.endswith(".json"):
                parts = f.replace(".json", "").split("_")
                if len(parts) == 4:
                    dates.add(parts[3])
        if dates:
            best_date = sorted(dates)[-1]   # latest date string
            found = load_sweep_cache_for_date(option, best_date)
    except Exception:
        pass
    return found, best_date


def compute_consensus(sweep_data: dict) -> dict:
    """Weighted score per ETF across all available sweep years."""
    per_year = []
    for year, sig in sweep_data.items():
        signal     = sig.get('next_signal', '?')
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
    def minmax(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)
    df['n_return'] = minmax(df['ann_return'])
    df['n_z']      = minmax(df['z_score'])
    df['n_sharpe'] = minmax(df['sharpe'])
    df['n_negdd']  = minmax(-df['max_dd'])
    df['wtd_score'] = (0.40 * df['n_return'] +
                       0.20 * df['n_z']      +
                       0.20 * df['n_sharpe'] +
                       0.20 * df['n_negdd'])
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
            'cum_score':   round(cum_score, 4),
            'score_share': round(cum_score / total_score, 3),
            'n_years':     len(v['years']),
            'years':       v['years'],
            'avg_return':  round(np.mean(v['returns']), 4),
            'avg_z':       round(np.mean(v['z_scores']), 3),
            'avg_sharpe':  round(np.mean(v['sharpes']), 3),
            'avg_max_dd':  round(np.mean(v['max_dds']), 4),
        }
    winner = max(etf_summary, key=lambda e: etf_summary[e]['cum_score'])
    return {
        'winner':      winner,
        'etf_summary': etf_summary,
        'per_year':    df.to_dict('records'),
        'n_years':     len(per_year),
    }


# ── GitHub Actions helpers ───────────────────────────────────────────────────
def trigger_github_training(start_year: int, force_refresh: bool = False) -> bool:
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


def trigger_consensus_sweep(force_refresh: bool = False) -> bool:
    pat = os.getenv("GITHUB_PAT")
    if not pat:
        st.error("❌ GITHUB_PAT secret not found in HF Space secrets.")
        return False
    url = (f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/actions/workflows/"
           f"{GITHUB_SWEEP_WORKFLOW}/dispatches")
    payload = {
        "ref": "main",
        "inputs": {
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
        st.error(f"❌ Failed to trigger consensus sweep: {e}")
        return False


def get_latest_workflow_run() -> dict:
    pat = os.getenv("GITHUB_PAT")
    if not pat:
        return {}
    headers = {"Authorization": f"Bearer {pat}",
               "Accept": "application/vnd.github+json"}
    candidates = []
    for wf in [GITHUB_WORKFLOW, GITHUB_SWEEP_WORKFLOW]:
        url = (f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/actions/workflows/"
               f"{wf}/runs?per_page=1")
        try:
            r = req.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                runs = r.json().get("workflow_runs", [])
                if runs:
                    candidates.append(runs[0])
        except Exception:
            pass
    if not candidates:
        return {}
    return max(candidates, key=lambda x: x.get("created_at", ""))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"🕒 **EST:** {get_est_time().strftime('%H:%M:%S')}")

    option_choice = st.radio(
        "Select Universe",
        ["Option A (FI/Commodities)", "Option B (Equity Sectors)"],
        index=0,
        horizontal=True
    )
    option = 'a' if "Option A" in option_choice else 'b'
    colour_map = get_colour_map(option)

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
    st.caption("🤖 Split: 80/10/10 · Trained on GitHub Actions")

    # Display training status
    latest_run = get_latest_workflow_run()
    is_training = latest_run.get("status") in ("queued", "in_progress")
    run_started = latest_run.get("created_at", "")[:16].replace("T", " ") if latest_run else ""
    if is_training:
        st.warning(f"⏳ Training in progress (started {run_started} UTC)")


# ── Handle refresh dataset only ───────────────────────────────────────────────
if refresh_only_button:
    with st.status("📡 Refreshing dataset...", expanded=True):
        from config import ALL_TICKERS
        etf_data   = fetch_etf_data(ALL_TICKERS)
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


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Single-Year Results", "🔄 Multi-Year Consensus Sweep"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single-Year Results (loads pre‑computed sweep file for chosen year)
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Single‑Year Results — {option_choice}")

    # Year selector
    year_options = list(range(2008, 2026))
    selected_year = st.selectbox("Select start year to view:", year_options, index=len(year_options)-1)

    # Load the sweep file for that year and option
    year_data = _load_sweep_file(option, selected_year)
    if not year_data:
        st.info(f"No sweep results found for year {selected_year}. Run the consensus sweep first.")
    else:
        # Extract data
        signal = year_data.get('next_signal', '?')
        ann_return = year_data.get('ann_return', 0.0)
        z_score = year_data.get('conviction_z', 0.0)
        sharpe = year_data.get('sharpe', 0.0)
        max_dd = year_data.get('max_dd', 0.0)
        conviction_label = year_data.get('conviction_label', '?')
        lookback = year_data.get('lookback_days', '?')
        sweep_date = year_data.get('sweep_date', '?')
        if isinstance(sweep_date, str):
            sweep_date = sweep_date[:4] + "-" + sweep_date[4:6] + "-" + sweep_date[6:]

        # Display hero banner
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                    border:2px solid #00d1b2;border-radius:16px;
                    padding:32px;text-align:center;margin:16px 0;
                    box-shadow:0 8px 24px rgba(0,0,0,0.4);">
          <div style="font-size:11px;letter-spacing:3px;color:#aaa;margin-bottom:12px;">
            TFT MODEL · START YEAR {selected_year} · RESULTS FROM {sweep_date}
          </div>
          <div style="font-size:72px;font-weight:900;color:#00d1b2;">
            {signal}
          </div>
          <div style="font-size:14px;color:#ccc;margin-top:8px;">
            Conviction: {conviction_label} · Lookback: {lookback}d
          </div>
          <div style="display:flex;justify-content:center;gap:32px;margin-top:20px;flex-wrap:wrap;">
            <div style="text-align:center;">
              <div style="font-size:11px;color:#aaa;">Ann. Return</div>
              <div style="font-size:22px;font-weight:700;color:{'#00b894' if ann_return>0 else '#d63031'};">
                {ann_return*100:.1f}%</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:11px;color:#aaa;">Z-Score</div>
              <div style="font-size:22px;font-weight:700;color:#74b9ff;">{z_score:.2f}σ</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:11px;color:#aaa;">Sharpe</div>
              <div style="font-size:22px;font-weight:700;color:#a29bfe;">{sharpe:.2f}</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:11px;color:#aaa;">Max Drawdown</div>
              <div style="font-size:22px;font-weight:700;color:#fd79a8;">{max_dd*100:.1f}%</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Show per‑ETF scores if available
        etf_scores = year_data.get('etf_scores', {})
        if etf_scores:
            st.subheader("ETF Scores (Probability)")
            sorted_scores = sorted(etf_scores.items(), key=lambda x: x[1], reverse=True)
            max_score = max(etf_scores.values())
            for name, score in sorted_scores:
                bar_w = int(score / max_score * 100)
                is_winner = (name == signal)
                bar_color = "#00d1b2" if is_winner else "#6c757d"
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;">
                    <span style="font-weight:500;">{name}</span>
                    <span>{score:.4f}</span>
                  </div>
                  <div style="background:#f0f0f0;border-radius:4px;height:20px;width:100%;">
                    <div style="width:{bar_w}%;height:100%;background:{bar_color};border-radius:4px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.info("📈 For full backtest details (equity curve, allocations, trade log), please refer to the **Multi‑Year Consensus Sweep** tab, which aggregates all years. The single‑year view shows the summary metrics from the sweep results.")

    st.divider()
    st.caption(f"Performance shown uses fixed risk parameters: fee {fee_bps} bps, stop loss {stop_loss_pct*100:.0f}%, re‑entry {z_reentry}σ.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Multi-Year Consensus Sweep (loads all years for selected option)
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"🔄 Multi-Year Consensus Sweep — {option_choice}")
    st.markdown(
        f"**All start years 2008–2025** are precomputed daily (after market close). "
        f"The table below shows the performance of each start year’s model on its out‑of‑sample test period. "
        f"**Weighted score** = 40% Ann. Return + 20% Z-Score + 20% Sharpe + 20% (–MaxDD), min‑max normalised."
    )

    # Load the most recent full set of sweep files
    sweep_data, sweep_date = load_sweep_cache_any(option)
    if not sweep_data:
        st.info("No sweep results found. They will be generated after the next daily run.")
        st.stop()

    st.caption(f"Results date: {sweep_date}")

    # Compute consensus
    consensus = compute_consensus(sweep_data)
    if not consensus:
        st.warning("Could not compute consensus from sweep data.")
        st.stop()

    winner    = consensus['winner']
    w_info    = consensus['etf_summary'][winner]
    win_color = colour_map.get(winner, "#00d1b2")
    score_share = w_info['score_share'] * 100
    split_signal = w_info['score_share'] < 0.4
    sig_label = "⚠️ Split Signal" if split_signal else "✅ Clear Signal"
    note = f"Score share {score_share:.0f}% · {w_info['n_years']} years · avg score {w_info['cum_score']:.4f}"

    # Winner banner
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                border:2px solid {win_color};border-radius:16px;
                padding:32px;text-align:center;margin:16px 0;
                box-shadow:0 8px 24px rgba(0,0,0,0.4);">
      <div style="font-size:11px;letter-spacing:3px;color:#aaa;margin-bottom:8px;">
        WEIGHTED CONSENSUS · TFT · {len(sweep_data)} START YEARS · Results {sweep_date}
      </div>
      <div style="font-size:72px;font-weight:900;color:{win_color};
                  text-shadow:0 0 30px {win_color}88;letter-spacing:2px;">
        {winner}
      </div>
      <div style="font-size:14px;color:#ccc;margin-top:8px;">{sig_label} · {note}</div>
      <div style="display:flex;justify-content:center;gap:32px;margin-top:20px;flex-wrap:wrap;">
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg Return</div>
          <div style="font-size:22px;font-weight:700;color:{'#00b894' if w_info['avg_return']>0 else '#d63031'};">
            {w_info['avg_return']*100:.1f}%</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg Z</div>
          <div style="font-size:22px;font-weight:700;color:#74b9ff;">{w_info['avg_z']:.2f}σ</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg Sharpe</div>
          <div style="font-size:22px;font-weight:700;color:#a29bfe;">{w_info['avg_sharpe']:.2f}</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg MaxDD</div>
          <div style="font-size:22px;font-weight:700;color:#fd79a8;">{w_info['avg_max_dd']*100:.1f}%</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Also‑ranked
    others = sorted([(e, v) for e, v in consensus['etf_summary'].items() if e != winner],
                    key=lambda x: -x[1]['cum_score'])
    also_parts = []
    for etf, v in others:
        col = colour_map.get(etf, "#888")
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

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Weighted Score per ETF**")
        etf_sum = consensus['etf_summary']
        sorted_etfs = sorted(etf_sum.keys(), key=lambda e: -etf_sum[e]['cum_score'])
        bar_colors  = [colour_map.get(e, "#888") for e in sorted_etfs]
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
            col = colour_map.get(etf, "#888")
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
        fig_scatter.add_hline(y=0, line_dash="dot",
                              line_color="rgba(255,255,255,0.3)",
                              annotation_text="Neutral")
        fig_scatter.update_layout(
            template="plotly_dark", height=380,
            xaxis_title="Start Year", yaxis_title="Z-Score (σ)",
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Per‑year breakdown table
    st.subheader("📋 Full Per-Year Breakdown")
    st.caption(
        "**Wtd Score** = 40% Ann. Return + 20% Z-Score + 20% Sharpe + 20% (–Max DD), "
        "min‑max normalised across years."
    )

    tbl_rows = []
    for row in sorted(consensus['per_year'], key=lambda r: r['year']):
        tbl_rows.append({
            'Start Year':   row['year'],
            'Signal':       row['signal'],
            'Wtd Score':    round(row['wtd_score'], 3),
            'Conviction':   row['conviction'],
            'Z-Score':      f"{row['z_score']:.2f}σ",
            'Ann. Return':  f"{row['ann_return']*100:.2f}%",
            'Sharpe':       f"{row['sharpe']:.2f}",
            'Max Drawdown': f"{row['max_dd']*100:.2f}%",
            'Lookback':     f"{row['lookback']}d",
        })

    tbl_df = pd.DataFrame(tbl_rows)

    def style_signal(val):
        col = colour_map.get(val, "#888")
        return f"background-color:{col}22;color:{col};font-weight:700;"

    def style_return(val):
        try:
            v = float(val.replace('%', ''))
            return 'color:#00b894;font-weight:600' if v > 0 else 'color:#d63031;font-weight:600'
        except Exception:
            return ''

    def style_wtd(val):
        try:
            float(val)
            return 'color:#00d1b2;font-weight:700'
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

    # How to read
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
Scores are min‑max normalised so no single metric dominates.
The ETF with the highest cumulative weighted score across all years wins.

**Split Signal warning**
If the winning ETF has a score share below 40%, signals are fragmented across years.
Treat the result with caution — no single ETF dominates across regimes.
""")
