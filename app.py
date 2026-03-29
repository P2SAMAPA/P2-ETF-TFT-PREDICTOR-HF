import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from huggingface_hub import HfApi, hf_hub_download
from datetime import datetime
import json

# Hardcoded strategy parameters (display only)
TRANSACTION_FEE_BPS = 12
STOP_LOSS_PCT = -0.12
RE_ENTRY_CONVICTION = 0.90
MIN_ENTRY_CONVICTION = 0.50

st.set_page_config(
    page_title="ETF Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-tft-outputs"
GITHUB_REPO = "P2SAMAPA/P2-ETF-TFT-PREDICTOR-HF"

OPTION_A_ETFS = ["TLT", "IEF", "SHY", "LQD", "HYG", "GLD", "DBC"]
OPTION_B_ETFS = ["SPY", "QQQ", "IWM", "EEM", "EFA", "XLF", "XLK", "XLE", "XLV", "XLI"]

# Session state
if 'approach' not in st.session_state:
    st.session_state.approach = "Per-Year Models"

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def get_latest_sweep_files(option_key, approach):
    """Return dict {year: (file_path, date_tag)} for the latest sweep files."""
    api = HfApi()
    try:
        prefix = "sweep" if approach == "Per-Year Models" else "global_sweep"
        base = f"{prefix}/option_{option_key}/signals_"
        files = api.list_repo_files(repo_id=HF_OUTPUT_REPO, repo_type="dataset")
        year_files = {}
        for f in files:
            if f.startswith(base) and f.endswith(".json"):
                parts = f.replace(".json", "").split("_")
                if len(parts) >= 3:
                    year_str = parts[-2]
                    date_str = parts[-1]
                    try:
                        year = int(year_str)
                        if year not in year_files or date_str > year_files[year][1]:
                            year_files[year] = (f, date_str)
                    except:
                        pass
        return year_files
    except Exception as e:
        st.error(f"Error listing files: {e}")
        return {}

def load_sweep_json(file_path):
    try:
        local_path = hf_hub_download(repo_id=HF_OUTPUT_REPO, repo_type="dataset",
                                     filename=file_path, token=None)  # no token needed for public read
        with open(local_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load {file_path}: {e}")
        return None

def get_latest_signal(option_key, approach):
    """Return (next_signal, conviction_z, conviction_label, next_date) from most recent year."""
    year_files = get_latest_sweep_files(option_key, approach)
    if not year_files:
        return None, None, None, None
    latest_year = max(year_files.keys())
    file_path, _ = year_files[latest_year]
    data = load_sweep_json(file_path)
    if data:
        return (data.get('next_signal'), data.get('conviction_z'),
                data.get('conviction_label'), data.get('next_date'))
    return None, None, None, None

def compute_weighted_consensus(year_files):
    """
    For each year, load metrics: ann_return, conviction_z, sharpe, max_dd.
    Compute year weight = 0.6*norm_ret + 0.2*norm_conv + 0.1*norm_sharpe + 0.1*norm_dd
    Then weighted average of ETF scores across years.
    Returns (weighted_scores_dict, years_weights_df)
    """
    years_data = []
    all_returns = []
    all_conv = []
    all_sharpe = []
    all_dd = []   # raw max_dd (negative)

    for year, (file_path, _) in year_files.items():
        data = load_sweep_json(file_path)
        if data and 'etf_scores' in data:
            ann_ret = data.get('ann_return')
            conv_z = data.get('conviction_z')
            sharpe = data.get('sharpe')
            max_dd = data.get('max_dd')
            if None not in (ann_ret, conv_z, sharpe, max_dd):
                years_data.append({
                    'year': year,
                    'ann_return': ann_ret,
                    'conviction_z': conv_z,
                    'sharpe': sharpe,
                    'max_dd': max_dd,
                    'etf_scores': data['etf_scores']
                })
                all_returns.append(ann_ret)
                all_conv.append(conv_z)
                all_sharpe.append(sharpe)
                all_dd.append(max_dd)

    if not years_data:
        return None, None

    # Normalise each metric (min-max)
    def min_max_normalize(values):
        vmin, vmax = min(values), max(values)
        if vmax == vmin:
            return [0.5] * len(values)
        return [(x - vmin) / (vmax - vmin) for x in values]

    norm_ret = min_max_normalize(all_returns)
    norm_conv = min_max_normalize(all_conv)
    norm_sharpe = min_max_normalize(all_sharpe)

    # For max_dd (negative), lower drawdown (closer to zero) is better.
    # Compute a score = 1 / (1 + abs(max_dd)) then normalise.
    dd_scores = [1 / (1 + abs(dd)) for dd in all_dd]
    norm_dd = min_max_normalize(dd_scores)

    # Compute year weights
    weights = []
    for i, year_data in enumerate(years_data):
        w = (0.6 * norm_ret[i] +
             0.2 * norm_conv[i] +
             0.1 * norm_sharpe[i] +
             0.1 * norm_dd[i])
        weights.append(w)
        year_data['weight'] = w

    # Create DataFrame for display
    df_weights = pd.DataFrame(years_data)[['year', 'ann_return', 'conviction_z', 'sharpe', 'max_dd', 'weight']]
    df_weights = df_weights.sort_values('year')

    # Weighted average of ETF scores across years
    etf_names = list(years_data[0]['etf_scores'].keys())
    weighted_scores = {etf: 0.0 for etf in etf_names}
    total_weight = sum(weights)
    for i, year_data in enumerate(years_data):
        w = weights[i]
        scores = year_data['etf_scores']
        for etf in etf_names:
            weighted_scores[etf] += w * scores.get(etf, 0.0)
    if total_weight > 0:
        for etf in weighted_scores:
            weighted_scores[etf] /= total_weight

    return weighted_scores, df_weights

# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 ETF Predictor")
    st.markdown("---")

    approach = st.radio(
        "Model Approach",
        ["Per-Year Models", "Global Model"],
        help="Per-Year: separate model for each year. Global: single model trained on all data."
    )
    st.session_state.approach = approach

    option = st.radio(
        "Asset Class",
        ["FI (Fixed Income)", "Equity"],
        help="FI: bonds, commodities; Equity: US & international stocks"
    )
    if option == "FI (Fixed Income)":
        option_key = "a"
        etf_list = OPTION_A_ETFS
    else:
        option_key = "b"
        etf_list = OPTION_B_ETFS

    st.markdown("---")
    st.subheader("Strategy Parameters (Fixed)")
    st.markdown(f"**Transaction Fee**: {TRANSACTION_FEE_BPS} bps")
    st.markdown(f"**Stop Loss (2‑day cumulative)**: {abs(STOP_LOSS_PCT*100):.0f}%")
    st.markdown(f"**Re‑entry Conviction (σ)**: {RE_ENTRY_CONVICTION:.2f}")
    st.markdown(f"**Min Entry Conviction (σ)**: {MIN_ENTRY_CONVICTION:.2f}")

    # Refresh button moved to sidebar
    if st.button("🔄 Refresh Sweep Data", help="Reload all sweep data from Hugging Face"):
        st.rerun()

    st.markdown("---")
    st.markdown("**Data Sources**")
    st.markdown("- FRED (DTB3, term spreads)")
    st.markdown("- Yahoo Finance (ETF prices)")
    st.markdown("- VIX term structure")
    st.markdown("- Credit spreads")

    st.markdown("---")
    st.markdown(f"GitHub: [{GITHUB_REPO}](https://github.com/{GITHUB_REPO})")
    st.markdown(f"Outputs: [{HF_OUTPUT_REPO}](https://huggingface.co/datasets/{HF_OUTPUT_REPO})")

# ------------------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------------------
st.title("ETF Temporal Fusion Transformer Predictor")

# Hero box: Next signal for next trading day
next_signal, conv_z, conv_label, next_date = get_latest_signal(option_key, st.session_state.approach)
if next_signal:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Next Trading Day Signal", next_signal)
    with col2:
        st.metric("Conviction Z‑score", f"{conv_z:.2f}" if conv_z else "N/A")
    with col3:
        st.metric("Conviction Label", conv_label if conv_label else "N/A")
    st.caption(f"*For trading date: {next_date if next_date else 'Not available'}*")
    st.markdown("---")
else:
    st.info("No sweep data available yet. Please run the consensus sweep workflow.")
    st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["Single Year", "Consensus Sweep"])

# ------------------------------------------------------------------------------
# Tab 1: Single Year
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Single‑Year Sweep Results")
    col1, col2 = st.columns([1, 2])
    with col1:
        years = list(range(2008, 2026))
        selected_year = st.selectbox("Select Year", years, index=len(years)-1)
    with col2:
        st.markdown("**Sweep data from the most recent run**")

    year_files = get_latest_sweep_files(option_key, st.session_state.approach)
    if selected_year in year_files:
        file_path, date_tag = year_files[selected_year]
        data = load_sweep_json(file_path)
        if data:
            st.success(f"Loaded sweep for {selected_year} (run {date_tag})")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Next Signal", data.get('next_signal', 'N/A'))
            with col2:
                st.metric("Conviction Z", data.get('conviction_z', 'N/A'))
            with col3:
                st.metric("Conviction Label", data.get('conviction_label', 'N/A'))

            scores = data.get('etf_scores', {})
            if scores:
                df_scores = pd.DataFrame(list(scores.items()), columns=['ETF', 'Score'])
                df_scores = df_scores.sort_values('Score', ascending=True)
                fig = px.bar(df_scores, x='Score', y='ETF', orientation='h', title="ETF Conviction Scores")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Strategy Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Annual Return", f"{data.get('ann_return', 0)*100:.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{data.get('sharpe', 0):.2f}")
            with col3:
                st.metric("Max Drawdown", f"{data.get('max_dd', 0)*100:.2f}%")
        else:
            st.warning(f"Could not load data for {selected_year}")
    else:
        st.info(f"No sweep data available for {selected_year}. Run the consensus sweep workflow.")

# ------------------------------------------------------------------------------
# Tab 2: Consensus Sweep (all years, weighted by strategy metrics)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader(f"Consensus Sweep – {st.session_state.approach}")
    st.markdown("Weighted consensus across all years based on **strategy performance**:")
    st.markdown("- 60% Annual Return, 20% Conviction Z, 10% Sharpe Ratio, 10% Inverse Max Drawdown")

    year_files = get_latest_sweep_files(option_key, st.session_state.approach)
    if not year_files:
        st.warning("No sweep files found. Please run the consensus sweep workflow first.")
    else:
        weighted_scores, df_weights = compute_weighted_consensus(year_files)
        if weighted_scores:
            # Show top ETF
            top_etf = max(weighted_scores, key=weighted_scores.get)
            top_score = weighted_scores[top_etf]
            st.metric("🏆 Top Consensus Pick", top_etf, f"{top_score:.3f}")

            # Bar chart of consensus ETF scores
            df_consensus = pd.DataFrame(list(weighted_scores.items()), columns=['ETF', 'Weighted Score'])
            df_consensus = df_consensus.sort_values('Weighted Score', ascending=True)
            fig = px.bar(df_consensus, x='Weighted Score', y='ETF', orientation='h',
                         title="Weighted Consensus ETF Scores")
            st.plotly_chart(fig, use_container_width=True)

            # Show detailed year weights
            with st.expander("Show year‑by‑year metrics and weights"):
                st.dataframe(df_weights.style.format({
                    'ann_return': '{:.2%}',
                    'conviction_z': '{:.3f}',
                    'sharpe': '{:.2f}',
                    'max_dd': '{:.2%}',
                    'weight': '{:.3f}'
                }))
                st.caption("Weight = 0.6*Norm(AnnRet) + 0.2*Norm(ConvZ) + 0.1*Norm(Sharpe) + 0.1*Norm(InvDD)")
        else:
            st.error("Could not compute consensus – missing metrics in sweep files.")
