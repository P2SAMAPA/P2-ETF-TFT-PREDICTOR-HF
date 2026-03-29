import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from huggingface_hub import HfApi, hf_hub_download
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

if 'approach' not in st.session_state:
    st.session_state.approach = "Per-Year Models"

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def get_latest_sweep_files(option_key, approach):
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
                                     filename=file_path, token=None)
        with open(local_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load {file_path}: {e}")
        return None

def compute_weighted_consensus(year_files):
    """
    Compute weighted consensus across years based on:
    60% annual return, 20% conviction Z, 10% Sharpe, 10% inverse max drawdown.
    Returns (weighted_scores_dict, df_weights) where df_weights includes year, top_etf, ann_return, conviction_z, sharpe, max_dd, weight.
    """
    years_data = []
    all_returns = []
    all_conv = []
    all_sharpe = []
    all_dd = []

    for year, (file_path, _) in year_files.items():
        data = load_sweep_json(file_path)
        if data and 'etf_scores' in data:
            ann_ret = data.get('ann_return')
            conv_z = data.get('conviction_z')
            sharpe = data.get('sharpe')
            max_dd = data.get('max_dd')
            top_etf = data.get('next_signal')  # ETF for that year
            if None not in (ann_ret, conv_z, sharpe, max_dd, top_etf):
                years_data.append({
                    'year': year,
                    'top_etf': top_etf,
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

    def min_max_normalize(values):
        vmin, vmax = min(values), max(values)
        if vmax == vmin:
            return [0.5] * len(values)
        return [(x - vmin) / (vmax - vmin) for x in values]

    norm_ret = min_max_normalize(all_returns)
    norm_conv = min_max_normalize(all_conv)
    norm_sharpe = min_max_normalize(all_sharpe)
    dd_scores = [1 / (1 + abs(dd)) for dd in all_dd]
    norm_dd = min_max_normalize(dd_scores)

    weights = []
    for i, year_data in enumerate(years_data):
        w = (0.6 * norm_ret[i] +
             0.2 * norm_conv[i] +
             0.1 * norm_sharpe[i] +
             0.1 * norm_dd[i])
        weights.append(w)
        year_data['weight'] = w

    # Create DataFrame for display
    df_weights = pd.DataFrame(years_data)[['year', 'top_etf', 'ann_return', 'conviction_z', 'sharpe', 'max_dd', 'weight']]
    df_weights = df_weights.sort_values('year')

    # Weighted average of ETF scores
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
# Sidebar (unchanged)
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
    else:
        option_key = "b"

    st.markdown("---")
    st.subheader("Strategy Parameters (Fixed)")
    st.markdown(f"**Transaction Fee**: {TRANSACTION_FEE_BPS} bps")
    st.markdown(f"**Stop Loss (2‑day cumulative)**: {abs(STOP_LOSS_PCT*100):.0f}%")
    st.markdown(f"**Re‑entry Conviction (σ)**: {RE_ENTRY_CONVICTION:.2f}")
    st.markdown(f"**Min Entry Conviction (σ)**: {MIN_ENTRY_CONVICTION:.2f}")

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
# Main area: two tabs
# ------------------------------------------------------------------------------
st.title("ETF Temporal Fusion Transformer Predictor")

tab1, tab2 = st.tabs(["Single Year", "Consensus Sweep"])

# ------------------------------------------------------------------------------
# Tab 1: Single Year (unchanged)
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Single‑Year Prediction")
    years = list(range(2008, 2026))
    selected_year = st.selectbox("Select Year", years, index=len(years)-1)

    year_files = get_latest_sweep_files(option_key, st.session_state.approach)
    if selected_year in year_files:
        file_path, date_tag = year_files[selected_year]
        data = load_sweep_json(file_path)
        if data:
            st.success(f"Data for {selected_year} (run {date_tag})")

            # Hero box for this year
            next_signal = data.get('next_signal', 'N/A')
            conv_z = data.get('conviction_z', 'N/A')
            conv_label = data.get('conviction_label', 'N/A')
            next_date = data.get('next_date', 'Not available')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Next Trading Day Signal", next_signal)
            with col2:
                st.metric("Conviction Z‑score", f"{conv_z:.2f}" if isinstance(conv_z, (int, float)) else conv_z)
            with col3:
                st.metric("Conviction Label", conv_label)
            st.caption(f"*For trading date: {next_date}*")
            st.markdown("---")

            # Bar chart of ETF conviction scores for this year
            scores = data.get('etf_scores', {})
            if scores:
                df_scores = pd.DataFrame(list(scores.items()), columns=['ETF', 'Score'])
                df_scores = df_scores.sort_values('Score', ascending=True)
                fig = px.bar(df_scores, x='Score', y='ETF', orientation='h', title="ETF Conviction Scores")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Strategy Metrics for this Year")
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
# Tab 2: Consensus Sweep (added top_etf column to year table)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Weighted Consensus Across All Years")
    st.markdown("Consensus weights: 60% Annual Return, 20% Conviction Z, 10% Sharpe, 10% Inverse Max Drawdown")

    year_files = get_latest_sweep_files(option_key, st.session_state.approach)
    if not year_files:
        st.warning("No sweep files found. Please run the consensus sweep workflow first.")
    else:
        weighted_scores, df_weights = compute_weighted_consensus(year_files)
        if weighted_scores:
            # Hero box: top consensus ETF
            top_etf = max(weighted_scores, key=weighted_scores.get)
            top_score = weighted_scores[top_etf]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🏆 Top Consensus Pick", top_etf)
            with col2:
                st.metric("Weighted Score", f"{top_score:.3f}")
            st.markdown("---")

            # Bar chart of all weighted consensus ETF scores
            df_consensus = pd.DataFrame(list(weighted_scores.items()), columns=['ETF', 'Weighted Score'])
            df_consensus = df_consensus.sort_values('Weighted Score', ascending=True)
            fig = px.bar(df_consensus, x='Weighted Score', y='ETF', orientation='h',
                         title="Weighted Consensus ETF Scores")
            st.plotly_chart(fig, use_container_width=True)

            # Year‑by‑year metrics table including the top ETF for each year
            with st.expander("Show year‑by‑year metrics and weights"):
                # Rename 'top_etf' column to 'Top ETF' for display
                display_df = df_weights.rename(columns={'top_etf': 'Top ETF'})
                st.dataframe(display_df.style.format({
                    'ann_return': '{:.2%}',
                    'conviction_z': '{:.3f}',
                    'sharpe': '{:.2f}',
                    'max_dd': '{:.2%}',
                    'weight': '{:.3f}'
                }))
                st.caption("Weight = 0.6*Norm(AnnRet) + 0.2*Norm(ConvZ) + 0.1*Norm(Sharpe) + 0.1*Norm(InvDD)")
        else:
            st.error("Could not compute consensus – missing metrics in sweep files.")
