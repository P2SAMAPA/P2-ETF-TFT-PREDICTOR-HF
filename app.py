import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import sys
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
import tempfile
import time
import subprocess

# Set page config
st.set_page_config(
    page_title="ETF Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-tft-outputs"
HF_DATA_REPO = "P2SAMAPA/my-etf-data"
GITHUB_REPO = "P2SAMAPA/P2-ETF-TFT-PREDICTOR-HF"

# Option definitions
OPTION_A_ETFS = ["TLT", "IEF", "SHY", "LQD", "HYG", "GLD", "DBC"]
OPTION_B_ETFS = ["SPY", "QQQ", "IWM", "EEM", "EFA", "XLF", "XLK", "XLE", "XLV", "XLI"]

# Initialize session state
if 'hf_token' not in st.session_state:
    st.session_state.hf_token = None
if 'approach' not in st.session_state:
    st.session_state.approach = "Per-Year Models"

# Sidebar
with st.sidebar:
    st.title("🤖 ETF Predictor")
    st.markdown("---")
    
    # Model approach selection
    approach = st.radio(
        "Model Approach",
        ["Per-Year Models", "Global Model"],
        help="Per-Year: separate model for each year (slower, more accurate historically). Global: single model trained on all data (faster, consistent view)."
    )
    st.session_state.approach = approach
    
    # Option selection
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
    
    # HF Token input (for triggering workflows)
    token = st.text_input("Hugging Face Token (for workflow triggers)", type="password")
    if token:
        st.session_state.hf_token = token
    
    st.markdown("---")
    st.markdown("**Data Sources**")
    st.markdown("- FRED (DTB3, term spreads)")
    st.markdown("- Yahoo Finance (ETF prices)")
    st.markdown("- VIX term structure")
    st.markdown("- Credit spreads")
    
    st.markdown("---")
    st.markdown(f"GitHub: [{GITHUB_REPO}](https://github.com/{GITHUB_REPO})")
    st.markdown(f"Outputs: [{HF_OUTPUT_REPO}](https://huggingface.co/datasets/{HF_OUTPUT_REPO})")

# Main area
st.title("ETF Temporal Fusion Transformer Predictor")
st.markdown("### Real-time signals & strategy backtest")

# Tabs
tab1, tab2 = st.tabs(["Single Year", "Consensus Sweep"])

# Helper functions
def get_latest_sweep_files(option_key, approach):
    """Return a dict {year: (file_path, date_tag)} for the latest sweep files."""
    api = HfApi()
    try:
        # Determine prefix based on approach
        if approach == "Per-Year Models":
            prefix = f"sweep/option_{option_key}/signals_"
        else:
            prefix = f"global_sweep/option_{option_key}/signals_"
        
        files = api.list_repo_files(repo_id=HF_OUTPUT_REPO, repo_type="dataset")
        # Filter files that match pattern: prefix<year>_<date>.json
        year_files = {}
        for f in files:
            if f.startswith(prefix) and f.endswith(".json"):
                parts = f.replace(".json", "").split("_")
                if len(parts) >= 3:
                    year_str = parts[-2]  # assumes signals_2008_20260329.json
                    date_str = parts[-1]
                    try:
                        year = int(year_str)
                        # Keep only the latest date for each year
                        if year not in year_files or date_str > year_files[year][1]:
                            year_files[year] = (f, date_str)
                    except:
                        pass
        return year_files
    except Exception as e:
        st.error(f"Error listing files: {e}")
        return {}

def load_sweep_json(file_path):
    """Download and load a sweep JSON from HF."""
    try:
        local_path = hf_hub_download(repo_id=HF_OUTPUT_REPO, repo_type="dataset", filename=file_path, token=st.session_state.hf_token)
        with open(local_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load {file_path}: {e}")
        return None

def compute_weighted_consensus(year_files):
    """Given a dict of year -> (file_path, date), compute weighted average of signals."""
    weights = []
    scores = []
    for year, (file_path, _) in year_files.items():
        data = load_sweep_json(file_path)
        if data and 'etf_scores' in data:
            # Weight by year recency (e.g., 1 for oldest, increasing)
            weight = year - 2007  # 2008 -> 1, 2025 -> 18
            weights.append(weight)
            # Convert scores to numpy array for averaging
            score_dict = data['etf_scores']
            scores.append(np.array(list(score_dict.values())))
    if not scores:
        return None, None
    scores = np.array(scores)
    weights = np.array(weights)
    weighted_avg = np.average(scores, weights=weights, axis=0)
    etf_names = list(score_dict.keys())
    return dict(zip(etf_names, weighted_avg)), weights

# Tab 1: Single Year
with tab1:
    st.subheader("Single-Year Sweep Results")
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
            # Display signal
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Next Signal", data.get('next_signal', 'N/A'))
            with col2:
                st.metric("Conviction Z", data.get('conviction_z', 'N/A'))
            with col3:
                st.metric("Conviction Label", data.get('conviction_label', 'N/A'))
            
            # ETF scores bar chart
            scores = data.get('etf_scores', {})
            if scores:
                df_scores = pd.DataFrame(list(scores.items()), columns=['ETF', 'Score'])
                df_scores = df_scores.sort_values('Score', ascending=True)
                fig = px.bar(df_scores, x='Score', y='ETF', orientation='h', title="ETF Conviction Scores")
                st.plotly_chart(fig, use_container_width=True)
            
            # Strategy metrics
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

# Tab 2: Consensus Sweep (all years)
with tab2:
    st.subheader(f"Consensus Sweep – {st.session_state.approach}")
    st.markdown("Weighted consensus across all years (more recent years have higher weight)")
    
    if st.button("Refresh Sweep Data"):
        st.rerun()
    
    year_files = get_latest_sweep_files(option_key, st.session_state.approach)
    if not year_files:
        st.warning("No sweep files found. Please run the consensus sweep workflow first.")
    else:
        # Show available years
        available_years = sorted(year_files.keys())
        st.write(f"Available years: {available_years}")
        
        weighted_scores, weights = compute_weighted_consensus(year_files)
        if weighted_scores:
            df_consensus = pd.DataFrame(list(weighted_scores.items()), columns=['ETF', 'Weighted Score'])
            df_consensus = df_consensus.sort_values('Weighted Score', ascending=True)
            fig = px.bar(df_consensus, x='Weighted Score', y='ETF', orientation='h', title="Consensus ETF Scores")
            st.plotly_chart(fig, use_container_width=True)
            
            # Top pick
            top_etf = df_consensus.iloc[-1]['ETF']
            top_score = df_consensus.iloc[-1]['Weighted Score']
            st.metric("Top Consensus Pick", top_etf, f"{top_score:.3f}")
        else:
            st.error("Could not compute consensus.")
    
    # Run consensus sweep button (only for per-year, as global is nightly)
    if st.session_state.approach == "Per-Year Models":
        st.markdown("---")
        st.subheader("Run Consensus Sweep")
        st.markdown("Trigger the GitHub Actions workflow to generate the latest sweep files.")
        if st.button("Run Consensus Sweep (Per-Year)"):
            if not st.session_state.hf_token:
                st.error("Please enter your Hugging Face token in the sidebar.")
            else:
                # Trigger workflow via GitHub API
                with st.spinner("Triggering workflow..."):
                    # We'll use the GitHub API to dispatch the workflow
                    # This requires a GitHub token. We'll ask the user to set it in secrets or just provide a link.
                    # For simplicity, we'll just show a link.
                    st.info(f"Manually trigger the [consensus sweep workflow](https://github.com/{GITHUB_REPO}/actions/workflows/consensus_sweep.yml) on GitHub.")
    else:
        st.markdown("---")
        st.info("Global model consensus is updated nightly via GitHub Actions. To force a new run, visit the [global consensus sweep workflow](https://github.com/P2SAMAPA/P2-ETF-TFT-PREDICTOR-HF/actions/workflows/global_consensus_sweep.yml).")

# Footer
st.markdown("---")
st.caption("Data updates daily. Model retrained weekly. Predictions are for informational purposes only.")
