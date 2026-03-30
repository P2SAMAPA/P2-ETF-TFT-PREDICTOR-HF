import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from huggingface_hub import HfApi, hf_hub_download
import json

TRANSACTION_FEE_BPS = 12
STOP_LOSS_PCT = -0.12
RE_ENTRY_CONVICTION = 0.90
MIN_ENTRY_CONVICTION = 0.50

st.set_page_config(page_title="ETF Predictor", page_icon="🤖", layout="wide")
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
        # 🔴 FIX 1: Map approach to folder names - support both old 'sweep' and new 'yearwise_sweep'
        if approach == "Per-Year Models":
            prefixes = ["yearwise_sweep", "sweep"]  # Try new first, then old
        else:
            prefixes = ["global_sweep"]
            
        year_files = {}
        files = api.list_repo_files(repo_id=HF_OUTPUT_REPO, repo_type="dataset")
        
        for prefix in prefixes:
            base = f"{prefix}/option_{option_key}/signals_"
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

def clean_etf_name(name):
    """🔴 FIX 2: Strip _Ret suffix from ETF names"""
    if name and isinstance(name, str):
        return name.replace('_Ret', '')
    return name

def calculate_conviction_z(scores):
    """🔴 FIX 3: Calculate conviction_z from etf_scores if missing"""
    if not scores:
        return None, None
    
    values = list(scores.values())
    if not values:
        return None, None
    
    max_score = max(values)
    mean_score = np.mean(values)
    std_score = np.std(values) + 1e-9
    z_score = (max_score - mean_score) / std_score
    
    # Determine label
    if z_score > 1.5:
        label = "Strong"
    elif z_score > 0.8:
        label = "Moderate"
    else:
        label = "Weak"
    
    return z_score, label

def compute_combined_consensus(year_files, decay_alpha=0.9, perf_weight=0.7, freq_weight=0.3):
    """
    Consensus using:
      - Recency‑weighted performance score (positive‑return years only, with exponential decay)
      - Frequency of being top pick across positive‑return years
    Returns (final_scores_dict, df_weights)
    """
    # First, collect data for all years
    years_data = []
    for year, (file_path, _) in year_files.items():
        data = load_sweep_json(file_path)
        if data and 'etf_scores' in data:
            # 🔴 FIX 4: Clean ETF names and calculate missing fields
            raw_scores = data['etf_scores']
            data['etf_scores'] = {clean_etf_name(k): v for k, v in raw_scores.items()}
            
            # Calculate conviction_z if missing
            if 'conviction_z' not in data or data['conviction_z'] is None or data['conviction_z'] == 'N/A':
                z, _ = calculate_conviction_z(data['etf_scores'])
                data['conviction_z'] = z
            
            # Clean next_signal
            if 'next_signal' in data:
                data['next_signal'] = clean_etf_name(data['next_signal'])
            
            ann_ret = data.get('ann_return')
            conv_z = data.get('conviction_z')
            sharpe = data.get('sharpe')
            max_dd = data.get('max_dd')
            
            # Only require ann_return and sharpe (conv_z might be None)
            if ann_ret is not None and sharpe is not None and max_dd is not None:
                years_data.append({
                    'year': year,
                    'ann_return': ann_ret,
                    'conviction_z': conv_z if conv_z is not None else 0,
                    'sharpe': sharpe,
                    'max_dd': max_dd,
                    'etf_scores': data['etf_scores'],
                    'next_signal': data.get('next_signal', 'N/A')
                })

    if not years_data:
        return None, None

    # Identify years with positive return (for performance weighting)
    pos_years = [yd for yd in years_data if yd['ann_return'] > 0]
    if not pos_years:
        # No positive years – fall back to all years
        pos_years = years_data

    # Normalisation function
    def min_max_normalize(values):
        if not values:
            return []
        vmin, vmax = min(values), max(values)
        if vmax == vmin:
            return [0.5] * len(values)
        return [(x - vmin) / (vmax - vmin) for x in values]

    # Compute recency‑weighted performance score for each ETF
    # Step 1: compute year weight without decay (based on positive years only)
    all_returns = [yd['ann_return'] for yd in pos_years]
    all_conv = [yd['conviction_z'] for yd in pos_years]
    all_sharpe = [yd['sharpe'] for yd in pos_years]
    all_dd = [yd['max_dd'] for yd in pos_years]

    norm_ret = min_max_normalize(all_returns)
    norm_conv = min_max_normalize(all_conv)
    norm_sharpe = min_max_normalize(all_sharpe)
    dd_scores = [1/(1+abs(dd)) for dd in all_dd]
    norm_dd = min_max_normalize(dd_scores)

    # For each positive year, base weight (without decay)
    base_weights = []
    for i in range(len(pos_years)):
        w = 0.6*norm_ret[i] + 0.2*norm_conv[i] + 0.1*norm_sharpe[i] + 0.1*norm_dd[i]
        base_weights.append(w)

    # Apply exponential decay based on recency: decay = alpha^(current_year - year)
    # Use the latest year as reference
    latest_year = max(yd['year'] for yd in pos_years)
    decay_factors = [decay_alpha ** (latest_year - yd['year']) for yd in pos_years]
    # Normalise decay factors to sum to number of years? Not necessary; we'll multiply.
    year_weights = [base_weights[i] * decay_factors[i] for i in range(len(pos_years))]

    # Compute weighted average of ETF conviction scores using these year_weights
    etf_names = list(pos_years[0]['etf_scores'].keys())
    perf_scores = {etf: 0.0 for etf in etf_names}
    total_weight = sum(year_weights)
    for i, yd in enumerate(pos_years):
        w = year_weights[i]
        scores = yd['etf_scores']
        for etf in etf_names:
            perf_scores[etf] += w * scores.get(etf, 0.0)
    if total_weight > 0:
        for etf in perf_scores:
            perf_scores[etf] /= total_weight

    # Frequency of being top pick among positive years
    top_counts = {etf: 0 for etf in etf_names}
    for yd in pos_years:
        top_etf = yd['next_signal']
        if top_etf in top_counts:
            top_counts[top_etf] += 1
    # Normalise frequencies to [0,1] (min‑max)
    freq_values = list(top_counts.values())
    if max(freq_values) > min(freq_values):
        norm_freq = {etf: (cnt - min(freq_values)) / (max(freq_values) - min(freq_values))
                     for etf, cnt in top_counts.items()}
    else:
        norm_freq = {etf: 0.5 for etf in etf_names}

    # Combine performance score and frequency
    final_scores = {}
    for etf in etf_names:
        final_scores[etf] = perf_weight * perf_scores.get(etf, 0) + freq_weight * norm_freq.get(etf, 0)

    # Prepare year‑by‑year DataFrame for display (include all years, with weight = base weight * decay)
    df_weights = pd.DataFrame([{
        'year': yd['year'],
        'Top ETF': yd['next_signal'],
        'ann_return': yd['ann_return'],
        'conviction_z': yd['conviction_z'],
        'sharpe': yd['sharpe'],
        'max_dd': yd['max_dd'],
        'base_weight': None,  # we don't have base weight for non‑positive years in this loop; simplify
    } for yd in years_data])
    # Add a column for the weight used (only for positive years, with decay)
    weight_map = {}
    for i, yd in enumerate(pos_years):
        weight_map[yd['year']] = year_weights[i]
    df_weights['weight'] = df_weights['year'].map(weight_map).fillna(0)
    df_weights = df_weights.sort_values('year')

    return final_scores, df_weights

# ------------------------------------------------------------------------------
# Sidebar (unchanged)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 ETF Predictor")
    st.markdown("---")
    approach = st.radio("Model Approach", ["Per-Year Models", "Global Model"])
    st.session_state.approach = approach
    option = st.radio("Asset Class", ["FI (Fixed Income)", "Equity"])
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
    if st.button("🔄 Refresh Sweep Data"):
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
tab1, tab2 = st.tabs(["Single Year", "Consensus Sweep"])

# Tab 1: Single Year (with fixes)
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
            
            # 🔴 FIX: Clean and process data
            next_signal = clean_etf_name(data.get('next_signal', 'N/A'))
            
            # Calculate conviction if missing
            scores = data.get('etf_scores', {})
            if scores:
                scores = {clean_etf_name(k): v for k, v in scores.items()}
                data['etf_scores'] = scores
            
            if 'conviction_z' not in data or data['conviction_z'] is None or data['conviction_z'] == 'N/A':
                conv_z, conv_label = calculate_conviction_z(scores)
                data['conviction_z'] = conv_z if conv_z is not None else 'N/A'
                data['conviction_label'] = conv_label if conv_label else 'N/A'
            else:
                conv_z = data['conviction_z']
                # Determine label if not present
                if 'conviction_label' not in data:
                    if conv_z > 1.5:
                        data['conviction_label'] = "Strong"
                    elif conv_z > 0.8:
                        data['conviction_label'] = "Moderate"
                    else:
                        data['conviction_label'] = "Weak"
            
            conv_z_display = f"{data['conviction_z']:.2f}" if isinstance(data['conviction_z'], (int, float)) else data['conviction_z']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Next Trading Day Signal", next_signal)
            with col2:
                st.metric("Conviction Z‑score", conv_z_display)
            with col3:
                st.metric("Conviction Label", data.get('conviction_label', 'N/A'))
            
            # Show model type info
            model_type = data.get('model_type', 'unknown')
            st.caption(f"*Model type: {model_type}*")
            
            st.markdown("---")
            
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
        st.info(f"No sweep data available for {selected_year}. Available years: {list(year_files.keys())}")

# Tab 2: Consensus Sweep (with fixes)
with tab2:
    st.subheader("Weighted Consensus Across All Years")
    st.markdown("**Combined method:**")
    st.markdown("- Performance weighting (60% return, 20% conviction, 10% Sharpe, 10% inverse drawdown) with **exponential decay for recency** (α=0.9)")
    st.markdown("- Frequency of being top pick across profitable years")
    st.markdown("- Final score = 70% performance score + 30% frequency score")

    year_files = get_latest_sweep_files(option_key, st.session_state.approach)
    if not year_files:
        st.warning("No sweep files found. Please run the consensus sweep workflow first.")
        st.info(f"Looking for: {st.session_state.approach} with option {option_key}")
    else:
        st.info(f"Found {len(year_files)} years of data: {sorted(year_files.keys())}")
        final_scores, df_weights = compute_combined_consensus(year_files, decay_alpha=0.9, perf_weight=0.7, freq_weight=0.3)
        if final_scores:
            top_etf = max(final_scores, key=final_scores.get)
            top_score = final_scores[top_etf]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🏆 Top Consensus Pick", top_etf)
            with col2:
                st.metric("Combined Score", f"{top_score:.3f}")
            st.markdown("---")
            # Bar chart of all final scores
            df_consensus = pd.DataFrame(list(final_scores.items()), columns=['ETF', 'Combined Score'])
            df_consensus = df_consensus.sort_values('Combined Score', ascending=True)
            fig = px.bar(df_consensus, x='Combined Score', y='ETF', orientation='h',
                         title="ETF Combined Consensus Scores")
            st.plotly_chart(fig, use_container_width=True)
            # Year‑by‑year table
            with st.expander("Show year‑by‑year metrics and weights"):
                st.dataframe(df_weights.style.format({
                    'ann_return': '{:.2%}',
                    'conviction_z': '{:.3f}',
                    'sharpe': '{:.2f}',
                    'max_dd': '{:.2%}',
                    'weight': '{:.3f}'
                }))
                st.caption("Weight = base performance weight (positive years only) × decay factor (α^(latest_year - year)). Zero for negative return years.")
        else:
            st.error("Could not compute consensus – missing data or no valid years found.")
            st.info("Debug: Check that sweep files contain 'ann_return', 'sharpe', 'max_dd', and 'etf_scores' fields.")
