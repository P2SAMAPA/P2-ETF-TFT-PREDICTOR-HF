import streamlit as st
import pandas as pd

def render_comparison_dashboard(transformer_results, regime_results, tc_pct):
    """
    Renders the side-by-side comparison between the two model approaches.
    """
    st.title("🏔️ Alpha Engine v13: Model Tournament")
    
    # --- TOP LEVEL METRICS ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model A: Transformer (Attention)")
        st.metric("Top Pick", f"{transformer_results['ticker']} ({transformer_results['horizon']})")
    
    with col2:
        st.subheader("📊 Model B: Regime Switcher (XGB/RF)")
        st.metric("Top Pick", f"{regime_results['ticker']} ({regime_results['horizon']})")

    st.divider()

    # --- COMPARISON TABLE ---
    st.subheader("🏆 Strategy Performance Comparison (Out-of-Sample)")
    
    comparison_data = {
        "Metric": [
            "Predicted ETF", 
            "Optimal Holding Period", 
            "Ann. Return (OOS)", 
            "Sharpe Ratio (SOFR)", 
            "Hit Ratio (15D)", 
            "Hit Ratio (30D)"
        ],
        "Transformer (Sequence)": [
            transformer_results['ticker'],
            transformer_results['horizon'],
            f"{transformer_results['ann_return']:.2%}",
            f"{transformer_results['sharpe']:.2f}",
            f"{transformer_results['hit_15']:.1%}",
            f"{transformer_results['hit_30']:.1%}"
        ],
        "Regime Switcher (XGB/RF)": [
            regime_results['ticker'],
            regime_results['horizon'],
            f"{regime_results['ann_return']:.2%}",
            f"{regime_results['sharpe']:.2f}",
            f"{regime_results['hit_15']:.1%}",
            f"{regime_results['hit_30']:.1%}"
        ]
    }
    
    st.table(pd.DataFrame(comparison_data))

    # --- TRANSACTION COST ANALYSIS ---
    st.sidebar.header("🕹️ Global Controls")
    tc_input = st.sidebar.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05)
    
    st.sidebar.info(f"Models are currently optimizing to beat a {tc_input}% drag per trade.")
    
    return tc_input / 100

def render_verification_logs(transformer_logs, regime_logs):
    tab1, tab2 = st.tabs(["Transformer Logs", "Regime Switcher Logs"])
    with tab1:
        st.dataframe(transformer_logs, use_container_width=True)
    with tab2:
        st.dataframe(regime_logs, use_container_width=True)
