import streamlit as st
import pandas as pd

def render_comparison_dashboard(transformer_results, regime_results, tc_pct):
    st.title("🏔️ Alpha Engine ver1.0")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🤖 Model A: Transformer")
        # Added key to metrics to prevent duplication errors
        st.metric("Top Pick", f"{transformer_results['ticker']} ({transformer_results['horizon']})", key="m1")
    
    with col2:
        st.subheader("📊 Model B: Regime Switcher")
        st.metric("Top Pick", f"{regime_results['ticker']} ({regime_results['horizon']})", key="m2")

    st.divider()
    st.subheader("🏆 Strategy Performance Comparison (Out-of-Sample)")
    
    comparison_data = {
        "Metric": ["Predicted ETF", "Optimal Hold", "Ann. Return", "Sharpe", "Hit Ratio 15D", "Hit Ratio 30D"],
        "Transformer": [
            transformer_results['ticker'], transformer_results['horizon'],
            f"{transformer_results['ann_return']:.2%}", f"{transformer_results['sharpe']:.2f}",
            f"{transformer_results['hit_15']:.1%}", f"{transformer_results['hit_30']:.1%}"
        ],
        "Regime Switcher": [
            regime_results['ticker'], regime_results['horizon'],
            f"{regime_results['ann_return']:.2%}", f"{regime_results['sharpe']:.2f}",
            f"{regime_results['hit_15']:.1%}", f"{regime_results['hit_30']:.1%}"
        ]
    }
    st.table(pd.DataFrame(comparison_data))

    # The Slider fix: Add a unique key
    st.sidebar.header("🕹️ Global Controls")
    tc_input = st.sidebar.slider(
        "Transaction Cost (%)", 
        0.0, 1.0, 0.1, 0.05, 
        key="tc_slider_unique" # <--- THIS FIXES THE ERROR
    )
    
    return tc_input / 100

def render_verification_logs(transformer_logs, regime_logs):
    tab1, tab2 = st.tabs(["Transformer Logs", "Regime Switcher Logs"])
    with tab1:
        st.dataframe(transformer_logs, use_container_width=True)
    with tab2:
        st.dataframe(regime_logs, use_container_width=True)
