import streamlit as st
import pandas as pd

def render_comparison_dashboard(transformer_results, regime_results):
    st.title("🏔️ Alpha Engine ver1.0")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🤖 Model A: Transformer")
        st.metric("Top Pick", f"{transformer_results['ticker']} ({transformer_results['horizon']})")
    
    with col2:
        st.subheader("📊 Model B: Regime Switcher")
        st.metric("Top Pick", f"{regime_results['ticker']} ({regime_results['horizon']})")

    st.divider()
    st.subheader("🏆 Strategy Performance Comparison (Out-of-Sample)")
    
    comparison_data = {
        "Metric": ["Predicted ETF", "Optimal Hold", "Ann. Return", "Sharpe Ratio", "Hit Ratio 15D", "Hit Ratio 30D"],
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

def render_verification_logs(transformer_logs, regime_logs):
    st.divider()
    st.subheader("📝 Prediction Logs (Last 15 Days)")
    tab1, tab2 = st.tabs(["Transformer Details", "Regime Switcher Details"])
    with tab1:
        # Updated to use width='stretch' per 2026 Streamlit standards
        st.dataframe(transformer_logs, width='stretch')
    with tab2:
        st.dataframe(regime_logs, width='stretch')
