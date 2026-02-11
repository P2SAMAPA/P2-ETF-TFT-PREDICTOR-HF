import streamlit as st
import pandas as pd

def render_comparison_dashboard(transformer_results, regime_results, sofr_rate):
    st.title("🏔️ Alpha Engine ver1.0")
    
    # Header with US Markets Context
    st.markdown(f"### 📅 US Markets Open: Thursday, 12th February 2026")
    st.info(f"Current Risk-Free Rate (SOFR): **{sofr_rate:.2%}**")

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
        "Metric": ["Predicted ETF", "Optimal Holding Period", "Annualised Return (%)", f"Sharpe Ratio (vs SOFR)", "Hit Ratio (15 Day)", "Hit Ratio (30 Day)"],
        "Transformer (Attention)": [
            transformer_results['ticker'], transformer_results['horizon'],
            f"{transformer_results['ann_return']:.2%}", f"{transformer_results['sharpe']:.2f}",
            f"{transformer_results['hit_15']:.1%}", f"{transformer_results['hit_30']:.1%}"
        ],
        "Regime Switcher (XGB/RF)": [
            regime_results['ticker'], regime_results['horizon'],
            f"{regime_results['ann_return']:.2%}", f"{regime_results['sharpe']:.2f}",
            f"{regime_results['hit_15']:.1%}", f"{regime_results['hit_30']:.1%}"
        ]
    }
    st.table(pd.DataFrame(comparison_data))

def render_tactical_logs(transformer_df, regime_df):
    st.divider()
    st.subheader("📝 Prediction Logs (Last 15 Trading Days)")
    
    def color_pnl(val):
        color = '#2ecc71' if val > 0 else '#e74c3c'
        return f'color: {color}; font-weight: bold'

    tab1, tab2 = st.tabs(["Transformer Details", "Regime Switcher Details"])
    
    with tab1:
        st.dataframe(transformer_df.style.applymap(color_pnl, subset=['Prediction']), width='stretch')
    with tab2:
        st.dataframe(regime_df.style.applymap(color_pnl, subset=['Prediction']), width='stretch')
