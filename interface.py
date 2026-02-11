import streamlit as st
import pandas as pd

def render_comparison_dashboard(transformer_results, regime_results, sofr_rate):
    st.title("🏔️ Alpha Engine ver1.0")
    
    # Thursday, 12th February 2026 Header
    st.markdown(f"### 📅 Market Forecast: Thursday, 12th February 2026")
    st.caption(f"Risk-Free Rate (SOFR): **{sofr_rate:.2%}**")

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
        "Metric": ["Predicted ETF", "Optimal Holding Period", "Annualised Return (%)", f"Sharpe Ratio (SOFR @ {sofr_rate:.2%})", "Hit Ratio (15 Day)", "Hit Ratio (30 Day)"],
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
    st.subheader("📝 Prediction Logs (Last 15 Days)")
    
    def color_coding(val):
        color = '#2ecc71' if val > 0 else '#e74c3c' # Green for profit, Red for loss
        return f'color: {color}; font-weight: bold'

    tab1, tab2 = st.tabs(["Transformer Details", "Regime Switcher Details"])
    
    with tab1:
        if not transformer_df.empty:
            st.dataframe(transformer_df.style.applymap(color_coding, subset=['Prediction']), width='stretch')
    
    with tab2:
        if not regime_df.empty:
            st.dataframe(regime_df.style.applymap(color_coding, subset=['Prediction']), width='stretch')
