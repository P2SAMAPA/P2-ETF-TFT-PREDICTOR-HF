import streamlit as st
import pandas as pd

def render_comparison_dashboard(transformer_results, regime_results, sofr_rate):
    st.title("🏔️ Alpha Engine ver1.0")
    st.markdown("### 📅 US Markets: Thursday, 12th February 2026")
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
        "Metric": ["Predicted ETF", "Optimal Period", "Annualised Return", "Sharpe Ratio", "Hit Ratio (15D)"],
        "Transformer (Attention)": [
            transformer_results['ticker'], transformer_results['horizon'],
            f"{transformer_results['ann_return']:.2%}", f"{transformer_results['sharpe']:.2f}",
            f"{transformer_results['hit_15']:.1%}"
        ],
        "Regime Switcher (XGB/RF)": [
            regime_results['ticker'], regime_results['horizon'],
            f"{regime_results['ann_return']:.2%}", f"{regime_results['sharpe']:.2f}",
            f"{regime_results['hit_15']:.1%}"
        ]
    }
    st.table(pd.DataFrame(comparison_data))

def render_tactical_logs(transformer_df, regime_df):
    st.divider()
    st.subheader("📝 Reality Check: Prediction vs. Market Truth")
    
    def highlight_reality(row):
        # A 'Hit' occurs only if Prediction and Actual move in the SAME direction
        pred = row['Predicted Return']
        act = row['Actual Return']
        is_hit = (pred > 0 and act > 0) or (pred < 0 and act < 0)
        color = 'background-color: #d4edda; color: #155724' if is_hit else 'background-color: #f8d7da; color: #721c24'
        return [color] * len(row)

    tab1, tab2 = st.tabs(["Transformer Accuracy", "Regime Switcher Accuracy"])
    with tab1:
        st.dataframe(transformer_df.style.apply(highlight_reality, axis=1), use_container_width=True)
    with tab2:
        st.dataframe(regime_df.style.apply(highlight_reality, axis=1), use_container_width=True)
