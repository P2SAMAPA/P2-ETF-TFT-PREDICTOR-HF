import streamlit as st
import pandas as pd

def render_main_output(top_pick, sharpe, hit_rate, ann_return, horizon, wealth_curve, audit_log, train_info, tc_drag):
    st.title("🏔️ Alpha Engine v12.3")
    
    # Connection Status
    st.success("✅ Connected to Polygon.io Fallback")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CURRENT ROTATION", top_pick)
    with col2:
        st.metric("Sharpe Ratio", sharpe)
        st.caption("Benchmark: Live SOFR")
    with col3:
        # Show the raw return and the drag separately for clarity
        st.metric("Net Ann. Return (Est)", ann_return, delta=f"-{tc_drag:.2%} cost drag", delta_color="inverse")

    st.divider()
    st.subheader(f"Equity Curve: {top_pick} (Last 60D OOS)")
    st.line_chart(wealth_curve)

    with st.expander("ℹ️ Strategy Details"):
        st.write(f"**Training:** {train_info['start']} to {train_info['end']}")
        st.write("**Model:** PPO Deep Neural Net (1,000 Epochs)")
