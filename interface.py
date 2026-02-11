import streamlit as st
import pandas as pd

def render_main_output(top_pick, sharpe, win_rate, ann_return, timeframe, wealth_curve, audit_log, train_info):
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("🕹️ Strategy Controls")
    
    # Re-adding the Transaction Cost Slider
    tc_pct = st.sidebar.slider(
        "Transaction Cost / Slippage (%)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.05,
        help="Estimate of total cost per trade (Commissions + Bid/Ask Spread)."
    )
    
    st.sidebar.divider()
    st.sidebar.write(f"**Model:** PPO Deep Neural Net")
    st.sidebar.write(f"**Training:** {train_info['start']} to {train_info['end']}")

    # --- MAIN UI ---
    st.title("🏔️ Alpha Engine v12.1")
    
    col1, col2, col3 = st.columns(3)
    
    # Adjusting Annualized Return for Transaction Costs
    # (Simplified: assumes 1 trade per week on average for tactical rotation)
    est_trades_per_year = 52
    net_ann_return = float(ann_return.strip('%')) / 100 - (tc_pct/100 * est_trades_per_year)

    with col1:
        st.metric("CURRENT ROTATION", top_pick)
        st.caption("Updated: Live Market Data")

    with col2:
        st.metric("Sharpe Ratio", sharpe)
        st.caption("Benchmark: Live SOFR")

    with col3:
        # Displaying the "Net" return after costs
        st.metric("Net Ann. Return (Est)", f"{net_ann_return:.2%}", 
                  delta=f"-{(tc_pct/100 * est_trades_per_year):.1%} cost drag", 
                  delta_color="inverse")

    st.divider()

    # Performance Graph
    st.subheader(f"Equity Curve: {top_pick} (Last 60D OOS)")
    st.line_chart(wealth_curve)

    # Verification Table
    st.write("#### Historical Verification Log (Last 15 Trading Days)")
    st.table(audit_log.head(15))

    return tc_pct # Returning this in case app.py needs to use it in its logic
