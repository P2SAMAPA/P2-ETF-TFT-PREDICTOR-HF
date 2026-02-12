import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

def render_sidebar():
    st.sidebar.header("Model Configuration")
    regime_year = st.sidebar.select_slider("Data Anchor", options=[2008, 2015, 2019, 2021], value=2015)
    tx_cost_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 100, 15)
    return regime_year, tx_cost_bps

def color_returns(val):
    if isinstance(val, str) and '%' in val:
        try:
            num = float(val.replace('%', ''))
            color = '#00d4ff' if num > 0 else '#fb7185'
            return f'color: {color}'
        except: return ''
    return ''

def render_main_output(top_pick, ann_return, sharpe, hit_rate, top_horizon, wealth, audit_df, oos_returns):
    st.markdown(f"### 🤖 Transformer Strategy Cycle: **{datetime.now().strftime('%B %d, %Y')}**")
    
    # Row 1: Metrics
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("TOP PREDICTION", top_pick)
    with c2: 
        st.metric("ANNUALIZED RETURN (OOS)", f"{ann_return}%")
        st.caption(f"↑ {sharpe} Sharpe Ratio")
    with c3: st.metric("15-DAY HIT RATIO", f"{hit_rate:.0%}")
    st.divider()

    # Row 2: Signal Box and Chart
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="padding:40px; border-radius:10px; border:2px solid #00d4ff; background-color:#0e1117; text-align:center;">
            <h1 style="color:#00d4ff; margin:0; font-size:100px;">{top_pick}</h1>
            <p style="font-size:24px; color:#8892b0; letter-spacing: 2px;">MODEL: TRANSFORMER (1D)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("OOS Cumulative Return")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wealth.index, y=wealth, mode='lines', line=dict(color='#00d4ff', width=3)))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Monthly Performance
    st.subheader("📅 Monthly Performance Attribution (OOS)")
    monthly = oos_returns.groupby([oos_returns.index.year, oos_returns.index.month]).apply(lambda x: (1 + x).prod() - 1)
    monthly_df = monthly.unstack()
    monthly_df.columns = [datetime(2000, m, 1).strftime('%b') for m in monthly_df.columns]
    monthly_df['Annual Total'] = (1 + monthly_df).prod(axis=1) - 1
    
    fmt_df = monthly_df.map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    st.table(fmt_df.style.map(color_returns))

    # Row 4: Audit
    st.subheader("🔍 Verification Log (Last 15 Trading Days)")
    st.table(audit_df.style.map(color_returns))
