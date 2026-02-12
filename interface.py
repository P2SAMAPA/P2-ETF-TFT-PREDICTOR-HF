import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

def render_sidebar():
    st.sidebar.header("Model Configuration")
    regime_year = st.sidebar.select_slider("Data Anchor", options=[2008, 2015, 2019, 2021], value=2015)
    tx_cost_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 100, 15)
    return regime_year, tx_cost_bps

def color_returns(val):
    if isinstance(val, str) and '%' in val:
        num = float(val.replace('%', ''))
        color = '#00d4ff' if num > 0 else '#fb7185'
        return f'color: {color}'
    return ''

def render_main_output(top_pick, ann_return, sharpe, hit_rate, top_horizon, wealth, audit_df, returns_df):
    st.markdown(f"### 🔥 High-Beta Strategy Cycle: **{datetime.now().strftime('%B %d, %Y')}**")
    
    # Row 1: Metrics (Fixed Layout)
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
            <p style="font-size:24px; color:#8892b0; letter-spacing: 2px;">HOLDING PERIOD: {top_horizon}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Cumulative Return (OOS period)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wealth.index, y=wealth, mode='lines', line=dict(color='#00d4ff', width=3)))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), template="plotly_dark", yaxis_title="Growth of $1")
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Monthly Returns Table (Geometric Compounding)
    st.subheader("📅 Monthly Performance Attribution (OOS)")
    # Group by Year/Month
    monthly = returns_df.groupby([returns_df.index.year, returns_df.index.month]).apply(lambda x: (1 + x).prod() - 1)
    monthly_matrix = monthly.unstack()
    monthly_matrix.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly_matrix.columns)]
    
    # Add Annual Geometric Total
    monthly_matrix['Annual'] = (1 + monthly_matrix).prod(axis=1) - 1
    
    # Formatting
    fmt_matrix = monthly_matrix.applymap(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    st.table(fmt_matrix.style.applymap(color_returns))

    # Row 4: Audit Table
    st.subheader("🔍 Verification Log (Last 15 Trading Days)")
    st.table(audit_df.style.applymap(color_returns))
