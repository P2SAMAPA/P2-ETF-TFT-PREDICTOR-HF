import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

def render_sidebar():
    st.sidebar.header("Model Configuration")
    regime_year = st.sidebar.select_slider("Data Anchor", options=[2008, 2015, 2019, 2021], value=2015)
    tx_cost_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 100, 15)
    return regime_year, tx_cost_bps

def render_main_output(top_pick, sharpe, hit_rate, ann_return, top_horizon, wealth, audit_df):
    st.markdown(f"### 🔥 High-Beta Strategy Cycle: **{datetime.now().strftime('%B %d, %Y')}**")
    
    # Row 1: Metrics
    c1, c2, c3 = st.columns(3)
    
    with c1: 
        st.metric("TOP PREDICTION", top_pick)
    
    with c2: 
        # UI updated: font-weight changed to normal
        st.markdown(f"""
            <div style="line-height: 1.2;">
                <p style="font-size: 14px; color: #8892b0; margin-bottom: 0px; text-transform: uppercase;">Annualised Returns (OOS)</p>
                <p style="font-size: 38px; font-weight: normal; margin: 0px;">{ann_return}</p>
                <p style="font-size: 14px; color: #8892b0; margin-top: -5px;">Sharpe Ratio: {sharpe}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with c3: 
        st.metric("15-DAY HIT RATIO", f"{hit_rate:.0%}")
    
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
        st.subheader("OOS Cumulative Return")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wealth.index, y=wealth, mode='lines', line=dict(color='#00d4ff', width=3)))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), template="plotly_dark", yaxis_title="Growth of $1")
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Audit Table
    st.subheader("🔍 Verification Log (Last 15 Trading Days)")
    st.table(audit_df.style.map(lambda x: f"color: {'#00d4ff' if float(str(x).strip('%')) > 0 else '#fb7185'}", subset=['Net Return']))
