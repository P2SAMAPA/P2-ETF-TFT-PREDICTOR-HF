import streamlit as st
import pandas as pd

def render_main_output(prediction, sharpe, hit_ratio, ann_return, horizon, wealth_curve, audit_df, training_info):
    st.title(f"🚀 Alpha Engine v9: Tactical Overhaul")
    
    # Methodology & Dates Section
    with st.expander("ℹ️ Strategy Methodology & Training Details", expanded=False):
        st.write(f"**Training Period:** {training_info['start']} to {training_info['end']}")
        st.write(f"**OOS Test Period:** Last 60 Trading Days")
        st.markdown("""
        **Core Algorithm: PPO (Proximal Policy Optimization)**
        * **Clipped Updates:** PPO prevents the model from making radical strategy shifts based on single-day noise.
        * **Actor-Critic Architecture:** The 'Actor' chooses the ETF, while the 'Critic' evaluates if that choice actually reduced risk.
        
        **Advanced Features Added:**
        * **Rolling Z-Score Scaling:** Normalizes features based on a 60-day moving window to catch local peaks.
        * **Macro Integration:** Ingests 10Y2Y Treasury spreads and MOVE index to sense credit stress.
        * **Momentum Buffers:** RSI and MACD signals are used to penalize 'Overbought' holdings.
        """)

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("TOP PICK", prediction)
    with col2: st.metric("ANNUALISED (Last 60D)", ann_return)
    with col3: st.metric("SHARPE (Last 60D)", sharpe)

    st.markdown("---")
    
    left_col, right_col = st.columns([1, 1.5])
    with left_col:
        st.markdown(f"""
            <div style="background-color: #0e1117; padding: 40px; border-radius: 10px; border: 2px solid #ff4b4b; text-align: center;">
                <h1 style="font-size: 80px; color: #ff4b4b; margin: 0;">{prediction}</h1>
                <p style="font-size: 20px; color: #8892b0;">TACTICAL HORIZON: {horizon}</p>
            </div>
            """, unsafe_allow_html=True)
            
    with right_col:
        st.subheader("Last 60 Days Performance (OOS)")
        st.line_chart(wealth_curve, height=350)

    st.subheader("🔍 Verification Log (Last 45 Days)")
    st.table(audit_df.style.map(lambda x: f"color: {'#00d4ff' if '-' not in str(x) else '#fb7185'}", subset=['Net Return']))
