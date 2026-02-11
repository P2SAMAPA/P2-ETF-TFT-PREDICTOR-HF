import streamlit as st
import pandas as pd

def render_main_output(prediction, sharpe, hit_ratio, ann_return, horizon, wealth_curve, audit_df):
    # Header
    st.title(f"🔥 High-Beta Strategy Cycle: {pd.Timestamp.now().strftime('%B %d, %Y')}")
    
    # Top Row Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("TOP PREDICTION", prediction)
    with col2:
        st.metric("ANNUALISED RETURN (2025-26 OOS)", ann_return)
        st.caption(f"Sharpe (vs SOFR): {sharpe}")
    with col3:
        st.metric("HIT RATIO (2025+)", f"{hit_ratio:.0%}")

    st.markdown("---")

    # Main Dashboard Area
    left_col, right_col = st.columns([1, 1.5])

    with left_col:
        st.write("")
        st.container(border=True).inner_content = st.markdown(
            f"""
            <div style="background-color: #0e1117; padding: 60px; border-radius: 10px; border: 2px solid #00d4ff; text-align: center;">
                <h1 style="font-size: 100px; color: #00d4ff; margin: 0;">{prediction}</h1>
                <p style="font-size: 24px; color: #8892b0; letter-spacing: 2px;">HOLDING PERIOD: {horizon}</p>
            </div>
            """, unsafe_allow_html=True
        )

    with right_col:
        st.subheader("OOS Performance: Jan 2025 - Present")
        st.line_chart(wealth_curve, height=400, use_container_width=True)

    st.markdown("---")

    # Corrected Verification Log Header
    st.subheader(f"🔍 Verification Log (Last {len(audit_df)} Trading Days)")
    
    # Styled Table
    st.table(audit_df.style.map(
        lambda x: f"color: {'#00d4ff' if '-' not in str(x) else '#fb7185'}", 
        subset=['Net Return']
    ))
