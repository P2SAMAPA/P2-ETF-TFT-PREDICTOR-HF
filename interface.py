import streamlit as st
import pandas as pd

def render_sidebar(start_year, end_year):
    with st.sidebar:
        st.header("⚙️ Engine Settings")
        friction = st.slider("Transaction Friction (bps)", 0, 100, 15)
        st.info(f"📅 Data Range: {start_year}-{end_year}\nSplit: 80% Train / 20% OOS Check")
        st.divider()
        return friction

def render_dashboard(res_a, res_b):
    st.title("🏔️ Alpha Engine v1.5")
    st.caption("Objective: Maximum Total Return | Out-of-Sample (20% Test Split)")

    # Metrics Table
    comparison_data = {
        "Metric": ["Top Pick", "Period", "Annualised Return", "Hit Ratio"],
        "Model A (Transformer)": [res_a['ticker'], res_a['horizon'], f"{res_a['ann_return']:.2%}", f"{res_a['hit_oos']:.1%}"],
        "Model B (Best of XGB/RF)": [res_b['ticker'], res_b['horizon'], f"{res_b['ann_return']:.2%}", f"{res_b['hit_oos']:.1%}"]
    }
    st.table(pd.DataFrame(comparison_data))

    # OOS Cumulative Return Chart
    st.subheader("📈 Cumulative Return (20% OOS Period)")
    cum_df = pd.DataFrame({
        "Model A": (1 + res_a['logs']['Actual Return']).cumprod(),
        "Model B": (1 + res_b['logs']['Actual Return']).cumprod()
    })
    st.line_chart(cum_df)

def color_pnl(val):
    """Auditor's Color Logic: Red for Loss, Green for Profit"""
    if isinstance(val, (int, float)):
        return 'background-color: #d4edda; color: #155724' if val > 0 else 'background-color: #f8d7da; color: #721c24'
    return ''

def render_logs(df_a, df_b):
    st.divider()
    t1, t2 = st.tabs(["Model A (Transformer) OOS Log", "Model B (Regime) OOS Log"])
    with t1:
        st.dataframe(df_a.style.applymap(color_pnl, subset=['Predicted', 'Actual Return']), use_container_width=True)
    with t2:
        st.dataframe(df_b.style.applymap(color_pnl, subset=['Predicted', 'Actual Return']), use_container_width=True)
