import streamlit as st
import pandas as pd

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Engine Settings")
        friction = st.slider("Transaction Friction (bps)", 0, 100, 15)
        st.info("📅 Training: 2018-2026\nAudit Window: Last 15 Days")
        return friction

def render_dashboard(res_a, res_b, sofr):
    st.title("🏔️ Alpha Engine v1.3")
    st.caption(f"Target: Highest Total Return | Audit Period: Last 15 Trading Days")

    # Comparison Table
    comparison_data = {
        "Metric": ["Top Pick", "Period", "Annualised Return", "Hit Ratio (15D)"],
        "Model A (Transformer)": [res_a['ticker'], res_a['horizon'], f"{res_a['ann_return']:.2%}", f"{res_a['hit_15']:.1%}"],
        "Model B (Best of XGB/RF)": [res_b['ticker'], res_b['horizon'], f"{res_b['ann_return']:.2%}", f"{res_b['hit_15']:.1%}"]
    }
    st.table(pd.DataFrame(comparison_data))

    # OOS Cumulative Return Chart
    st.subheader("📈 Cumulative Audit Performance (Total Return)")
    cum_df = pd.DataFrame({
        "Model A": (1 + res_a['logs']['Actual Return']).cumprod(),
        "Model B": (1 + res_b['logs']['Actual Return']).cumprod()
    })
    st.line_chart(cum_df)

def color_pnl(val):
    if isinstance(val, (int, float)):
        if val > 0: return 'background-color: #d4edda; color: #155724' # Green
        if val < 0: return 'background-color: #f8d7da; color: #721c24' # Red
    return ''

def render_logs(df_a, df_b):
    st.divider()
    t1, t2 = st.tabs(["Model A (Transformer) Audit", "Model B (Regime) Audit"])
    with t1:
        st.dataframe(df_a.style.applymap(color_pnl, subset=['Predicted', 'Actual Return']), use_container_width=True)
    with t2:
        st.dataframe(df_b.style.applymap(color_pnl, subset=['Predicted', 'Actual Return']), use_container_width=True)
