import streamlit as st
import pandas as pd

def render_sidebar(df):
    with st.sidebar:
        st.header("⚙️ Engine Settings")
        friction = st.slider("Transaction Friction (bps)", 0, 100, 15)
        
        # Dynamic Date Calculation for the 80/20 split
        start_date = df.index.min().strftime('%Y')
        end_date = df.index.max().strftime('%Y')
        oos_start = df.index[int(len(df)*0.8)].strftime('%Y-%m-%d')
        
        st.info(f"📅 Data: {start_date}-{end_date}\nOOS Check Starts: {oos_start}")
        return friction

def render_dashboard(res_a, res_b):
    st.title("🏔️ Alpha Engine v1.6")
    
    # Metrics Table
    comparison_data = {
        "Metric": ["Top Pick", "Period", "Ann. Return", "Hit Ratio"],
        "Model A (Transformer)": [res_a['ticker'], res_a['horizon'], f"{res_a['ann_return']:.2%}", f"{res_a['hit_oos']:.1%}"],
        "Model B (Best of B1/B2)": [res_b['ticker'], res_b['horizon'], f"{res_b['ann_return']:.2%}", f"{res_b['hit_oos']:.1%}"]
    }
    st.table(pd.DataFrame(comparison_data))

    # FIXED: Cumulative Return using Log-Sum for stability
    st.subheader("📈 Total Return (20% OOS Period)")
    
    # We create a clean index to ensure both lines show up
    combined_results = pd.merge(
        res_a['logs'][['Actual Return']].rename(columns={'Actual Return': 'Model A'}),
        res_b['logs'][['Actual Return']].rename(columns={'Actual Return': 'Model B'}),
        left_index=True, right_index=True, how='outer'
    ).fillna(0)
    
    # Calculate equity curve: (1 + r).cumprod()
    equity_curve = (1 + combined_results).cumprod()
    st.line_chart(equity_curve)

def color_pnl(val):
    if isinstance(val, (int, float)):
        return 'color: #155724; background-color: #d4edda;' if val > 0 else 'color: #721c24; background-color: #f8d7da;'
    return ''

def render_logs(df_a, df_b):
    st.subheader("📝 Detailed Audit Trail (Ticker-Level)")
    t1, t2 = st.tabs(["Model A (Transformer)", "Model B (Regime)"])
    
    # Ensure Ticker is the first column after Date
    with t1:
        st.dataframe(df_a.style.applymap(color_pnl, subset=['Predicted', 'Actual Return']), use_container_width=True)
    with t2:
        st.dataframe(df_b.style.applymap(color_pnl, subset=['Predicted', 'Actual Return']), use_container_width=True)
