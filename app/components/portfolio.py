import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import config

def render(df: pd.DataFrame):
    st.markdown("<h2 style='margin-bottom: 1.5rem;'>Portfolio Risk Intelligence</h2>", unsafe_allow_html=True)
    
    # ----------------------------------------------------
    # 1. Top KPI Metric Row
    # ----------------------------------------------------
    total_customers = len(df)
    portfolio_def_rate = (df['TARGET'] == 1).mean() * 100
    mean_health = df['health'].mean()
    red_tier_count = (df['ews_tier'] == 'Red').sum()
    
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric(
            label="Total Customers Monitored", 
            value=f"{total_customers:,}", 
            delta="Active Portfolio"
        )
    with kpi_cols[1]:
        st.metric(
            label="Portfolio Default Rate (Actual)", 
            value=f"{portfolio_def_rate:.2f}%", 
            delta="Historical Target Baseline",
            delta_color="off"
        )
    with kpi_cols[2]:
        st.metric(
            label="Mean Credit Health Score", 
            value=f"{mean_health:.1f}", 
            delta="Range: 300 - 850"
        )
    with kpi_cols[3]:
        st.metric(
            label="Red-Tier Early Warning Alert", 
            value=f"{red_tier_count:,}", 
            delta=f"{(red_tier_count/total_customers*100):.1f}% of portfolio",
            delta_color="inverse"
        )
    
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # ----------------------------------------------------
    # 2. Score Band Histogram & EWS Tier Donut Chart
    # ----------------------------------------------------
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("### Credit Health Score Distribution")
        
        # Color mapping for bands
        color_map = {
            "Excellent": "#2ecc71",
            "Good": "#3498db",
            "Fair": "#f1c40f",
            "Watch": "#e67e22",
            "Critical": "#e74c3c"
        }
        
        # Histogram of health score colored by band
        # To make it render nicely, we sort categories so legend is ordered
        df_sorted_band = df.copy()
        df_sorted_band['score_band'] = pd.Categorical(
            df_sorted_band['score_band'], 
            categories=["Excellent", "Good", "Fair", "Watch", "Critical"], 
            ordered=True
        )
        
        fig_hist = px.histogram(
            df_sorted_band,
            x="health",
            color="score_band",
            color_discrete_map=color_map,
            nbins=50,
            labels={"health": "Health Score", "count": "Customer Count"},
            category_orders={"score_band": ["Excellent", "Good", "Fair", "Watch", "Critical"]},
            opacity=0.85
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Credit Band",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, title_font=dict(color="#8a8d9a"), tickfont=dict(color="#8a8d9a")),
            yaxis=dict(showgrid=True, gridcolor="#1a1d29", title_font=dict(color="#8a8d9a"), tickfont=dict(color="#8a8d9a"))
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_right:
        st.markdown("### Portfolio Risk Tiers (EWS)")
        tier_counts = df['ews_tier'].value_counts().reset_index()
        tier_colors = {
            "Green": "#2ecc71",
            "Yellow": "#f1c40f",
            "Amber": "#e67e22",
            "Red": "#e74c3c"
        }
        
        fig_donut = px.pie(
            tier_counts,
            values="count",
            names="ews_tier",
            hole=0.6,
            color="ews_tier",
            color_discrete_map=tier_colors,
            category_orders={"ews_tier": ["Green", "Yellow", "Amber", "Red"]}
        )
        fig_donut.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        fig_donut.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#0e1117', width=2))
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # ----------------------------------------------------
    # 3. Model Calibration (PD Deciles vs Actual Defaults)
    # ----------------------------------------------------
    col_bottom_left, col_bottom_right = st.columns([3, 2])
    
    with col_bottom_left:
        st.markdown("### PD Calibration: Model Predicted vs. Actual Defaults")
        
        # Calculate PD deciles
        # Use qcut to bucket PD into 10 groups
        df_cal = df.copy()
        df_cal['pd_decile'] = pd.qcut(df_cal['pd'], 10, labels=False, duplicates='drop') + 1
        
        cal_summary = df_cal.groupby('pd_decile').agg(
            mean_pred_pd=('pd', 'mean'),
            actual_default_rate=('TARGET', 'mean')
        ).reset_index()
        
        # Multiply rates by 100 for percentages
        cal_summary['mean_pred_pd'] *= 100
        cal_summary['actual_default_rate'] *= 100
        
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Bar(
            x=cal_summary['pd_decile'],
            y=cal_summary['mean_pred_pd'],
            name="Average Predicted PD",
            marker_color="#00c2ff",
            opacity=0.8
        ))
        fig_cal.add_trace(go.Bar(
            x=cal_summary['pd_decile'],
            y=cal_summary['actual_default_rate'],
            name="Actual Default Rate",
            marker_color="#2ecc71",
            opacity=0.8
        ))
        fig_cal.update_layout(
            barmode='group',
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Probability of Default Decile (1 = Lowest Risk, 10 = Highest)", title_font=dict(color="#8a8d9a"), tickfont=dict(color="#8a8d9a")),
            yaxis=dict(title="Rate (%)", showgrid=True, gridcolor="#1a1d29", title_font=dict(color="#8a8d9a"), tickfont=dict(color="#8a8d9a")),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_cal, use_container_width=True)
        
    with col_bottom_right:
        st.markdown("### Credit Band Risk Metrics")
        
        # Summarize by score band
        band_summary = df.groupby('score_band', observed=False).agg(
            count=('SK_ID_CURR', 'count'),
            avg_health=('health', 'mean'),
            avg_pd=('pd', 'mean'),
            actual_default=('TARGET', 'mean')
        ).reindex(["Excellent", "Good", "Fair", "Watch", "Critical"])
        
        # Format columns
        band_summary['count'] = band_summary['count'].map('{:,}'.format)
        band_summary['avg_health'] = band_summary['avg_health'].map('{:.1f}'.format)
        band_summary['avg_pd'] = (band_summary['avg_pd'] * 100).map('{:.2f}%'.format)
        band_summary['actual_default'] = (band_summary['actual_default'] * 100).map('{:.2f}%'.format)
        
        # Rename for presentation
        band_summary.columns = ["Active Accounts", "Avg Health Score", "Avg Predicted PD", "Actual Default Rate"]
        
        st.dataframe(band_summary, use_container_width=True)
        
        st.markdown("""
        <div style='background-color: #1a1d29; padding: 1rem; border-radius: 8px; border: 1px solid #282a36; font-size: 0.85rem; color: #8a8d9a;'>
            <b>Calibration Note:</b> A perfectly calibrated risk model will show actual default rates closely matching or tracing the predicted PDs. In decile 10 (highest risk), the default rate rises sharply, demonstrating excellent risk segmentation.
        </div>
        """, unsafe_allow_html=True)
