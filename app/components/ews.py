import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import config

def render(df: pd.DataFrame):
    st.markdown("<h2 style='margin-bottom: 1.5rem;'>Early Warning System (EWS) Registry</h2>", unsafe_allow_html=True)
    
    rule_cols = list(config.EWS_RULES.keys())
    
    # ----------------------------------------------------
    # 1. KPI Alert Row
    # ----------------------------------------------------
    green_count = (df['ews_tier'] == 'Green').sum()
    yellow_count = (df['ews_tier'] == 'Yellow').sum()
    amber_count = (df['ews_tier'] == 'Amber').sum()
    red_count = (df['ews_tier'] == 'Red').sum()
    
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown(f"""
            <div style='background-color: #1a1d29; border-left: 5px solid #2ecc71; padding: 1rem; border-radius: 8px; border-top: 1px solid #282a36; border-right: 1px solid #282a36; border-bottom: 1px solid #282a36;'>
                <p style='color: #8a8d9a; margin: 0; font-size: 0.85rem; text-transform: uppercase;'>Green (Low Risk)</p>
                <h2 style='margin: 0.2rem 0 0 0; color: #2ecc71;'>{green_count:,}</h2>
            </div>
        """, unsafe_allow_html=True)
    with kpi_cols[1]:
        st.markdown(f"""
            <div style='background-color: #1a1d29; border-left: 5px solid #f1c40f; padding: 1rem; border-radius: 8px; border-top: 1px solid #282a36; border-right: 1px solid #282a36; border-bottom: 1px solid #282a36;'>
                <p style='color: #8a8d9a; margin: 0; font-size: 0.85rem; text-transform: uppercase;'>Yellow (Watchlist)</p>
                <h2 style='margin: 0.2rem 0 0 0; color: #f1c40f;'>{yellow_count:,}</h2>
            </div>
        """, unsafe_allow_html=True)
    with kpi_cols[2]:
        st.markdown(f"""
            <div style='background-color: #1a1d29; border-left: 5px solid #e67e22; padding: 1rem; border-radius: 8px; border-top: 1px solid #282a36; border-right: 1px solid #282a36; border-bottom: 1px solid #282a36;'>
                <p style='color: #8a8d9a; margin: 0; font-size: 0.85rem; text-transform: uppercase;'>Amber (Medium Risk)</p>
                <h2 style='margin: 0.2rem 0 0 0; color: #e67e22;'>{amber_count:,}</h2>
            </div>
        """, unsafe_allow_html=True)
    with kpi_cols[3]:
        st.markdown(f"""
            <div style='background-color: #1a1d29; border-left: 5px solid #e74c3c; padding: 1rem; border-radius: 8px; border-top: 1px solid #282a36; border-right: 1px solid #282a36; border-bottom: 1px solid #282a36;'>
                <p style='color: #8a8d9a; margin: 0; font-size: 0.85rem; text-transform: uppercase;'>Red (Critical Alert)</p>
                <h2 style='margin: 0.2rem 0 0 0; color: #e74c3c;'>{red_count:,}</h2>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # ----------------------------------------------------
    # 2. Sidebar Filters (EWS Control Panel)
    # ----------------------------------------------------
    st.markdown("### Risk Screening Filters")
    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
    
    with col_f1:
        selected_tiers = st.multiselect(
            "Select EWS Risk Tiers:",
            options=["Green", "Yellow", "Amber", "Red"],
            default=["Red", "Amber"]
        )
    with col_f2:
        selected_bands = st.multiselect(
            "Select Credit score bands:",
            options=["Excellent", "Good", "Fair", "Watch", "Critical"],
            default=["Critical", "Watch", "Fair"]
        )
    with col_f3:
        min_rules = st.slider(
            "Minimum Fired Rules Limit:",
            min_value=0,
            max_value=len(config.EWS_RULES),
            value=1
        )
        
    # Apply filtering
    filtered_df = df[
        (df['ews_tier'].isin(selected_tiers)) &
        (df['score_band'].isin(selected_bands)) &
        (df['ews_rules_fired'] >= min_rules)
    ].copy()
    
    # Sort by probability of default (descending)
    filtered_df = filtered_df.sort_values('pd', ascending=False)
    
    # ----------------------------------------------------
    # 3. Rule frequency statistics
    # ----------------------------------------------------
    col_chart, col_table = st.columns([2, 3])
    
    with col_chart:
        st.markdown("### Systematic Risk Factor Frequency")
        # Sum rules fired across all customers
        rule_sums = df[rule_cols].sum(axis=0).reset_index()
        rule_sums.columns = ["EWS Rule", "Fired Count"]
        rule_sums = rule_sums.sort_values("Fired Count", ascending=True)
        
        fig_rules = px.bar(
            rule_sums,
            y="EWS Rule",
            x="Fired Count",
            orientation='h',
            color_discrete_sequence=['#00c2ff'],
            opacity=0.85
        )
        fig_rules.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Number of Breaches", title_font=dict(color="#8a8d9a"), tickfont=dict(color="#8a8d9a"), showgrid=True, gridcolor="#1a1d29"),
            yaxis=dict(tickfont=dict(color="#ffffff", size=10), showgrid=False)
        )
        st.plotly_chart(fig_rules, use_container_width=True)
        
    with col_table:
        st.markdown(f"### Screened Registry ({len(filtered_df):,} customers matched)")
        
        if len(filtered_df) == 0:
            st.warning("No customers matched the current filter combination.")
        else:
            # We construct a human-readable list of fired rules per row
            # To keep it extremely fast, we do this on a subset to show (e.g. top 1000)
            subset_show = filtered_df.head(100).copy()
            
            fired_rules_list = []
            for _, row in subset_show.iterrows():
                rules_triggered = [r for r in rule_cols if row[r] == 1]
                fired_rules_list.append(", ".join(rules_triggered) if rules_triggered else "None")
                
            subset_show['Fired Rules'] = fired_rules_list
            
            # Format and display
            show_df = subset_show[[
                'SK_ID_CURR', 'pd', 'health', 'score_band', 'ews_tier', 'ews_rules_fired', 'Fired Rules'
            ]].copy()
            
            # Formats
            show_df['pd'] = (show_df['pd'] * 100).map('{:.2f}%'.format)
            show_df['health'] = show_df['health'].map('{:d}'.format)
            show_df.columns = ["Customer ID", "PD (%)", "Health Score", "Credit Band", "EWS Tier", "Rules Fired", "Rules Triggered"]
            
            st.dataframe(show_df, use_container_width=True, hide_index=True)
            
            st.markdown("""
            <div style='background-color: #1a1d29; padding: 0.8rem; border-radius: 6px; font-size: 0.82rem; color: #8a8d9a; border: 1px solid #282a36;'>
                💡 <b>Investigator Tip:</b> Copy a <b>Customer ID</b> from the table above and paste it into the <b>Customer Lookup</b> tab to view their full credit profile and local SHAP drivers.
            </div>
            """, unsafe_allow_html=True)
