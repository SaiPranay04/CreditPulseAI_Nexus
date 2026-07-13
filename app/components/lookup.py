import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import config

# Business-friendly names for the features to show on the SHAP explanation chart
FEATURE_LABELS = {
    # Application base features
    'EXT_MEAN': 'Average External Bureau Score',
    'EXT_SOURCE_2': 'External Source 2 Score',
    'EXT_SOURCE_3': 'External Source 3 Score',
    'EXT_SOURCE_1': 'External Source 1 Score',
    'CREDIT_INCOME_RATIO': 'Credit-to-Income Leverage Ratio',
    'ANNUITY_INCOME_RATIO': 'Annuity-to-Income Repayment Ratio',
    'PAYMENT_CREDIT_RATIO': 'Annuity-to-Credit Ratio',
    'INCOME_PER_PERSON': 'Income per Family Member',
    'DAYS_BIRTH': 'Age in Years (at Application)',
    'DAYS_EMPLOYED': 'Years Employed',
    'EMPLOYED_AGE_RATIO': 'Employment Duration relative to Age',
    'AMT_INCOME_TOTAL': 'Annual Income',
    'AMT_CREDIT': 'Requested Loan Credit Limit',
    'AMT_ANNUITY': 'Requested Loan Annual Annuity',
    'EMPLOYED_ANOM': 'Employment Anomaly Flag',
    
    # Bureau features
    'BUREAU_LOAN_COUNT': 'Total Active & Closed Bureau Credits',
    'BUREAU_ACTIVE_COUNT': 'Active Bureau Credit Count',
    'BUREAU_DAYS_CREDIT_MEAN': 'Average Age of Bureau Accounts',
    'BUREAU_DAYS_CREDIT_MAX': 'Days Since Most Recent Bureau Loan',
    'BUREAU_CREDIT_SUM': 'Total Bureau Credit Borrowed',
    'BUREAU_CREDIT_SUM_DEBT': 'Total Active Bureau Debt',
    'BUREAU_DEBT_CREDIT_RATIO': 'Overall Bureau Debt-to-Credit Ratio',
    'BUREAU_CREDIT_DAY_OVERDUE_MEAN': 'Average Overdue Days at Bureau',
    'BUREAU_DPD_RATIO': 'Bureau DPD Occurrence Frequency',
    
    # Prev applications
    'PREV_COUNT': 'Total Prior Applications',
    'PREV_REFUSED_RATIO': 'Prior Application Refusal Rate',
    'PREV_APPROVED_RATIO': 'Prior Application Approval Rate',
    'PREV_CREDIT_APP_RATIO_MEAN': 'Credit Requested vs. Offered Ratio',
    'PREV_DAYS_DECISION_MEAN': 'Average Days Since Prior Decisions',
    
    # Installments features
    'INST_LATE_RATIO': 'Late Payment Rate (Installments)',
    'INST_LATE_DAYS_MEAN': 'Average Days Payments are Delayed',
    'INST_UNDERPAY_RATIO': 'Underpayment Frequency (Paid < Due)',
    'INST_PAYMENT_INSTALMENT_RATIO': 'Repayment Rate (Sum Paid / Sum Due)',
    'INST_COUNT': 'Total Historical Installments Due',
    
    # POS cash features
    'POS_DPD_MEAN': 'Average Days Past Due on POS Cash',
    'POS_DPD_POS_RATIO': 'Late POS Payments Frequency',
    'POS_MONTHS_COUNT': 'POS Cash Balance History Months',
    'POS_COMPLETED_COUNT': 'Completed POS Contracts Count',
    
    # Credit Card features
    'CC_UTILIZATION_MEAN': 'Average Credit Card Utilization',
    'CC_UTILIZATION_MAX': 'Peak Credit Card Utilization',
    'CC_DRAWINGS_LIMIT_RATIO': 'Drawings relative to Credit Limit',
    'CC_MIN_PAYMENT_MISS_RATIO': 'Missed Minimum Card Payment Rate',
    'CC_BALANCE_TREND': 'Card Balance Growth Trend',
    'CC_MONTHS_COUNT': 'Credit Card History Months',
    'CC_DPD_MEAN': 'Average Days Past Due on Credit Card',
    'CC_ATM_DRAWINGS_RATIO': 'ATM Cash Withdrawal Share',
}

def render(df: pd.DataFrame, shap_df: pd.DataFrame):
    st.markdown("<h2 style='margin-bottom: 1.5rem;'>Individual Customer Risk Analysis</h2>", unsafe_allow_html=True)
    
    # Example lists for ease of use
    defaults = [100002, 100031, 100047, 100083]
    healthies = [100003, 100004, 100006, 100009]
    
    col_sel_1, col_sel_2 = st.columns([2, 3])
    
    with col_sel_1:
        # User input for ID
        selected_id_input = st.text_input(
            "Search Customer ID (SK_ID_CURR):", 
            value="100002",
            help="Enter a unique 6-digit Customer ID to retrieve their credit report."
        )
        
    with col_sel_2:
        # Quick samples
        sample_choice = st.selectbox(
            "Quick Select Example Profile:",
            options=["None", "High Risk / Default Profiles", "Low Risk / Healthy Profiles"],
            index=0
        )
        
        if sample_choice == "High Risk / Default Profiles":
            st.info(f"High risk example IDs: {defaults}")
        elif sample_choice == "Low Risk / Healthy Profiles":
            st.success(f"Low risk example IDs: {healthies}")
            
    # Parse selected ID
    try:
        cust_id = int(selected_id_input)
    except ValueError:
        st.error("Customer ID must be a numeric integer.")
        st.stop()
        
    if cust_id not in df.index:
        st.error(f"Customer ID {cust_id} not found in the portfolio.")
        st.markdown("**Search Tips:** Try one of the test profiles like `100002` (High Risk) or `100003` (Low Risk).")
        st.stop()
        
    # Get customer data
    cust = df.loc[cust_id]
    
    # ----------------------------------------------------
    # Layout Grid: Left (Gauge/Details), Right (SHAP Drivers)
    # ----------------------------------------------------
    col_left, col_right = st.columns([2, 3])
    
    with col_left:
        st.markdown("### Risk & Health Summary")
        
        # Color mapping for bands
        band_colors = {
            "Excellent": "#2ecc71",
            "Good": "#3498db",
            "Fair": "#f1c40f",
            "Watch": "#e67e22",
            "Critical": "#e74c3c"
        }
        color = band_colors.get(cust['score_band'], "#e74c3c")
        
        # Plotly indicator gauge for health score
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = int(cust['health']),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Credit Band: {cust['score_band']}", 'font': {'size': 20, 'color': color}},
            gauge = {
                'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "#8a8d9a"},
                'bar': {'color': color},
                'bgcolor': "#1a1d29",
                'borderwidth': 2,
                'bordercolor': "#282a36",
                'steps': [
                    {'range': [300, 450], 'color': '#3a1315'},
                    {'range': [450, 550], 'color': '#3d2511'},
                    {'range': [550, 650], 'color': '#3f3914'},
                    {'range': [650, 750], 'color': '#11293a'},
                    {'range': [750, 850], 'color': '#0d2d14'}
                ]
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=10),
            height=260,
            font={'color': '#ffffff', 'family': "Outfit"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # 1. PD Box
        st.markdown(f"""
            <div style='background-color: #1a1d29; border: 1px solid #282a36; padding: 1.2rem; border-radius: 10px; margin-bottom: 1.5rem; text-align: center;'>
                <p style='color: #8a8d9a; margin: 0 0 0.2rem 0; font-size: 0.9rem; text-transform: uppercase;'>Probability of Default (PD)</p>
                <h1 style='font-size: 2.8rem; margin: 0; color: {color};'>{(cust['pd']*100):.2f}%</h1>
                <p style='color: #8a8d9a; margin: 0.3rem 0 0 0; font-size: 0.85rem;'>Target Risk Threshold: {config.EWS_RULES['HIGH_PD'][2]*100:.0f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 2. EWS Triggered Rules Card
        st.markdown("### Early Warning Alerts")
        rules_triggered = []
        for r_name in config.EWS_RULES.keys():
            if cust[r_name] == 1:
                rules_triggered.append(r_name)
                
        tier_color_map = {
            "Green": "#2ecc71",
            "Yellow": "#f1c40f",
            "Amber": "#e67e22",
            "Red": "#e74c3c"
        }
        tier_col = tier_color_map.get(cust['ews_tier'], "#2ecc71")
        
        # Render Warning Tier Badge
        st.markdown(f"""
            <div style='background-color: #1a1d29; border-left: 5px solid {tier_col}; padding: 1rem; border-radius: 6px; border-top: 1px solid #282a36; border-right: 1px solid #282a36; border-bottom: 1px solid #282a36; margin-bottom: 1rem;'>
                <h4 style='margin: 0; color: #ffffff; display: inline;'>EWS Alert Level: </h4>
                <span style='color: {tier_col}; font-weight: 700; font-size: 1.1rem;'>{cust['ews_tier']}</span>
                <p style='color: #8a8d9a; margin: 0.3rem 0 0 0; font-size: 0.85rem;'>Triggered {cust['ews_rules_fired']} of {len(config.EWS_RULES)} risk limits</p>
            </div>
        """, unsafe_allow_html=True)
        
        if len(rules_triggered) > 0:
            for rule in rules_triggered:
                # Format rule nicely
                desc = config.EWS_RULES[rule]
                # Format standard rules for business interpretation
                st.markdown(f"🚨 **{rule}**: `{desc[0]}` {desc[1]} `{desc[2]}`")
        else:
            st.success("✅ Clean Record — No Early Warning limits breached.")
            
    with col_right:
        st.markdown("### Top Credit Score Drivers (Local SHAP)")
        
        # Filter SHAP records for this customer
        cust_shap = shap_df[shap_df['SK_ID_CURR'] == cust_id].copy()
        
        if len(cust_shap) == 0:
            st.warning("No SHAP explainers calculated for this customer.")
        else:
            # Map features to friendly names
            cust_shap['feature_label'] = cust_shap['feature'].map(lambda x: FEATURE_LABELS.get(x, x))
            
            # Sort by absolute SHAP value for visualization ordering (largest at top)
            cust_shap['abs_val'] = cust_shap['shap_value'].abs()
            cust_shap = cust_shap.sort_values('abs_val', ascending=True)
            
            # Set bar colors: Red for pushing towards default (positive SHAP), Green for away (negative SHAP)
            cust_shap['color'] = np.where(cust_shap['shap_value'] >= 0, '#e74c3c', '#2ecc71')
            
            # Horizontal bar plot
            fig_shap = go.Figure()
            fig_shap.add_trace(go.Bar(
                y=cust_shap['feature_label'],
                x=cust_shap['shap_value'],
                orientation='h',
                marker_color=cust_shap['color'],
                hovertext=[f"Feature value: {v:.4f}" for v in cust_shap['feature_value']],
                opacity=0.85
            ))
            
            fig_shap.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(
                    title="SHAP Value (Impact on Default Risk)", 
                    title_font=dict(color="#8a8d9a"), 
                    tickfont=dict(color="#8a8d9a"),
                    showgrid=True,
                    gridcolor="#1a1d29"
                ),
                yaxis=dict(
                    tickfont=dict(color="#ffffff", size=11),
                    showgrid=False
                )
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            
            st.markdown("""
            <div style='background-color: #1b1314; border: 1px solid #3c191c; padding: 0.8rem; border-radius: 6px; font-size: 0.82rem; color: #e6a8ac;'>
                👉 <b>How to read:</b> <span style='color: #e74c3c; font-weight: bold;'>Red bars</span> indicate features that increased this customer's risk of default. <span style='color: #2ecc71; font-weight: bold;'>Green bars</span> indicate positive features that lowered their default risk. Hover over a bar to view its actual feature value.
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br/><hr style='border: 0; border-top: 1px solid #1a1d29;'/><br/>", unsafe_allow_html=True)
    
    # ----------------------------------------------------
    # 3. Customer Profile details
    # ----------------------------------------------------
    st.markdown("### Detailed Customer Risk Profile")
    
    # Format some of the columns nicely
    # Convert age and employment years
    age_yrs = -cust['DAYS_BIRTH'] / 365.25
    emp_yrs = -cust['DAYS_EMPLOYED'] / 365.25 if not pd.isna(cust['DAYS_EMPLOYED']) else 0.0
    
    # Display in 4 clean columns
    profile_cols = st.columns(4)
    
    with profile_cols[0]:
        st.markdown("**Financial Ratios**")
        st.markdown(f"💰 **Income**: ₹{cust['AMT_INCOME_TOTAL']:,.2f}")
        st.markdown(f"💳 **Credit Limit**: ₹{cust['AMT_CREDIT']:,.2f}")
        st.markdown(f"📈 **Annuity Due**: ₹{cust['AMT_ANNUITY']:,.2f}")
        st.markdown(f"⚖️ **Credit/Income Ratio**: {cust['CREDIT_INCOME_RATIO']:.2f}")
        
    with profile_cols[1]:
        st.markdown("**Personal Details**")
        st.markdown(f"👤 **Age**: {int(age_yrs)} years")
        st.markdown(f"💼 **Employment**: {emp_yrs:.1f} years" if emp_yrs > 0 else "💼 **Employment**: Unemployed / Retired")
        st.markdown(f"👪 **Income/Person**: ₹{cust['INCOME_PER_PERSON']:,.2f}")
        st.markdown(f"🔗 **Age-Employed Ratio**: {(emp_yrs/age_yrs*100):.1f}%" if emp_yrs > 0 else "🔗 **Age-Employed Ratio**: 0.0%")
        
    with profile_cols[2]:
        st.markdown("**Credit History**")
        st.markdown(f"🔍 **Avg Bureau Score**: {cust['EXT_MEAN']:.4f}")
        st.markdown(f"📊 **Annuity/Income Ratio**: {cust['ANNUITY_INCOME_RATIO']:.2f}")
        st.markdown(f"📉 **Payment/Credit Ratio**: {cust['PAYMENT_CREDIT_RATIO']:.4f}")
        
    with profile_cols[3]:
        st.markdown("**Early Warning Flags**")
        st.markdown(f"🚨 **Rules Fired**: {cust['ews_rules_fired']}")
        st.markdown(f"📈 **Late Installment Rate**: {(cust['INST_LATE_RATIO']*100):.1f}%" if not pd.isna(cust['INST_LATE_RATIO']) else "📈 **Late Installment Rate**: N/A")
        st.markdown(f"❌ **Prior Refusal Rate**: {(cust['PREV_REFUSED_RATIO']*100):.1f}%" if not pd.isna(cust['PREV_REFUSED_RATIO']) else "❌ **Prior Refusal Rate**: N/A")
        st.markdown(f"💳 **Max Card Util**: {(cust['CC_UTILIZATION_MAX']*100):.1f}%" if not pd.isna(cust['CC_UTILIZATION_MAX']) else "💳 **Max Card Util**: No Credit Card")
