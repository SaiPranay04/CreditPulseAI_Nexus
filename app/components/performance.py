import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Retrieve the same labels map for consistency
from app.components.lookup import FEATURE_LABELS

def render(metrics: dict, shap_global: pd.DataFrame):
    st.markdown("<h2 style='margin-bottom: 1.5rem;'>Model Stacking & Validation Performance</h2>", unsafe_allow_html=True)
    
    # ----------------------------------------------------
    # 1. Headline Statistics Row
    # ----------------------------------------------------
    train_time_min = metrics.get('total_time_seconds', 0) / 60
    opt_thresh = metrics.get('optimal_threshold', 0.5)
    f1 = metrics.get('f1_score', 0) * 100
    gini = metrics.get('gini', 0) * 100
    
    perf_cols = st.columns(4)
    with perf_cols[0]:
        st.metric(
            label="Stacked Ensemble OOF AUC", 
            value=f"{metrics.get('stacked_oof_auc', 0):.4f}", 
            delta=f"Gini Index: {gini:.1f}%"
        )
    with perf_cols[1]:
        st.metric(
            label="Optimal Decision Threshold", 
            value=f"{opt_thresh:.4f}", 
            delta="Maximizes F1-Score"
        )
    with perf_cols[2]:
        st.metric(
            label="Peak F1-Score Achieve", 
            value=f"{f1:.2f}%", 
            delta=f"Precision: {metrics.get('precision', 0)*100:.1f}% | Recall: {metrics.get('recall', 0)*100:.1f}%",
            delta_color="off"
        )
    with perf_cols[3]:
        st.metric(
            label="Training Wall Clock Time", 
            value=f"{train_time_min:.2f} mins", 
            delta="GPU Accelerated Stack"
        )
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # ----------------------------------------------------
    # 2. AUC Comparisons & ROC Curve
    # ----------------------------------------------------
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.markdown("### Model Stacking Uplift (Out-of-Fold AUC)")
        
        # Build comparison df
        auc_data = {
            'Model Name': ['LightGBM', 'XGBoost', 'CatBoost', 'Stacked Ensemble (Meta)'],
            'OOF AUC': [
                metrics.get('lgb_oof_auc', 0),
                metrics.get('xgb_oof_auc', 0),
                metrics.get('cat_oof_auc', 0),
                metrics.get('stacked_oof_auc', 0)
            ]
        }
        df_auc = pd.DataFrame(auc_data)
        
        fig_auc = px.bar(
            df_auc,
            x='Model Name',
            y='OOF AUC',
            text='OOF AUC',
            color='Model Name',
            color_discrete_map={
                'LightGBM': '#3498db',
                'XGBoost': '#e67e22',
                'CatBoost': '#9b59b6',
                'Stacked Ensemble (Meta)': '#2ecc71'
            }
        )
        fig_auc.update_traces(
            texttemplate='%{text:.4f}', 
            textposition='outside',
            marker=dict(line=dict(color='#0e1117', width=1))
        )
        fig_auc.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(range=[0.70, 0.80], showgrid=True, gridcolor="#1a1d29", title_font=dict(color="#8a8d9a"), tickfont=dict(color="#8a8d9a")),
            xaxis=dict(title_font=dict(color="#8a8d9a"), tickfont=dict(color="#8a8d9a")),
            showlegend=False
        )
        st.plotly_chart(fig_auc, use_container_width=True)
        
    with col_r:
        st.markdown("### Stacked Ensemble ROC Curve")
        
        # Load ROC points
        roc_pts = metrics.get('roc_points', [])
        if len(roc_pts) > 0:
            df_roc = pd.DataFrame(roc_pts)
            
            fig_roc = go.Figure()
            # ROC line
            fig_roc.add_trace(go.Scatter(
                x=df_roc['fpr'],
                y=df_roc['tpr'],
                mode='lines',
                name=f"Stacked Stacker (AUC = {metrics.get('stacked_oof_auc', 0):.4f})",
                line=dict(color='#2ecc71', width=3)
            ))
            # Diagonal random line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name="Random Baseline",
                line=dict(color='#8a8d9a', width=1, dash='dash')
            ))
            
            fig_roc.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(title="False Positive Rate (FPR)", title_font=dict(color="#8a8d9a"), tickfont=dict(color="#8a8d9a"), showgrid=True, gridcolor="#1a1d29"),
                yaxis=dict(title="True Positive Rate (TPR)", title_font=dict(color="#8a8d9a"), tickfont=dict(color="#8a8d9a"), showgrid=True, gridcolor="#1a1d29"),
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.warning("No ROC curve points found in metrics.json")
            
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # ----------------------------------------------------
    # 3. Cross-Validation Folds & Confusion Matrix
    # ----------------------------------------------------
    col_fold, col_cm = st.columns([4, 3])
    
    with col_fold:
        st.markdown("### 5-Fold Cross-Validation Scores")
        
        folds_list = metrics.get('folds', [])
        if len(folds_list) > 0:
            df_folds = pd.DataFrame(folds_list)
            # Rename columns nicely
            df_folds.columns = ["Fold ID", "LightGBM AUC", "XGBoost AUC", "CatBoost AUC", "Fold Training Time (s)"]
            
            # Format numbers
            df_folds["LightGBM AUC"] = df_folds["LightGBM AUC"].map('{:.4f}'.format)
            df_folds["XGBoost AUC"] = df_folds["XGBoost AUC"].map('{:.4f}'.format)
            df_folds["CatBoost AUC"] = df_folds["CatBoost AUC"].map('{:.4f}'.format)
            df_folds["Fold Training Time (s)"] = df_folds["Fold Training Time (s)"].map('{:.1f}s'.format)
            
            st.dataframe(df_folds, use_container_width=True, hide_index=True)
        else:
            st.warning("No cross-validation fold details available.")
            
    with col_cm:
        st.markdown("### Confusion Matrix (Decision Boundary)")
        
        cm = metrics.get('confusion_matrix', {})
        if len(cm) > 0:
            tn, fp, fn, tp = cm.get('tn', 0), cm.get('fp', 0), cm.get('fn', 0), cm.get('tp', 0)
            
            # Beautiful HTML representation of Confusion Matrix
            st.markdown(f"""
                <table style="width:100%; text-align:center; border-collapse: collapse; font-family:'Outfit', sans-serif; background-color:#1a1d29; border: 1px solid #282a36; border-radius:10px;">
                    <tr>
                        <td style="border: 1px solid #282a36; padding: 1rem; font-weight: bold; background-color:#0b0d13;"></td>
                        <td style="border: 1px solid #282a36; padding: 1rem; font-weight: bold; background-color:#0b0d13; color:#2ecc71;">PREDICTED: HEALTHY (0)</td>
                        <td style="border: 1px solid #282a36; padding: 1rem; font-weight: bold; background-color:#0b0d13; color:#e74c3c;">PREDICTED: DEFAULT (1)</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #282a36; padding: 1rem; font-weight: bold; background-color:#0b0d13; color:#2ecc71;">ACTUAL: HEALTHY (0)</td>
                        <td style="border: 1px solid #282a36; padding: 1.5rem; background-color:#15291b;">
                            <span style="font-size:1.3rem; font-weight:bold; color:#2ecc71;">{tn:,}</span><br/>
                            <span style="font-size:0.8rem; color:#8a8d9a;">True Negative (Correct)</span>
                        </td>
                        <td style="border: 1px solid #282a36; padding: 1.5rem; background-color:#321f20;">
                            <span style="font-size:1.3rem; font-weight:bold; color:#e74c3c;">{fp:,}</span><br/>
                            <span style="font-size:0.8rem; color:#8a8d9a;">False Positive (Type I)</span>
                        </td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #282a36; padding: 1rem; font-weight: bold; background-color:#0b0d13; color:#e74c3c;">ACTUAL: DEFAULT (1)</td>
                        <td style="border: 1px solid #282a36; padding: 1.5rem; background-color:#321f20;">
                            <span style="font-size:1.3rem; font-weight:bold; color:#e74c3c;">{fn:,}</span><br/>
                            <span style="font-size:0.8rem; color:#8a8d9a;">False Negative (Type II)</span>
                        </td>
                        <td style="border: 1px solid #282a36; padding: 1.5rem; background-color:#15291b;">
                            <span style="font-size:1.3rem; font-weight:bold; color:#2ecc71;">{tp:,}</span><br/>
                            <span style="font-size:0.8rem; color:#8a8d9a;">True Positive (Correct)</span>
                        </td>
                    </tr>
                </table>
            """, unsafe_allow_html=True)
        else:
            st.warning("No confusion matrix statistics found.")
            
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # ----------------------------------------------------
    # 4. Global SHAP Feature Importances
    # ----------------------------------------------------
    st.markdown("### Top 20 Global Portfolio Drivers (SHAP Feature Importance)")
    
    # Take top 20
    top_shap = shap_global.head(20).copy()
    top_shap['feature_label'] = top_shap['feature'].map(lambda x: FEATURE_LABELS.get(x, x))
    top_shap = top_shap.sort_values('importance', ascending=True)
    
    fig_global = px.bar(
        top_shap,
        y="feature_label",
        x="importance",
        orientation="h",
        color_discrete_sequence=["#00c2ff"],
        opacity=0.85
    )
    fig_global.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Mean Absolute SHAP Value (Global Impact)", title_font=dict(color="#8a8d9a"), tickfont=dict(color="#8a8d9a"), showgrid=True, gridcolor="#1a1d29"),
        yaxis=dict(tickfont=dict(color="#ffffff", size=10), showgrid=False)
    )
    st.plotly_chart(fig_global, use_container_width=True)
    
    # Note on CPU/GPU
    st.markdown(f"""
    <div style='background-color: #1a1d29; border: 1px solid #282a36; padding: 1rem; border-radius: 8px; font-size: 0.85rem; color: #8a8d9a;'>
        🏆 <b>Model Stacking Advantage:</b> Stacking combining LightGBM, XGBoost, and CatBoost achieves an Out-of-Fold AUC of <b>{metrics.get('stacked_oof_auc', 0):.4f}</b>. 
        The meta-learner automatically adjusts for the individual strengths of the tree architectures.
    </div>
    """, unsafe_allow_html=True)
