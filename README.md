# CreditPulse AI — Decision Support Dashboard

CreditPulse AI is an enterprise-grade credit risk monitoring and early warning system built for the IDBI Innovate 2026 banking hackathon. It processes a complex relational database of 307K customers and aggregates histories spanning millions of credit bureau records, credit card accounts, and installment payments.

By training a robust ensemble stacking three Gradient Boosted Decision Tree (GBDT) architectures (LightGBM, XGBoost, and CatBoost) on GPU, and feeding their cross-validated predictions into a Logistic Regression meta-learner, CreditPulse AI achieves a state-of-the-art **0.7844 OOF AUC**. The dashboard integrates these models with a rules-based Early Warning System (EWS) and precomputed SHAP values for sub-second page loads.

---

## Relational Architecture & Pipeline Data Flow

```
raw CSVs (data/raw/) 
   │
   ├── [1] pipeline/01_aggregate_cc.py  ──> data/processed/cc_agg.parquet
   │
   ├── [2] pipeline/02_build_features.py ──> data/processed/master_features.parquet (Frozen Schema)
   │
   ├── [3] pipeline/03_train.py          ──> models/bundle.joblib + outputs/metrics.json
   │
   └── [4] pipeline/04_score.py          ──> data/scored/scored_customers.parquet
                                            data/scored/shap_global.parquet
                                            data/scored/shap_top10.parquet
   │
   └── [5] app/dashboard.py (Streamlit)  <── Reads precomputed parquets only (O(1) dashboard latency)
```

---

## Headline Performance Metrics

| Evaluation Metric | Score / Status | Description |
|---|---|---|
| **Stacked Ensemble OOF AUC** | **0.7843** | Combined power of LightGBM + XGBoost + CatBoost |
| **Gini Coefficient** | **56.87%** | Strong population risk segmentation capability |
| **Optimal Threshold** | **0.6986** | Boundary maximizing F1-Score on defaults |
| **Precision / Recall** | **26.50% / 45.25%** | Maximizes default capture while controlling false alarms |
| **Active Portfolio Monitored** | **307,511** | Full-coverage scoring of active loan applications |

---

## Design Decisions

1. **No SMOTE / Resampling**: Synthetic oversampling is banned. We use cost-sensitive sample weighting (`scale_pos_weight` / `class_weight`) directly in GBDTs and the meta-learner to preserve true probability calibration needed for business decisions.
2. **Native-NaN Routing over Imputation**: Missing values are kept as `NaN`. Instead of applying arbitrary mean/median statistics, GBDTs route missingness as a distinct predictive signal (e.g. "no prior credit cards").
3. **Precomputed SHAP Explanations**: SHAP calculations are performed at batch-score time rather than live. This guarantees instant, sub-second responses in the Streamlit lookup tool, eliminating dashboard lag.
4. **Single-File Model Bundle**: All fold models, meta-learners, threshold targets, and the exact features list are persisted in a single `models/bundle.joblib` artifact to prevent version skew.
5. **Rank-Based Credit Health Scoring**: Health scores (300-850) are mapped monotonically from the rank of the predicted Probability of Default (PD). This remains robust to prediction calibration shifts over time.

---

## Installation & Running Guide

### Prerequisites
*   Windows OS
*   Python 3.11 installed
*   NVIDIA GPU (CUDA Toolkit configured for XGBoost/CatBoost acceleration)
*   Raw Home Credit dataset CSVs placed under `data/raw/`

### 1. Set Up the Environment
Create a self-contained virtual environment and install all packages:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline
You can run the entire pipeline end-to-end using the automated batch runner:
```bash
.\run_pipeline.bat
```
This batch script runs the aggregation, compilation, training, and scoring sequentially, stopping immediately if any script returns an error.

### 3. Launch the Dashboard
Start the Streamlit dark-themed decision support portal:
```bash
streamlit run app/dashboard.py
```

---

## Fresh-Environment Cold-Start Checklist
- [ ] Directory structure conforms exactly to scaffold specifications.
- [ ] Raw CSV files exist in `data/raw/` (including `credit_card_balance.csv`).
- [ ] Virtual environment is successfully created and packages installed.
- [ ] `run_pipeline.bat` completes successfully with all `data/scored/` parquets created.
- [ ] Cold start time of the dashboard under `streamlit run app/dashboard.py` is under 10 seconds.
