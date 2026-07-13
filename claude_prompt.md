# Claude Prompt — CreditPulse AI: Final Golden Pipeline

> Copy everything below the line and paste into Claude.

---

## CONTEXT

I am building a **hackathon prototype** for **IDBI Innovate 2026** (a banking innovation challenge). The project is a **Loan Default Prediction Platform** called **CreditPulse AI**. I need to submit a working GitHub repository with a prototype.

## DATASET

I am using the **Home Credit Default Risk** dataset (7 relational CSV tables, ~307K customers). Here's what I already know from my analysis:

### Table Summary

| Table | Rows | Columns | Join Key |
|---|---|---|---|
| `application_train.csv` | 307,511 | 122 | SK_ID_CURR (primary) |
| `application_test.csv` | 48,744 | 121 | SK_ID_CURR |
| `bureau.csv` | 1,716,428 | 17 | SK_ID_CURR → SK_ID_BUREAU |
| `bureau_balance.csv` | 27,299,925 | 3 | SK_ID_BUREAU |
| `previous_application.csv` | 1,670,214 | 37 | SK_ID_CURR → SK_ID_PREV |
| `POS_CASH_balance.csv` | 10,001,358 | 8 | SK_ID_PREV, SK_ID_CURR |
| `installments_payments.csv` | 13,605,401 | 8 | SK_ID_PREV, SK_ID_CURR |
| `credit_card_balance.csv` | 3,840,312 | 23 | SK_ID_PREV, SK_ID_CURR |

### Data Quality Issues (Already Diagnosed)

1. **Class Imbalance**: TARGET: 0=282,686 (91.93%), 1=24,825 (8.07%). Ratio ~11.4:1
2. **Massive Missing Values**:
   - Building info columns (COMMONAREA, NONLIVINGAPARTMENTS, etc.): ~70% missing
   - FLOORSMIN, YEARS_BUILD: ~67% missing
   - OWN_CAR_AGE: ~66% missing
   - LANDAREA: ~59% missing
   - EXT_SOURCE_1: ~40% missing (but highly predictive!)
   - EXT_SOURCE_3: ~20% missing
3. **Anomaly in DAYS_EMPLOYED**: 55,374 rows have value 365243 (placeholder for "not employed") — must flag and replace with NaN
4. **Data types**: 106 numeric columns, 16 categorical columns (string/object type)
5. **Key stats**: Income mean=168K, Credit mean=599K, Annuity mean=27K, Age range ~20-69 years (DAYS_BIRTH)
6. **Coverage across tables**: Bureau=85.7%, Previous apps=94.6%, POS=94.1%, Installments=94.8%, Credit card=28.3% (sparse!)
7. **Bureau balance STATUS**: Encoded as C(closed), X(unknown), 0(no DPD), 1-5 (DPD buckets 1-30, 31-60, etc.)
8. **Previous apps**: 62% Approved, 19% Cancelled, 17% Refused, 2% Unused
9. **Installments**: ~8.4% of payments are late, ~9.6% are underpaid

### Feature Engineering Already Done (in my notebook)

My notebook already builds a **162-column master table** by merging application_train with 4 aggregated auxiliary tables:

**Bureau aggregates** (per SK_ID_CURR):
- BUREAU_LOAN_COUNT, BUREAU_ACTIVE_COUNT, BUREAU_CLOSED_COUNT
- BUREAU_CREDIT_SUM, BUREAU_CREDIT_SUM_DEBT, BUREAU_CREDIT_SUM_OVERDUE
- BUREAU_MAX_OVERDUE, BUREAU_DAYS_CREDIT_MEAN/MIN
- BUREAU_LATE_MONTHS_SUM, BUREAU_MAX_STATUS_EVER
- BUREAU_CREDIT_TO_DEBT_RATIO

**Previous application aggregates**:
- PREV_APP_COUNT, PREV_APPROVED/REFUSED/CANCELED_COUNT
- PREV_AMT_APPLICATION/CREDIT/ANNUITY/DOWN_PAYMENT_MEAN
- PREV_DAYS_DECISION_MEAN/MIN, PREV_CNT_PAYMENT_MEAN
- PREV_REFUSAL_RATIO, PREV_APPROVAL_RATIO

**Installment aggregates**:
- INSTALL_COUNT, INSTALL_DAYS_LATE_MEAN/MAX
- INSTALL_LATE_COUNT, INSTALL_PAYMENT_DIFF_MEAN
- INSTALL_UNDERPAID_COUNT, INSTALL_LATE_RATIO

**POS Cash aggregates**:
- POS_COUNT, POS_CNT_INSTALMENT_MEAN/FUTURE_MEAN
- POS_DPD_MEAN/MAX, POS_DPD_DEF_MEAN
- POS_LATE_COUNT, POS_LATE_RATIO

**NOT YET aggregated**: `credit_card_balance.csv` (only 28.3% coverage but still valuable)

## WHAT I WANT YOU TO PRODUCE

Give me a **complete, step-by-step golden pipeline** that I can directly use to build the hackathon prototype. This is being submitted as a **GitHub repo** with a working demo.

### Requirements:

1. **This is a PROTOTYPE**, not production. Keep it achievable in 2-3 days by one developer.
2. The code will run locally on a Windows machine with Python 3.11, 16GB RAM.
3. Must work with the 7 CSV files (total ~2.7GB) — optimize for memory.
4. Must produce a **Streamlit dashboard** as the demo frontend.

### Pipeline Stages I Need (give me each as a separate, detailed step):

#### Stage 1: Data Loading & Preprocessing
- Memory-optimized loading (downcasting dtypes)
- DAYS_EMPLOYED anomaly fix (365243 → NaN + flag)
- Missing value strategy for each column group
- Drop 70% missing building columns vs keep with flag

#### Stage 2: Feature Engineering
- Expand my existing 162 features to ~300+ using:
  - Credit card balance aggregation (I'm missing this!)
  - More ratios: ANNUITY_INCOME_RATIO, CREDIT_INCOME_RATIO, CREDIT_GOODS_RATIO
  - Domain-specific: PAYMENT_RATE, DAYS_EMPLOYED_RATIO, INCOME_PER_FAMILY_MEMBER
  - Trend/slope features from temporal tables (not just mean/max)
  - Interaction features between bureau + application data
- Tell me EXACTLY which new features to create, with the formula

#### Stage 3: Preprocessing for ML
- Encoding strategy for 16 categorical columns (target encoding vs label encoding — be specific per column)
- Why NOT to use SMOTE, and what to use instead (cost-sensitive weights)
- Feature selection: which 70% missing columns to drop, which to keep
- Handle the 28.3% credit card coverage gap (NaN fill strategy)

#### Stage 4: Model Training
- **LightGBM** as primary model (give me exact hyperparameters for this dataset)
- **XGBoost** as secondary (exact hyperparameters)
- **Stacking ensemble** with Logistic Regression meta-learner
- 5-Fold Stratified CV with early stopping
- Exact scale_pos_weight calculation for imbalance
- Seed averaging (3 seeds)
- What AUC to realistically expect (the Kaggle competition gold was ~0.80)

#### Stage 5: Explainability
- SHAP summary plot (global)
- SHAP waterfall plot (per-customer)
- Top 10 risk factors per customer
- Counterfactual explanations: "What would reduce this customer's risk?"

#### Stage 6: Financial Health Score
- Composite 0-100 score using subscores:
  - Credit Behavior (from bureau data)
  - Repayment Discipline (from installments)
  - Debt Burden (from application ratios)
  - Financial Stability (employment, income, assets)
  - Liquidity (credit card utilization)
- Give me the exact formula/weights

#### Stage 7: Early Warning System
- Rule-based alerts derived from the data patterns I found:
  - Late payment streaks
  - High credit utilization
  - Bureau deterioration
  - Refused application history
  - Social circle defaults (DEF_30/60_CNT_SOCIAL_CIRCLE)
- Threshold values for each alert

#### Stage 8: Streamlit Dashboard
- 4-tab layout:
  - **Portfolio Overview**: Risk distribution, default rate, key metrics
  - **Customer Lookup**: Enter SK_ID_CURR → see PD, health score, SHAP waterfall, EWS alerts
  - **Early Warning**: Table of highest-risk customers with alert reasons
  - **Model Performance**: ROC curve, PR curve, confusion matrix, feature importance
- Premium dark-mode UI design
- Must work with the trained model artifacts (saved .pkl files)

#### Stage 9: GitHub Repo Structure
- Give me the exact folder structure
- README.md template (what to write for judges)
- requirements.txt
- How to make it reproducible (data download instructions, etc.)

### CONSTRAINTS:
- Do NOT use SMOTE or any oversampling technique
- Do NOT use neural networks or deep learning (no GPU available, keep it achievable for prototype)
- DO use LightGBM + XGBoost stacking (proven, fast, interpretable)
- DO use SHAP for explainability
- DO use Streamlit for dashboard
- Keep total training time under 30 minutes on CPU
- All code must be in Python scripts (not notebooks) for clean GitHub submission
- Use `joblib` to save/load models

### OUTPUT FORMAT:
Give me the response as:
1. **GitHub repo folder structure** (tree format)
2. **Step-by-step pipeline** with exact code snippets for each stage
3. **Key hyperparameters** table
4. **Expected results** (AUC range, training time)
5. **README.md content** for the repo

Focus on being PRACTICAL and BUILDABLE. I'd rather have 80% of the features working perfectly than 100% half-broken. This is a hackathon prototype — polish matters more than complexity.
