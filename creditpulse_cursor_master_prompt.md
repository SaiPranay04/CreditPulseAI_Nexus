# CreditPulse AI — Master Build Prompt for Cursor

**How to use:** Paste this ENTIRE document into Cursor as the opening prompt. Cursor must build STAGE BY STAGE in order, stopping at each ✅ CHECKPOINT for you to verify before continuing. Do not let it skip ahead.

---

## SYSTEM CONTEXT (Cursor: read fully before writing any code)

You are building **CreditPulse AI**, a loan default prediction platform for the IDBI Innovate 2026 banking hackathon. The architecture below is FINAL and LOCKED. Do not redesign, do not suggest alternatives, do not add features not specified. Build exactly what each stage says, then STOP and wait for the developer to verify the checkpoint before proceeding to the next stage.

### Hard Constraints (never violate)
- Windows, Python 3.11, 16GB RAM, CPU only. No GPU, no deep learning.
- Models: LightGBM + XGBoost, stacked with Logistic Regression meta-learner on OOF predictions.
- Class imbalance (11.4:1) handled via `scale_pos_weight` / `class_weight` — NO SMOTE, no resampling.
- All code as `.py` scripts. No notebooks.
- Model persistence via `joblib`. Data interchange via `parquet` (pyarrow).
- Total training time must stay under 30 minutes on CPU.
- Streamlit dashboard: dark mode, exactly 4 tabs — Portfolio Overview, Customer Lookup, Early Warning, Model Performance.
- SHAP explainability: PRECOMPUTED at batch-score time, never computed live in the dashboard.
- Memory discipline: downcast to float32/int32 immediately after every aggregation; `del` + `gc.collect()` after large merges; never merge raw child tables directly — always aggregate first.

### Dataset (Home Credit Default Risk, in `data/raw/`)
| Table | Rows | Join Key |
|---|---|---|
| application_train / test | 307,511 / 48,744 | SK_ID_CURR |
| bureau | 1,716,428 | SK_ID_CURR → SK_ID_BUREAU |
| bureau_balance | 27,299,925 | SK_ID_BUREAU |
| previous_application | 1,670,214 | SK_ID_CURR → SK_ID_PREV |
| POS_CASH_balance | 10,001,358 | SK_ID_PREV |
| installments_payments | 13,605,401 | SK_ID_PREV |
| credit_card_balance | 3,840,312 | SK_ID_PREV |

TARGET: 1 = default (8.07% positive). An existing 162-column master table already merges application_train with aggregated bureau, previous_application, installments, and POS cash features — Stage 2 will regenerate/extend it so the pipeline is fully reproducible from raw CSVs.

### Architecture (data flow — memorize this)
```
raw CSVs → 01_aggregate_cc.py → cc_agg.parquet
         → 02_build_features.py → data/processed/master_features.parquet  [SINGLE SOURCE OF TRUTH]
         → 03_train.py → models/bundle.joblib + outputs/metrics.json
         → 04_score.py → data/scored/scored_customers.parquet + shap_top10.parquet + shap_global.parquet
         → app/dashboard.py (reads ONLY scored parquets + bundle metadata; NO live inference)
```

---

## STAGE 0 — Project Scaffold + config.py

Create this exact folder tree:

```
creditpulse-ai/
├── README.md                 # placeholder for now; completed in Stage 6
├── requirements.txt
├── config.py
├── data/
│   ├── raw/                  # gitignored; 7 CSVs live here
│   ├── processed/
│   └── scored/
├── models/
├── outputs/
├── pipeline/
│   ├── 01_aggregate_cc.py
│   ├── 02_build_features.py
│   ├── 03_train.py
│   └── 04_score.py
├── app/
│   ├── dashboard.py
│   └── components/
│       ├── __init__.py
│       ├── portfolio.py
│       ├── lookup.py
│       ├── ews.py
│       └── performance.py
├── run_pipeline.bat          # runs 01→02→03→04 in order, aborts on any failure
└── .gitignore                # data/raw/, data/processed/, data/scored/, models/, __pycache__
```

`requirements.txt`: pandas, numpy, pyarrow, lightgbm, xgboost, scikit-learn, shap, joblib, streamlit, plotly.

`config.py` must contain (as plain Python constants — no YAML, no argparse):

```python
from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SCORED_DIR = ROOT / "data" / "scored"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

MASTER_FEATURES = PROCESSED_DIR / "master_features.parquet"
CC_AGG = PROCESSED_DIR / "cc_agg.parquet"
BUNDLE = MODELS_DIR / "bundle.joblib"
SCORED = SCORED_DIR / "scored_customers.parquet"
SHAP_TOP10 = SCORED_DIR / "shap_top10.parquet"
SHAP_GLOBAL = SCORED_DIR / "shap_global.parquet"
METRICS = OUTPUTS_DIR / "metrics.json"

# ---- Reproducibility ----
SEED = 42
N_FOLDS = 5

# ---- Missing data rule ----
DROP_MISSING_THRESHOLD = 0.60      # >60% missing → drop column
FLAG_MISSING_THRESHOLD = 0.20      # 20–60% → keep NaN + add _isna flag
DAYS_EMPLOYED_ANOMALY = 365243     # → NaN + EMPLOYED_ANOM flag

# ---- Model params (initial; tune only if train time < 30 min allows) ----
LGBM_PARAMS = dict(
    objective="binary", n_estimators=2000, learning_rate=0.03,
    num_leaves=34, colsample_bytree=0.85, subsample=0.85,
    reg_alpha=0.04, reg_lambda=0.07, min_child_samples=40,
    scale_pos_weight=11.4, n_jobs=-1, random_state=SEED, verbosity=-1,
)
XGB_PARAMS = dict(
    objective="binary:logistic", n_estimators=1200, learning_rate=0.05,
    max_depth=6, colsample_bytree=0.8, subsample=0.8,
    reg_alpha=0.05, reg_lambda=1.0, min_child_weight=30,
    scale_pos_weight=11.4, n_jobs=-1, random_state=SEED,
    tree_method="hist", eval_metric="auc",
)
EARLY_STOPPING_ROUNDS = 100

# ---- Health score: monotone map of PD → 300–850 band ----
# health = 850 - 550 * (rank_of_pd / n)   ← rank-based, robust to PD calibration
SCORE_MIN, SCORE_MAX = 300, 850
SCORE_BANDS = {          # inclusive lower bounds
    "Excellent": 750, "Good": 650, "Fair": 550, "Watch": 450, "Critical": 300,
}

# ---- Early Warning System rules (each fires independently; severity = count of fired rules) ----
EWS_RULES = {
    "HIGH_PD":            ("pd", ">=", 0.30),
    "ELEVATED_PD":        ("pd", ">=", 0.15),
    "CREDIT_OVERLOAD":    ("CREDIT_INCOME_RATIO", ">=", 6.0),
    "PAYMENT_STRAIN":     ("ANNUITY_INCOME_RATIO", ">=", 0.35),
    "LATE_HISTORY":       ("INST_LATE_RATIO", ">=", 0.20),
    "PRIOR_REFUSALS":     ("PREV_REFUSED_RATIO", ">=", 0.40),
    "CC_MAXED":           ("CC_UTILIZATION_MEAN", ">=", 0.85),
    "EMPLOYMENT_ANOMALY": ("EMPLOYED_ANOM", "==", 1),
}
EWS_SEVERITY = {"Red": 3, "Amber": 2, "Yellow": 1}   # min rules fired per tier

# ---- Dashboard ----
LOOKUP_INDEX = "SK_ID_CURR"
DARK_THEME = dict(bg="#0e1117", card="#1a1d29", accent="#00c2ff",
                  good="#2ecc71", warn="#f39c12", bad="#e74c3c", text="#e6e9ef")
```

> **Note to developer:** if a rule references a feature name that Stage 2 produces under a different name, fix the name in `EWS_RULES` — never rename the pipeline column to fit the rule.

**✅ CHECKPOINT 0:** Tree exists, `config.py` imports cleanly (`python -c "import config"`), `.gitignore` correct. STOP — wait for confirmation.

---

## STAGE 1 — `pipeline/01_aggregate_cc.py` (credit card aggregation)

Read `credit_card_balance.csv` in dtype-efficient mode (specify dtypes on read where possible). Aggregate to **one row per SK_ID_CURR** with EXACTLY these features (prefix all with `CC_`):

1. `CC_UTILIZATION_MEAN`, `CC_UTILIZATION_MAX` — AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL (guard divide-by-zero → NaN), then mean/max per customer
2. `CC_DRAWINGS_LIMIT_RATIO` — mean(AMT_DRAWINGS_CURRENT) / mean(AMT_CREDIT_LIMIT_ACTUAL)
3. `CC_MIN_PAYMENT_MISS_RATIO` — share of months where AMT_PAYMENT_CURRENT < AMT_INST_MIN_REGULARITY (both non-null)
4. `CC_BALANCE_TREND` — sign of (mean balance in 3 most recent months − mean balance in 3 oldest months) per customer: −1 / 0 / +1 (use MONTHS_BALANCE ordering)
5. `CC_MONTHS_COUNT` — number of monthly records
6. `CC_DPD_MEAN` — mean of SK_DPD
7. `CC_ATM_DRAWINGS_RATIO` — sum(AMT_DRAWINGS_ATM_CURRENT) / sum(AMT_DRAWINGS_CURRENT) (guard zero)

Downcast everything to float32/int32. Write `cc_agg.parquet`. Log rows in/out and peak feature count.
**Important:** only ~28% of customers have credit cards. Do NOT fill missing customers here — the left-join in Stage 2 produces NaN, which is signal (no card) and which LightGBM/XGBoost handle natively.

**✅ CHECKPOINT 1:** `cc_agg.parquet` exists, ~86K rows (one per card-holding customer), all `CC_` columns present. STOP.

---

## STAGE 2 — `pipeline/02_build_features.py` (master feature table)

This script rebuilds the full feature table from raw CSVs so the repo is reproducible end-to-end. Steps, in order, with `del`/`gc.collect()` between each block:

**2a. Base:** load application_train (train) and application_test, concat with an `IS_TRAIN` flag so all transforms apply identically to both.

**2b. Application-level cleanup + ratio features:**
- Replace DAYS_EMPLOYED == 365243 with NaN; add `EMPLOYED_ANOM` (int8 0/1).
- Ratio features: `CREDIT_INCOME_RATIO` = AMT_CREDIT/AMT_INCOME_TOTAL; `ANNUITY_INCOME_RATIO` = AMT_ANNUITY/AMT_INCOME_TOTAL; `PAYMENT_CREDIT_RATIO` = AMT_ANNUITY/AMT_CREDIT; `EMPLOYED_AGE_RATIO` = DAYS_EMPLOYED/DAYS_BIRTH; `INCOME_PER_PERSON` = AMT_INCOME_TOTAL/CNT_FAM_MEMBERS. Guard all divides.
- EXT_SOURCE combos: `EXT_MEAN`, `EXT_STD`, `EXT_PROD` (product of non-null, NaN if all null), `EXT_WEIGHTED` = 2·EXT_SOURCE_2 + 3·EXT_SOURCE_3 + EXT_SOURCE_1 computed on available values (skipna weighted). Do NOT impute EXT_SOURCE_1.

**2c. Bureau aggregation** (aggregate at SK_ID_CURR, prefix `BUREAU_`): counts of credits; count active (CREDIT_ACTIVE=="Active"); mean/max DAYS_CREDIT; sum AMT_CREDIT_SUM, sum AMT_CREDIT_SUM_DEBT; `BUREAU_DEBT_CREDIT_RATIO` = debt/credit sums; mean CREDIT_DAY_OVERDUE; count of distinct CREDIT_TYPE.
From bureau_balance (this is the 27M-row table — aggregate to SK_ID_BUREAU first, THEN join to bureau, THEN aggregate to SK_ID_CURR): `BUREAU_DPD_RATIO` = share of months with STATUS in {"1","2","3","4","5"}; `BUREAU_MONTHS_COUNT`. Read bureau_balance with usecols and category dtype for STATUS.

**2d. Previous application aggregation** (prefix `PREV_`): count; `PREV_REFUSED_RATIO` = share NAME_CONTRACT_STATUS=="Refused"; `PREV_APPROVED_RATIO`; mean AMT_APPLICATION; mean(AMT_CREDIT/AMT_APPLICATION) as `PREV_CREDIT_APP_RATIO`; mean CNT_PAYMENT; mean DAYS_DECISION.

**2e. Installments aggregation** (prefix `INST_`): `INST_LATE_RATIO` = share of rows where DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT; `INST_LATE_DAYS_MEAN` = mean(max(0, DAYS_ENTRY_PAYMENT − DAYS_INSTALMENT)); `INST_UNDERPAY_RATIO` = share where AMT_PAYMENT < AMT_INSTALMENT; `INST_PAYMENT_INSTALMENT_RATIO` = sum payments / sum instalments; `INST_COUNT`.

**2f. POS cash aggregation** (prefix `POS_`): `POS_DPD_MEAN`, `POS_DPD_POS_RATIO` (share of months SK_DPD>0), `POS_MONTHS_COUNT`, count of completed contracts (NAME_CONTRACT_STATUS=="Completed").

**2g. Merge:** left-join 2c–2f aggregates + `cc_agg.parquet` onto the base on SK_ID_CURR.

**2h. Missing-data rule (apply AFTER all merges, train-portion missingness only):**
- Missingness > `DROP_MISSING_THRESHOLD` → drop column (this removes the building-info block; keep TARGET/ID columns exempt).
- Between thresholds → keep NaN, add `{col}_isna` int8 flag.
- Below → keep NaN, no flag.

**2i. Categoricals:** label-encode object columns (fit on combined train+test), store the encoders' category lists inside the parquet metadata is NOT needed — just ensure deterministic encoding (sorted categories).

**2j. Downcast all floats → float32, ints → smallest safe int. Write `master_features.parquet`** containing SK_ID_CURR, TARGET (NaN for test rows), IS_TRAIN, and all features. Print: final shape, memory usage, count of `_isna` flags added, list of dropped columns.

**Explicitly SKIPPED by design (do not add):** trend/slope features over POS/bureau monthly histories beyond `CC_BALANCE_TREND`, pairwise interaction features, target encoding, model-based imputation. These are rejected for signal-to-effort reasons — do not "improve" the plan.

**✅ CHECKPOINT 2:** Parquet exists, ~180–200 columns, 356,255 rows (train+test), train rows have 8.07% TARGET mean, script peak RAM stayed sane. **SCHEMA IS NOW FROZEN — later stages must never rename or add feature columns.** STOP.

---

## STAGE 3 — `pipeline/03_train.py` (CV + stacking)

1. Load master features; split train rows (IS_TRAIN==1). Feature list = all columns except SK_ID_CURR, TARGET, IS_TRAIN. Persist this exact ordered list — it goes into the bundle.
2. `StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)`.
3. Per fold: train LightGBM with early stopping on the fold's validation AUC; store fold model + OOF predictions. Repeat for XGBoost. (Two loops or one loop training both — one loop preferred for memory.)
4. Stack: `LogisticRegression(class_weight="balanced", max_iter=1000)` trained on the 2-column OOF matrix [lgbm_oof, xgb_oof] → stacked OOF AUC.
5. Metrics → `outputs/metrics.json`: per-fold AUC (both models), OOF AUC (lgbm, xgb, stack), Gini, precision/recall/F1 at the chosen threshold, confusion matrix, fold train times, total wall time, ROC curve points (fpr/tpr downsampled to ~200 points), and top-30 LGBM gain importances.
6. Threshold: choose the OOF-stack threshold maximizing F1 on the positive class; store in bundle.
7. `models/bundle.joblib` = dict: `{"lgbm_folds": [...], "xgb_folds": [...], "lr_meta": ..., "feature_list": [...], "threshold": float, "metrics": {...}, "trained_at": iso_timestamp, "seed": SEED}`. ONE file, nothing else.
8. Enforce budget: print total training wall time; if > 25 min, print a warning telling the developer to reduce n_estimators.

**Fallback rule (only if the developer says time is critical):** ship 5-fold LightGBM alone; the stack is a bolt-on since OOF machinery is identical. Do not implement the fallback pre-emptively.

**✅ CHECKPOINT 3:** Bundle + metrics.json exist. Expected OOF AUC: LGBM ≥ 0.78, stack ≥ LGBM. Train time < 30 min. STOP.

---

## STAGE 4 — `pipeline/04_score.py` (PD + Health Score + EWS + SHAP, one pass)

Single script, single output pass, TRAIN rows only (307,511 — the demo scores the known portfolio):

1. **PD:** mean of 5 LGBM fold predictions and 5 XGB fold predictions → LR meta → `pd` column (float32).
2. **Health score:** rank-based map per config: `health = round(850 − 550 * rank(pd)/n)`; add `score_band` via SCORE_BANDS.
3. **EWS:** evaluate every rule in `EWS_RULES` against the feature columns + pd; store one int8 column per rule (fired 0/1), `ews_rules_fired` (count), `ews_tier` from EWS_SEVERITY ("Green" if 0). Rules referencing a missing/NaN feature simply don't fire (treat NaN comparisons as False).
4. **SHAP:** `shap.TreeExplainer` on each LGBM fold model, averaged across folds, float32, computed in chunks of ~50K rows to control RAM. Store:
   - `shap_global.parquet` — mean |SHAP| per feature, sorted desc (feature, importance).
   - `shap_top10.parquet` — long format: SK_ID_CURR, feature, shap_value, feature_value for each customer's top-10 |SHAP| features.
   - SHAP explains the LGBM component only — this is intentional and defensible; do not attempt stack-level SHAP.
5. Write `scored_customers.parquet`: SK_ID_CURR, TARGET, pd, health, score_band, all EWS columns, plus the ~12 feature columns the Lookup tab displays (income, credit, annuity, the ratio features, EXT_MEAN, age in years, employment years). Keep it lean — the dashboard should NOT load all 180 features.
6. Print: file sizes, tier distribution, band distribution, SHAP compute time.

**✅ CHECKPOINT 4:** Three parquets in `data/scored/`. `scored_customers.parquet` well under 200MB. Spot-check one known-default customer: high pd ↔ low health ↔ Red/Amber tier must be consistent. **Pipeline is DONE — nothing upstream changes after this point.** STOP.

---

## STAGE 5 — `app/dashboard.py` + components (build ONE TAB AT A TIME)

Global rules:
- `st.set_page_config(layout="wide", page_title="CreditPulse AI")`; dark theme via `.streamlit/config.toml` (create it: base="dark", primaryColor=accent from config) + Plotly `template="plotly_dark"` with DARK_THEME colors.
- ALL data loads behind `@st.cache_data`: scored parquet (indexed by SK_ID_CURR), shap parquets, metrics.json, bundle metadata ONLY (never load fold models into the app).
- NO live inference anywhere. NO raw CSV reads. If a number isn't in a scored parquet or metrics.json, it doesn't go on the dashboard.
- `dashboard.py` is a thin shell: page config, cached loaders, `st.tabs`, delegate to one component module per tab.

**5a. Portfolio Overview (`components/portfolio.py`):**
Top KPI row (st.metric ×4): total customers, portfolio default rate, mean health score, count of Red-tier customers. Then: health-score histogram colored by band; PD deciles vs actual default rate bar chart (shows calibration — judges love this); score-band × actual-default-rate table; risk-tier donut.
→ verify, then continue.

**5b. Customer Lookup (`components/lookup.py`):**
`st.selectbox`/text input for SK_ID_CURR → O(1) `.loc` on the cached indexed dataframe (this is the sub-second requirement — no filtering/scanning). Display: health score as a big gauge (Plotly indicator), PD, band, EWS tier + which rules fired, the ~12 profile features, and a horizontal signed bar chart of the customer's top-10 SHAP drivers (red = pushes toward default, green = away) pulled from shap_top10.parquet with plain-English feature labels (maintain a small FEATURE_LABELS dict in the component).
→ verify, then continue.

**5c. Early Warning (`components/ews.py`):**
Filterable table (tier multiselect, min rules-fired slider, band filter) of flagged customers sorted by pd desc, showing SK_ID_CURR, pd, health, tier, fired-rule names. Rule-frequency bar chart (which rules fire most across the portfolio). Tier counts as KPI row. Row click / ID copy hint pointing to Lookup tab.
→ verify, then continue.

**5d. Model Performance (`components/performance.py`):**
All from metrics.json: OOF AUC comparison (LGBM vs XGB vs Stack) bar; ROC curve from stored points; per-fold AUC table (shows CV honesty); confusion matrix at threshold; top-20 global SHAP importance bar (from shap_global.parquet) with plain-English labels; training-time stat line ("trained in X min on CPU" — this is a judging point, say it).

**✅ CHECKPOINT 5:** `streamlit run app/dashboard.py` cold-starts under ~10s, every tab renders with no live compute, lookup responds instantly. STOP.

---

## STAGE 6 — Hardening + README

1. `run_pipeline.bat`: `python pipeline/01... && python pipeline/02... && python pipeline/03... && python pipeline/04...` — chained with `&&` so any failure aborts (prevents train/score skew).
2. README.md: one-paragraph pitch (problem → 307K-customer relational dataset → stacked GBM with cost-sensitive weighting → precomputed SHAP → 4-tab decision dashboard); architecture diagram (the ASCII flow above); exact setup steps (venv, pip install, place CSVs in data/raw, run_pipeline.bat, streamlit run); headline metrics table (fill from metrics.json); "Design decisions" section with 5 bullets: why no SMOTE (cost-sensitive weighting preserves calibration), why native-NaN over imputation (GBMs route missingness as signal), why precomputed SHAP (demo reliability), why single artifact bundle (no version skew), why rank-based health score (robust to PD calibration drift).
3. Fresh-venv cold-start test instruction as a checklist in README.
4. Suggest the developer records a screen capture of the working demo as fallback.

**✅ CHECKPOINT 6 (FINAL):** Fresh clone + fresh venv → `run_pipeline.bat` → dashboard works. Done.

---

## NON-NEGOTIABLES SUMMARY (Cursor: re-read before every stage)
1. Stop at every checkpoint. Never build two stages in one pass.
2. Never rename a column after Checkpoint 2. The parquet schemas are contracts.
3. Dashboard never runs a model, never reads raw data, never computes SHAP.
4. All constants live in `config.py` — zero magic numbers in pipeline or app code.
5. Memory discipline in every pipeline script: dtype on read, downcast after aggregate, aggregate-then-merge, del + gc.
6. If train time exceeds budget, reduce n_estimators first — never remove CV folds or the OOF mechanism.
