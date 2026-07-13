from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).parent
RAW_DIR = ROOT / "home-credit-default-risk"
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
    objective="binary", metric="auc", n_estimators=2000, learning_rate=0.03,
    num_leaves=34, colsample_bytree=0.85, subsample=0.85,
    reg_alpha=0.04, reg_lambda=0.07, min_child_samples=40,
    scale_pos_weight=11.4, n_jobs=-1, random_state=SEED, verbosity=-1,
)
XGB_PARAMS = dict(
    objective="binary:logistic", n_estimators=1200, learning_rate=0.05,
    max_depth=6, colsample_bytree=0.8, subsample=0.8,
    reg_alpha=0.05, reg_lambda=1.0, min_child_weight=30,
    scale_pos_weight=11.4, n_jobs=-1, random_state=SEED,
    tree_method="hist", device="cuda", eval_metric="auc",
)
CATBOOST_PARAMS = dict(
    iterations=1000, learning_rate=0.05, depth=6,
    eval_metric="AUC", random_seed=SEED, task_type="GPU",
    verbose=100,
)
EARLY_STOPPING_ROUNDS = 100

# ---- Health score: monotone map of PD → 300–850 band ----
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
