"""
CreditPulse AI — FastAPI Backend
Reads precomputed Parquet outputs and serves JSON to the Next.js frontend.
"""
from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ── resolve project root (this file lives at app/api/main.py)
ROOT = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(ROOT))
import config

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(title="CreditPulse AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# DATA LOADING  (once at startup, cached)
# ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def _scored() -> pd.DataFrame:
    df = pd.read_parquet(config.SCORED)
    return df.set_index("SK_ID_CURR", drop=False)

@lru_cache(maxsize=1)
def _shap_top10() -> pd.DataFrame:
    return pd.read_parquet(config.SHAP_TOP10)

@lru_cache(maxsize=1)
def _shap_global() -> pd.DataFrame:
    return pd.read_parquet(config.SHAP_GLOBAL)

@lru_cache(maxsize=1)
def _metrics() -> dict:
    with open(config.METRICS) as f:
        return json.load(f)

def _nan_safe(obj):
    """Recursively replace NaN/Inf with None so JSON serialises cleanly."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_safe(v) for v in obj]
    return obj


# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "customers": len(_scored())}


# ─────────────────────────────────────────────
# PORTFOLIO
# ─────────────────────────────────────────────
@app.get("/api/portfolio/summary")
def portfolio_summary():
    df = _scored()
    total = len(df)
    def_rate = float((df["TARGET"] == 1).mean() * 100)
    mean_health = float(df["health"].mean())
    red_n = int((df["ews_tier"] == "Red").sum())

    # Tier counts
    tier_counts = df["ews_tier"].value_counts().to_dict()
    # Band counts
    band_order = ["Excellent", "Good", "Fair", "Watch", "Critical"]
    band_counts = {b: int(df["score_band"].eq(b).sum()) for b in band_order}

    return _nan_safe({
        "total": total,
        "default_rate": round(def_rate, 2),
        "mean_health": round(mean_health, 1),
        "red_tier_count": red_n,
        "red_tier_pct": round(red_n / total * 100, 1),
        "tier_counts": {k: int(v) for k, v in tier_counts.items()},
        "band_counts": band_counts,
    })


@app.get("/api/portfolio/health-distribution")
def health_distribution():
    df = _scored()
    bins = list(range(300, 860, 10))
    records = []
    band_order = ["Excellent", "Good", "Fair", "Watch", "Critical"]
    for band in band_order:
        sub = df[df["score_band"] == band]["health"]
        counts, edges = np.histogram(sub, bins=bins)
        for i, c in enumerate(counts):
            if c > 0:
                records.append({
                    "score": int(edges[i]),
                    "count": int(c),
                    "band": band,
                })
    return records


@app.get("/api/portfolio/calibration")
def calibration():
    df = _scored().copy()
    df["decile"] = pd.qcut(df["pd"], 10, labels=False, duplicates="drop") + 1
    cal = df.groupby("decile").agg(
        pred=("pd", "mean"),
        actual=("TARGET", "mean")
    ).reset_index()
    return _nan_safe([
        {
            "decile": int(row["decile"]),
            "predicted_pd": round(float(row["pred"]) * 100, 2),
            "actual_dr": round(float(row["actual"]) * 100, 2),
        }
        for _, row in cal.iterrows()
    ])


@app.get("/api/portfolio/band-summary")
def band_summary():
    df = _scored()
    band_order = ["Excellent", "Good", "Fair", "Watch", "Critical"]
    rows = []
    for band in band_order:
        sub = df[df["score_band"] == band]
        rows.append(_nan_safe({
            "band": band,
            "count": int(len(sub)),
            "avg_health": round(float(sub["health"].mean()), 1) if len(sub) else 0,
            "avg_pd": round(float(sub["pd"].mean()) * 100, 2) if len(sub) else 0,
            "actual_dr": round(float(sub["TARGET"].mean()) * 100, 2) if len(sub) else 0,
        }))
    return rows


# ─────────────────────────────────────────────
# CUSTOMER LOOKUP
# ─────────────────────────────────────────────
@app.get("/api/customer/{customer_id}")
def customer_profile(customer_id: int):
    df = _scored()
    if customer_id not in df.index:
        raise HTTPException(status_code=404, detail="Customer not found")
    row = df.loc[customer_id]

    ews_fired = {rule: int(row[rule]) for rule in config.EWS_RULES}
    rules_fired = [r for r, v in ews_fired.items() if v == 1]

    return _nan_safe({
        "sk_id_curr": int(customer_id),
        "pd": round(float(row["pd"]) * 100, 2),
        "health": int(row["health"]),
        "score_band": str(row["score_band"]),
        "ews_tier": str(row["ews_tier"]),
        "ews_rules_fired": int(row["ews_rules_fired"]),
        "rules_fired": rules_fired,
        "ews_details": {
            rule: {"col": col, "op": op, "val": val}
            for rule, (col, op, val) in config.EWS_RULES.items()
        },
        "profile": {
            "amt_income": float(row["AMT_INCOME_TOTAL"]),
            "amt_credit": float(row["AMT_CREDIT"]),
            "amt_annuity": float(row["AMT_ANNUITY"]),
            "credit_income_ratio": float(row["CREDIT_INCOME_RATIO"]),
            "annuity_income_ratio": float(row["ANNUITY_INCOME_RATIO"]),
            "payment_credit_ratio": float(row["PAYMENT_CREDIT_RATIO"]),
            "days_birth": float(row["DAYS_BIRTH"]),
            "days_employed": float(row["DAYS_EMPLOYED"]) if not pd.isna(row["DAYS_EMPLOYED"]) else None,
            "income_per_person": float(row["INCOME_PER_PERSON"]),
            "ext_mean": float(row["EXT_MEAN"]) if not pd.isna(row["EXT_MEAN"]) else None,
            "inst_late_ratio": float(row["INST_LATE_RATIO"]) if not pd.isna(row["INST_LATE_RATIO"]) else None,
            "prev_refused_ratio": float(row["PREV_REFUSED_RATIO"]) if not pd.isna(row["PREV_REFUSED_RATIO"]) else None,
            "cc_utilization_max": float(row["CC_UTILIZATION_MAX"]) if not pd.isna(row["CC_UTILIZATION_MAX"]) else None,
        }
    })


@app.get("/api/customer/{customer_id}/shap")
def customer_shap(customer_id: int):
    shap_df = _shap_top10()
    sub = shap_df[shap_df["SK_ID_CURR"] == customer_id].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("shap_value", key=abs, ascending=False)
    return _nan_safe([
        {
            "feature": row["feature"],
            "shap_value": round(float(row["shap_value"]), 5),
            "feature_value": round(float(row["feature_value"]), 4) if not pd.isna(row["feature_value"]) else None,
        }
        for _, row in sub.iterrows()
    ])


# ─────────────────────────────────────────────
# EARLY WARNING
# ─────────────────────────────────────────────
@app.get("/api/ews/summary")
def ews_summary():
    df = _scored()
    tier_counts = df["ews_tier"].value_counts().to_dict()
    total = len(df)
    return _nan_safe({
        "total": total,
        "tier_counts": {k: int(v) for k, v in tier_counts.items()},
        "tier_pcts": {k: round(int(v) / total * 100, 1) for k, v in tier_counts.items()},
    })


@app.get("/api/ews/rule-frequency")
def rule_frequency():
    df = _scored()
    rule_cols = list(config.EWS_RULES.keys())
    return [
        {"rule": rule, "count": int(df[rule].sum())}
        for rule in rule_cols
    ]


@app.get("/api/ews/registry")
def ews_registry(
    tiers: Optional[str] = Query(None, description="Comma-separated tiers"),
    bands: Optional[str] = Query(None, description="Comma-separated bands"),
    min_rules: int = Query(0),
    limit: int = Query(100),
):
    df = _scored()
    sel_tiers = tiers.split(",") if tiers else ["Red", "Amber", "Yellow", "Green"]
    sel_bands = bands.split(",") if bands else ["Critical", "Watch", "Fair", "Good", "Excellent"]

    filtered = df[
        df["ews_tier"].isin(sel_tiers) &
        df["score_band"].isin(sel_bands) &
        (df["ews_rules_fired"] >= min_rules)
    ].sort_values("pd", ascending=False).head(limit)

    rule_cols = list(config.EWS_RULES.keys())
    return _nan_safe([
        {
            "id": int(row["SK_ID_CURR"]),
            "pd": round(float(row["pd"]) * 100, 2),
            "health": int(row["health"]),
            "score_band": str(row["score_band"]),
            "ews_tier": str(row["ews_tier"]),
            "rules_fired": int(row["ews_rules_fired"]),
            "breached_rules": [r for r in rule_cols if row[r] == 1],
        }
        for _, row in filtered.iterrows()
    ])


@app.get("/api/ews/tier-band-cross")
def tier_band_cross():
    df = _scored()
    cross = df.groupby(["score_band", "ews_tier"]).size().reset_index(name="n")
    return _nan_safe([
        {"band": row["score_band"], "tier": row["ews_tier"], "count": int(row["n"])}
        for _, row in cross.iterrows()
    ])


# ─────────────────────────────────────────────
# MODEL PERFORMANCE
# ─────────────────────────────────────────────
@app.get("/api/performance/metrics")
def performance_metrics():
    return _nan_safe(_metrics())


@app.get("/api/performance/shap-global")
def shap_global_top20():
    df = _shap_global().head(20).copy()
    return _nan_safe([
        {"feature": row["feature"], "importance": round(float(row["importance"]), 6)}
        for _, row in df.iterrows()
    ])


# ─────────────────────────────────────────────
# EWS RULES CONFIG  (for frontend to display)
# ─────────────────────────────────────────────
@app.get("/api/config/ews-rules")
def ews_rules():
    return {
        rule: {"col": col, "op": op, "val": val}
        for rule, (col, op, val) in config.EWS_RULES.items()
    }
