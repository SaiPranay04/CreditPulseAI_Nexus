import pandas as pd
import numpy as np
import gc
import time
import joblib
import shap
import config

def score_pipeline():
    start_time = time.time()
    
    print(f"Loading model bundle from {config.BUNDLE}...")
    bundle = joblib.load(config.BUNDLE)
    
    lgb_models = bundle['lgb_folds']
    xgb_models = bundle['xgb_folds']
    cat_models = bundle.get('cat_folds', [])
    lr_meta = bundle['lr_meta']
    feature_list = bundle['feature_list']
    
    print(f"Loading master features from {config.MASTER_FEATURES}...")
    df = pd.read_parquet(config.MASTER_FEATURES)
    
    # Filter to TRAIN rows only (as per instructions, demo scores the known train portfolio)
    df = df[df['IS_TRAIN'] == 1].reset_index(drop=True)
    
    print(f"Scoring {len(df)} customers...")
    X = df[feature_list]
    
    # 1. Compute PD using stacked models
    print("Computing stacked probability of default (PD) predictions...")
    lgb_preds_all = np.column_stack([m.predict_proba(X)[:, 1] for m in lgb_models])
    xgb_preds_all = np.column_stack([m.predict_proba(X)[:, 1] for m in xgb_models])
    
    mean_lgb = lgb_preds_all.mean(axis=1)
    mean_xgb = xgb_preds_all.mean(axis=1)
    
    if len(cat_models) > 0:
        print("Including CatBoost predictions in stacking...")
        cat_preds_all = np.column_stack([m.predict_proba(X)[:, 1] for m in cat_models])
        mean_cat = cat_preds_all.mean(axis=1)
        X_meta = np.column_stack([mean_lgb, mean_xgb, mean_cat])
    else:
        X_meta = np.column_stack([mean_lgb, mean_xgb])
        
    df['pd'] = lr_meta.predict_proba(X_meta)[:, 1].astype(np.float32)
    
    # 2. Health score: rank-based map per config: health = round(850 - 550 * rank(pd)/n)
    print("Computing health scores and bands...")
    # Lower PD -> Higher Rank (Good health); Higher PD -> Lower Rank (Bad health)
    # So we rank PD in ascending order: rank = 1 (lowest PD) to n (highest PD)
    # Then pct_rank = rank/n goes from ~0 (lowest risk) to 1 (highest risk)
    # health = 850 - 550 * pct_rank will go from 850 down to 300.
    pd_rank_pct = df['pd'].rank(pct=True, method='min')
    df['health'] = (850 - 550 * pd_rank_pct).round().astype(np.int32)
    
    # Assign score bands
    # Inclusive lower bounds: "Excellent": 750, "Good": 650, "Fair": 550, "Watch": 450, "Critical": 300
    conditions = [
        df['health'] >= config.SCORE_BANDS["Excellent"],
        df['health'] >= config.SCORE_BANDS["Good"],
        df['health'] >= config.SCORE_BANDS["Fair"],
        df['health'] >= config.SCORE_BANDS["Watch"],
        df['health'] >= config.SCORE_BANDS["Critical"]
    ]
    choices = ["Excellent", "Good", "Fair", "Watch", "Critical"]
    df['score_band'] = np.select(conditions, choices, default="Critical")
    
    # 3. EWS rules evaluation
    print("Evaluating Early Warning System (EWS) rules...")
    for rule_name, (col, op, val) in config.EWS_RULES.items():
        # Handle rule validation against pd or feature columns
        series = df[col]
        if op == ">=":
            fired = (series >= val)
        elif op == "==":
            fired = (series == val)
        elif op == "<=":
            fired = (series <= val)
        else:
            fired = (series == val)
        
        # Guard missing/NaN values: they don't fire (evaluate to False)
        df[rule_name] = fired.fillna(False).astype(np.int8)
        
    rule_cols = list(config.EWS_RULES.keys())
    df['ews_rules_fired'] = df[rule_cols].sum(axis=1).astype(np.int8)
    
    # Severity tier based on EWS_SEVERITY: Red (>=3), Amber (>=2), Yellow (>=1), else Green
    ews_conditions = [
        df['ews_rules_fired'] >= config.EWS_SEVERITY["Red"],
        df['ews_rules_fired'] >= config.EWS_SEVERITY["Amber"],
        df['ews_rules_fired'] >= config.EWS_SEVERITY["Yellow"]
    ]
    ews_choices = ["Red", "Amber", "Yellow"]
    df['ews_tier'] = np.select(ews_conditions, ews_choices, default="Green")
    
    # 4. SHAP local explanations (LGBM component only, in 50K chunks)
    if config.SHAP_GLOBAL.exists() and config.SHAP_TOP10.exists():
        print("Precomputed SHAP files already exist. Skipping recalculation...")
    else:
        print("Computing SHAP values (LGBM components only)...")
        shap_start = time.time()
        
        # Initialize explainers for each LGBM fold model
        explainers = [shap.TreeExplainer(model) for model in lgb_models]
        
        # We will compute SHAP values in chunks to control memory usage
        chunk_size = 50000
        n_samples = len(X)
        shap_values_sum = np.zeros((n_samples, len(feature_list)), dtype=np.float32)
        
        for i in range(0, n_samples, chunk_size):
            print(f"  Processing SHAP chunk {i // chunk_size + 1}...")
            X_chunk = X.iloc[i : i + chunk_size]
            
            for exp in explainers:
                sv = exp.shap_values(X_chunk)
                
                # Make robust to SHAP output format variations (list of classes vs. 3D array vs. 2D array)
                if isinstance(sv, list):
                    # Binary classification: index 1 is class 1 (default)
                    sv = sv[1]
                elif len(sv.shape) == 3:
                    # Shape (samples, features, classes) -> take class 1
                    sv = sv[:, :, 1]
                    
                shap_values_sum[i : i + chunk_size] += sv.astype(np.float32)
                
        shap_values_avg = shap_values_sum / len(explainers)
        shap_duration = time.time() - shap_start
        print(f"SHAP calculations finished in {shap_duration:.2f}s")
        
        # 4a. Save shap_global.parquet (Mean |SHAP| per feature, sorted descending)
        print("Saving global SHAP importances...")
        global_shap = pd.DataFrame({
            'feature': feature_list,
            'importance': np.abs(shap_values_avg).mean(axis=0).astype(np.float32)
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        config.SCORED_DIR.mkdir(parents=True, exist_ok=True)
        global_shap.to_parquet(config.SHAP_GLOBAL, index=False)
        print(f"Saved {config.SHAP_GLOBAL} ({len(global_shap)} features)")
        
        # 4b. Save shap_top10.parquet (Long format for each customer's top 10 features)
        print("Constructing top-10 SHAP drivers per customer...")
        abs_shap = np.abs(shap_values_avg)
        
        # Efficient numpy vectorization to find top 10 features per row
        top10_idx = np.argpartition(abs_shap, -10, axis=1)[:, -10:]
        row_indices = np.arange(n_samples)[:, None]
        top10_abs_vals = abs_shap[row_indices, top10_idx]
        sort_order = np.argsort(top10_abs_vals, axis=1)[:, ::-1]
        top10_idx_sorted = top10_idx[row_indices, sort_order]
        
        curr_ids = df['SK_ID_CURR'].values
        repeated_ids = np.repeat(curr_ids, 10)
        flat_indices = top10_idx_sorted.flatten()
        
        feature_names_arr = np.array(feature_list)
        flat_features = feature_names_arr[flat_indices]
        
        flat_shap_vals = shap_values_avg[row_indices, top10_idx_sorted].flatten()
        flat_feat_vals = X.values[row_indices, top10_idx_sorted].flatten()
        
        shap_top10 = pd.DataFrame({
            'SK_ID_CURR': repeated_ids,
            'feature': flat_features,
            'shap_value': flat_shap_vals.astype(np.float32),
            'feature_value': flat_feat_vals.astype(np.float32)
        })
        
        shap_top10.to_parquet(config.SHAP_TOP10, index=False)
        print(f"Saved {config.SHAP_TOP10} ({len(shap_top10)} records)")
    
    # 5. Save scored_customers.parquet
    # We keep only essential profile features for the dashboard lookup to be super fast.
    profile_features = [
        'SK_ID_CURR', 'TARGET', 'pd', 'health', 'score_band', 'ews_rules_fired', 'ews_tier',
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
        'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'PAYMENT_CREDIT_RATIO',
        'EMPLOYED_AGE_RATIO', 'INCOME_PER_PERSON', 'EXT_MEAN',
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'INST_LATE_RATIO', 'PREV_REFUSED_RATIO', 'CC_UTILIZATION_MAX'
    ]
    # Add the individual EWS columns
    all_output_cols = profile_features + rule_cols
    
    # Select and save
    scored_df = df[all_output_cols].copy()
    scored_df.to_parquet(config.SCORED, index=False)
    print(f"Saved {config.SCORED} ({len(scored_df)} rows, {len(scored_df.columns)} columns)")
    
    # Print statistics
    print("\n--- SCORING RESULTS SUMMARY ---")
    print(f"Tier Distribution:\n{scored_df['ews_tier'].value_counts()}")
    print(f"Band Distribution:\n{scored_df['score_band'].value_counts()}")
    
    total_time = time.time() - start_time
    print(f"\nScoring pipeline completed in {total_time:.2f}s ({total_time/60:.2f} mins)")

if __name__ == "__main__":
    score_pipeline()
