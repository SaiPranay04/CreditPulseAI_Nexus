import pandas as pd
import numpy as np
import gc
import time
import json
import datetime
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, f1_score, roc_curve
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import config

def train_pipeline():
    start_time = time.time()
    
    print(f"Loading master features from {config.MASTER_FEATURES}...")
    df = pd.read_parquet(config.MASTER_FEATURES)
    
    # Train set only
    train_df = df[df['IS_TRAIN'] == 1].reset_index(drop=True)
    del df
    gc.collect()
    
    # Identify target and features
    target = 'TARGET'
    features = [c for c in train_df.columns if c not in ['SK_ID_CURR', 'TARGET', 'IS_TRAIN']]
    print(f"Training on {len(features)} features. Total samples: {len(train_df)}")
    
    X = train_df[features]
    y = train_df[target]
    
    # K-Fold CV setup
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    
    # OOF array initialization
    lgb_oof = np.zeros(len(train_df))
    xgb_oof = np.zeros(len(train_df))
    cat_oof = np.zeros(len(train_df))
    
    lgb_models = []
    xgb_models = []
    cat_models = []
    
    fold_metrics = []
    
    # Loop over folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_start = time.time()
        print(f"\n--- FOLD {fold+1} / {config.N_FOLDS} ---")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 1. Train LightGBM
        print("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )
        lgb_preds = lgb_model.predict_proba(X_val)[:, 1]
        lgb_oof[val_idx] = lgb_preds
        lgb_models.append(lgb_model)
        
        # 2. Train XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(**config.XGB_PARAMS, early_stopping_rounds=config.EARLY_STOPPING_ROUNDS)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
        xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
        xgb_oof[val_idx] = xgb_preds
        xgb_models.append(xgb_model)
        
        # 3. Train CatBoost
        print("Training CatBoost...")
        cat_model = CatBoostClassifier(**config.CATBOOST_PARAMS, early_stopping_rounds=config.EARLY_STOPPING_ROUNDS)
        cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        cat_preds = cat_model.predict_proba(X_val)[:, 1]
        cat_oof[val_idx] = cat_preds
        cat_models.append(cat_model)
        
        # Calculate fold AUCs
        lgb_auc = roc_auc_score(y_val, lgb_preds)
        xgb_auc = roc_auc_score(y_val, xgb_preds)
        cat_auc = roc_auc_score(y_val, cat_preds)
        
        fold_time = time.time() - fold_start
        print(f"Fold {fold+1} finished in {fold_time:.1f}s. LGB AUC: {lgb_auc:.4f}, XGB AUC: {xgb_auc:.4f}, CatBoost AUC: {cat_auc:.4f}")
        fold_metrics.append({
            'fold': fold + 1,
            'lgb_auc': float(lgb_auc),
            'xgb_auc': float(xgb_auc),
            'cat_auc': float(cat_auc),
            'duration': float(fold_time)
        })
        
        # Memory cleanup
        del X_train, y_train, X_val, y_val
        gc.collect()
        
    print("\n--- BASE MODEL EVALUATION ---")
    lgb_total_auc = roc_auc_score(y, lgb_oof)
    xgb_total_auc = roc_auc_score(y, xgb_oof)
    cat_total_auc = roc_auc_score(y, cat_oof)
    print(f"LightGBM OOF AUC: {lgb_total_auc:.4f}")
    print(f"XGBoost OOF AUC: {xgb_total_auc:.4f}")
    print(f"CatBoost OOF AUC: {cat_total_auc:.4f}")
    
    # 3. Stacking Ensemble Meta-Learner
    print("\nTraining Stacking Meta-Learner...")
    X_meta = np.column_stack([lgb_oof, xgb_oof, cat_oof])
    
    lr_meta = LogisticRegression(class_weight="balanced", random_state=config.SEED, max_iter=1000)
    lr_meta.fit(X_meta, y)
    
    stacked_oof = lr_meta.predict_proba(X_meta)[:, 1]
    stacked_auc = roc_auc_score(y, stacked_oof)
    gini = 2 * stacked_auc - 1
    print(f"Stacked OOF AUC: {stacked_auc:.4f}")
    print(f"Stacked Gini: {gini:.4f}")
    
    # 4. Find optimal threshold maximizing F1 on the positive class
    precisions, recalls, thresholds = precision_recall_curve(y, stacked_oof)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    print(f"Optimal threshold: {optimal_threshold:.4f} (F1 Score: {best_f1:.4f})")
    
    # 5. Evaluate final predictions at optimal threshold
    stacked_preds = (stacked_oof >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, stacked_preds).ravel()
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))
    
    # Precision/recall curves downsampled
    # Downsample points for plotly plotting in performance tab
    fpr, tpr, _ = roc_curve(y, stacked_oof)
    step = max(1, len(fpr) // 200)
    roc_points = [{'fpr': float(f), 'tpr': float(t)} for f, t in zip(fpr[::step], tpr[::step])]
    # Ensure final point is included
    if len(roc_points) > 0 and roc_points[-1] != {'fpr': 1.0, 'tpr': 1.0}:
        roc_points.append({'fpr': 1.0, 'tpr': 1.0})
        
    # Global feature importance: average LGBM gain importance
    print("Calculating feature importances...")
    importances = np.zeros(len(features))
    for model in lgb_models:
        importances += model.booster_.feature_importance(importance_type='gain')
    importances /= len(lgb_models)
    
    feat_imp = sorted(
        [{'feature': f, 'importance': float(i)} for f, i in zip(features, importances)],
        key=lambda x: x['importance'],
        reverse=True
    )[:30] # Top 30
    
    # Package metrics
    metrics = {
        'folds': fold_metrics,
        'lgb_oof_auc': float(lgb_total_auc),
        'xgb_oof_auc': float(xgb_total_auc),
        'cat_oof_auc': float(cat_total_auc),
        'stacked_oof_auc': float(stacked_auc),
        'gini': float(gini),
        'optimal_threshold': optimal_threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'roc_points': roc_points,
        'feature_importance_top30': feat_imp,
        'total_time_seconds': float(time.time() - start_time)
    }
    
    # Write metrics to json
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {config.METRICS}")
    
    # Package model bundle
    bundle = {
        "lgb_folds": lgb_models,
        "xgb_folds": xgb_models,
        "cat_folds": cat_models,
        "lr_meta": lr_meta,
        "feature_list": features,
        "threshold": optimal_threshold,
        "metrics": metrics,
        "trained_at": datetime.datetime.now().isoformat(),
        "seed": config.SEED
    }
    
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.BUNDLE, compress=3)
    print(f"Saved model bundle to {config.BUNDLE}")
    
    total_time = time.time() - start_time
    print(f"\nTraining pipeline completed in {total_time:.2f}s ({total_time/60:.2f} mins)")
    if total_time > 1500:
        print("WARNING: Total training time exceeded 25 minutes budget!")

if __name__ == "__main__":
    train_pipeline()
