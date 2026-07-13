import pandas as pd
import numpy as np
import gc
import time
from pathlib import Path
import config

def build_features():
    start_time = time.time()
    
    # ----------------------------------------------------
    # 2a. Base: Load application_train and application_test
    # ----------------------------------------------------
    print("Loading application tables...")
    train_path = config.RAW_DIR / "application_train.csv"
    test_path = config.RAW_DIR / "application_test.csv"
    
    app_train = pd.read_csv(train_path)
    app_test = pd.read_csv(test_path)
    
    app_train['IS_TRAIN'] = 1
    app_test['IS_TRAIN'] = 0
    app_test['TARGET'] = np.nan
    
    # Concat
    df = pd.concat([app_train, app_test], ignore_index=True)
    print(f"Base application shape: {df.shape}")
    
    # Free memory
    del app_train, app_test
    gc.collect()
    
    # ----------------------------------------------------
    # 2b. Application-level cleanup + ratio features
    # ----------------------------------------------------
    print("Computing application features...")
    # Replace DAYS_EMPLOYED == 365243 with NaN; add flag
    df['EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == config.DAYS_EMPLOYED_ANOMALY).astype('int8')
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(config.DAYS_EMPLOYED_ANOMALY, np.nan)
    
    # Ratios
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['PAYMENT_CREDIT_RATIO'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'].replace(0, np.nan)
    df['EMPLOYED_AGE_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'].replace(0, np.nan)
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, np.nan)
    
    # EXT_SOURCE combos
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_STD'] = df[ext_cols].std(axis=1).fillna(0)
    
    # Product of non-null
    df['EXT_PROD'] = df[ext_cols].prod(axis=1, skipna=True)
    # If all are null, prod is 1.0 under skipna=True, so we force NaN
    all_null = df[ext_cols].isnull().all(axis=1)
    df.loc[all_null, 'EXT_PROD'] = np.nan
    
    # Weighted EXT
    weighted_sum = 2 * df['EXT_SOURCE_2'].fillna(0) + 3 * df['EXT_SOURCE_3'].fillna(0) + df['EXT_SOURCE_1'].fillna(0)
    weight_sum = 2 * df['EXT_SOURCE_2'].notnull() + 3 * df['EXT_SOURCE_3'].notnull() + df['EXT_SOURCE_1'].notnull()
    df['EXT_WEIGHTED'] = weighted_sum / weight_sum.replace(0, np.nan)
    
    gc.collect()
    
    # ----------------------------------------------------
    # 2c. Bureau and Bureau Balance aggregation
    # ----------------------------------------------------
    print("Processing bureau and bureau_balance...")
    b_path = config.RAW_DIR / "bureau.csv"
    bb_path = config.RAW_DIR / "bureau_balance.csv"
    
    # Process bureau_balance first
    if bb_path.exists():
        print("Aggregating bureau balance...")
        bb = pd.read_csv(bb_path, usecols=['SK_ID_BUREAU', 'STATUS'], dtype={'STATUS': 'category'})
        # Delinquency flags: status in {"1","2","3","4","5"}
        bb['is_late'] = bb['STATUS'].isin(['1', '2', '3', '4', '5']).astype('float32')
        bb_agg = bb.groupby('SK_ID_BUREAU').agg(
            BB_MONTHS_COUNT=('STATUS', 'count'),
            BB_LATE_COUNT=('is_late', 'sum')
        ).reset_index()
        del bb
        gc.collect()
    else:
        bb_agg = None
        
    # Process bureau
    if b_path.exists():
        bureau = pd.read_csv(b_path)
        # Merge bb_agg if exists
        if bb_agg is not None:
            bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
            del bb_agg
            gc.collect()
            bureau['BB_LATE_COUNT'] = bureau['BB_LATE_COUNT'].fillna(0)
            bureau['BB_MONTHS_COUNT'] = bureau['BB_MONTHS_COUNT'].fillna(0)
        else:
            bureau['BB_LATE_COUNT'] = 0.0
            bureau['BB_MONTHS_COUNT'] = 0.0
            
        # Group by SK_ID_CURR
        bureau_agg = bureau.groupby('SK_ID_CURR').agg(
            BUREAU_LOAN_COUNT=('SK_ID_BUREAU', 'count'),
            BUREAU_ACTIVE_COUNT=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
            BUREAU_DAYS_CREDIT_MEAN=('DAYS_CREDIT', 'mean'),
            BUREAU_DAYS_CREDIT_MAX=('DAYS_CREDIT', 'max'),
            BUREAU_CREDIT_SUM=('AMT_CREDIT_SUM', 'sum'),
            BUREAU_CREDIT_SUM_DEBT=('AMT_CREDIT_SUM_DEBT', 'sum'),
            BUREAU_CREDIT_DAY_OVERDUE_MEAN=('CREDIT_DAY_OVERDUE', 'mean'),
            BUREAU_CREDIT_TYPE_COUNT=('CREDIT_TYPE', 'nunique'),
            BUREAU_BB_LATE_SUM=('BB_LATE_COUNT', 'sum'),
            BUREAU_BB_MONTHS_SUM=('BB_MONTHS_COUNT', 'sum')
        ).reset_index()
        
        # Ratios & custom columns
        bureau_agg['BUREAU_DEBT_CREDIT_RATIO'] = bureau_agg['BUREAU_CREDIT_SUM_DEBT'] / bureau_agg['BUREAU_CREDIT_SUM'].replace(0, np.nan)
        bureau_agg['BUREAU_DPD_RATIO'] = bureau_agg['BUREAU_BB_LATE_SUM'] / bureau_agg['BUREAU_BB_MONTHS_SUM'].replace(0, np.nan)
        bureau_agg.drop(columns=['BUREAU_BB_LATE_SUM', 'BUREAU_BB_MONTHS_SUM'], inplace=True)
        
        # Rename columns to match prefix
        bureau_agg.columns = ['SK_ID_CURR'] + ['BUREAU_' + col if not col.startswith('BUREAU_') else col for col in bureau_agg.columns[1:]]
        
        # Merge
        df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
        del bureau, bureau_agg
        gc.collect()
        
    # ----------------------------------------------------
    # 2d. Previous application aggregation
    # ----------------------------------------------------
    print("Processing previous applications...")
    prev_path = config.RAW_DIR / "previous_application.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path)
        # Ratios
        prev['credit_app_ratio'] = prev['AMT_CREDIT'] / prev['AMT_APPLICATION'].replace(0, np.nan)
        
        prev_agg = prev.groupby('SK_ID_CURR').agg(
            PREV_COUNT=('SK_ID_PREV', 'count'),
            PREV_REFUSED_RATIO=('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').mean()),
            PREV_APPROVED_RATIO=('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').mean()),
            PREV_AMT_APPLICATION_MEAN=('AMT_APPLICATION', 'mean'),
            PREV_CREDIT_APP_RATIO_MEAN=('credit_app_ratio', 'mean'),
            PREV_CNT_PAYMENT_MEAN=('CNT_PAYMENT', 'mean'),
            PREV_DAYS_DECISION_MEAN=('DAYS_DECISION', 'mean')
        ).reset_index()
        
        df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
        del prev, prev_agg
        gc.collect()

    # ----------------------------------------------------
    # 2e. Installments aggregation
    # ----------------------------------------------------
    print("Processing installments payments...")
    inst_path = config.RAW_DIR / "installments_payments.csv"
    if inst_path.exists():
        inst = pd.read_csv(inst_path)
        inst['is_late'] = (inst['DAYS_ENTRY_PAYMENT'] > inst['DAYS_INSTALMENT']).astype('float32')
        inst['late_days'] = (inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']).clip(lower=0)
        inst['is_underpaid'] = (inst['AMT_PAYMENT'] < inst['AMT_INSTALMENT']).astype('float32')
        
        inst_agg = inst.groupby('SK_ID_CURR').agg(
            INST_LATE_RATIO=('is_late', 'mean'),
            INST_LATE_DAYS_MEAN=('late_days', 'mean'),
            INST_UNDERPAY_RATIO=('is_underpaid', 'mean'),
            INST_PAYMENT_SUM=('AMT_PAYMENT', 'sum'),
            INST_INSTALMENT_SUM=('AMT_INSTALMENT', 'sum'),
            INST_COUNT=('NUM_INSTALMENT_NUMBER', 'count')
        ).reset_index()
        
        inst_agg['INST_PAYMENT_INSTALMENT_RATIO'] = inst_agg['INST_PAYMENT_SUM'] / inst_agg['INST_INSTALMENT_SUM'].replace(0, np.nan)
        inst_agg.drop(columns=['INST_PAYMENT_SUM', 'INST_INSTALMENT_SUM'], inplace=True)
        
        df = df.merge(inst_agg, on='SK_ID_CURR', how='left')
        del inst, inst_agg
        gc.collect()

    # ----------------------------------------------------
    # 2f. POS cash aggregation
    # ----------------------------------------------------
    print("Processing POS CASH balance...")
    pos_path = config.RAW_DIR / "POS_CASH_balance.csv"
    if pos_path.exists():
        pos = pd.read_csv(pos_path)
        pos['is_late'] = (pos['SK_DPD'] > 0).astype('float32')
        
        pos_agg = pos.groupby('SK_ID_CURR').agg(
            POS_DPD_MEAN=('SK_DPD', 'mean'),
            POS_DPD_POS_RATIO=('is_late', 'mean'),
            POS_MONTHS_COUNT=('MONTHS_BALANCE', 'count'),
            POS_COMPLETED_COUNT=('NAME_CONTRACT_STATUS', lambda x: (x == 'Completed').sum())
        ).reset_index()
        
        df = df.merge(pos_agg, on='SK_ID_CURR', how='left')
        del pos, pos_agg
        gc.collect()

    # ----------------------------------------------------
    # 2g. Merge Credit Card Aggregates
    # ----------------------------------------------------
    if config.CC_AGG.exists():
        print("Merging credit card aggregates...")
        cc_agg = pd.read_parquet(config.CC_AGG)
        df = df.merge(cc_agg, on='SK_ID_CURR', how='left')
        del cc_agg
        gc.collect()

    # ----------------------------------------------------
    # 2h. Missing-data rule (apply AFTER all merges, train-portion only)
    # ----------------------------------------------------
    print("Applying missingness thresholds...")
    train_mask = df['IS_TRAIN'] == 1
    cols_to_evaluate = [c for c in df.columns if c not in ['SK_ID_CURR', 'TARGET', 'IS_TRAIN']]
    
    dropped_cols = []
    flags_added = 0
    
    for col in cols_to_evaluate:
        missing_rate = df.loc[train_mask, col].isnull().mean()
        if missing_rate > config.DROP_MISSING_THRESHOLD and not col.startswith('CC_'):
            dropped_cols.append(col)
            df.drop(columns=[col], inplace=True)
        elif missing_rate >= config.FLAG_MISSING_THRESHOLD:
            df[f"{col}_isna"] = df[col].isnull().astype('int8')
            flags_added += 1
            
    print(f"Dropped {len(dropped_cols)} columns due to missingness > {config.DROP_MISSING_THRESHOLD:.0%}")
    print(f"Added {flags_added} missingness '_isna' flags (for columns with 20%-60% missingness)")
    gc.collect()
    
    # ----------------------------------------------------
    # 2i. Categoricals: Label encoding
    # ----------------------------------------------------
    print("Label encoding categorical columns...")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        # Fit on combined train+test (sorted categories for deterministic mapping)
        categories = sorted(df[col].dropna().unique())
        cat_mapping = {cat: i for i, cat in enumerate(categories)}
        # Encode
        df[col] = df[col].map(cat_mapping).astype('float32') # Use float to allow NaNs
        
    print(f"Encoded {len(cat_cols)} categorical columns.")
    gc.collect()
    
    # ----------------------------------------------------
    # 2j. Downcast all numeric columns to float32 or smallest safe int
    # ----------------------------------------------------
    print("Downcasting column dtypes...")
    for col in df.columns:
        if col in ['SK_ID_CURR', 'IS_TRAIN', 'TARGET']:
            continue
            
        col_type = df[col].dtype
        if np.issubdtype(col_type, np.integer):
            min_val, max_val = df[col].min(), df[col].max()
            if min_val >= 0:
                if max_val < 256:
                    df[col] = df[col].astype('uint8')
                elif max_val < 65536:
                    df[col] = df[col].astype('uint16')
                else:
                    df[col] = df[col].astype('uint32')
            else:
                if min_val >= -128 and max_val < 128:
                    df[col] = df[col].astype('int8')
                elif min_val >= -32768 and max_val < 32768:
                    df[col] = df[col].astype('int16')
                else:
                    df[col] = df[col].astype('int32')
        elif np.issubdtype(col_type, np.floating):
            df[col] = df[col].astype('float32')
            
    # Target and ID downcasting
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype('int32')
    df['IS_TRAIN'] = df['IS_TRAIN'].astype('int8')
    df['TARGET'] = df['TARGET'].astype('float32') # Float due to NaNs in test
    
    # Print summary
    print(f"Final master feature table shape: {df.shape}")
    mem_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Memory usage: {mem_usage:.2f} MB")
    
    # Save as parquet
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.MASTER_FEATURES, index=False)
    print(f"Saved master feature table to {config.MASTER_FEATURES}")
    print(f"Feature table compilation completed in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    build_features()
