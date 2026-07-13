import pandas as pd
import numpy as np
import gc
import time
from pathlib import Path
import config

def aggregate_credit_card():
    start_time = time.time()
    cc_path = config.RAW_DIR / "credit_card_balance.csv"
    print(f"Loading {cc_path}...")
    
    # Check if raw file exists
    if not cc_path.exists():
        raise FileNotFoundError(f"Raw credit card file not found at {cc_path}")
        
    # Read the CSV with optimized dtypes to save memory
    cols = [
        'SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_BALANCE', 
        'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_CURRENT', 
        'AMT_DRAWINGS_ATM_CURRENT', 'AMT_INST_MIN_REGULARITY', 
        'AMT_PAYMENT_CURRENT', 'SK_DPD'
    ]
    
    dtypes = {
        'SK_ID_CURR': 'int32',
        'SK_ID_PREV': 'int32',
        'MONTHS_BALANCE': 'int16',
        'AMT_BALANCE': 'float32',
        'AMT_CREDIT_LIMIT_ACTUAL': 'float32',
        'AMT_DRAWINGS_CURRENT': 'float32',
        'AMT_DRAWINGS_ATM_CURRENT': 'float32',
        'AMT_INST_MIN_REGULARITY': 'float32',
        'AMT_PAYMENT_CURRENT': 'float32',
        'SK_DPD': 'int32'
    }
    
    cc = pd.read_csv(cc_path, usecols=cols, dtype=dtypes)
    print(f"Loaded credit card balance shape: {cc.shape}")
    
    # 1. CC_UTILIZATION_MEAN / MAX
    # Utilization = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL. Guard zero or division.
    cc['utilization'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
    
    # 3. Minimum payment miss ratio: share of months where AMT_PAYMENT_CURRENT < AMT_INST_MIN_REGULARITY
    # Both must be non-null.
    cc['missed_min_payment'] = (cc['AMT_PAYMENT_CURRENT'] < cc['AMT_INST_MIN_REGULARITY']).astype('float32')
    cc.loc[cc['AMT_PAYMENT_CURRENT'].isnull() | cc['AMT_INST_MIN_REGULARITY'].isnull(), 'missed_min_payment'] = np.nan
    
    # 4. CC_BALANCE_TREND
    # Sign of (mean balance in 3 most recent months − mean balance in 3 oldest months) per customer.
    # Sort ascending by MONTHS_BALANCE per customer (older months are first, e.g. -12, newer are last, e.g. -1).
    cc = cc.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])
    
    # Rank months per group
    cc['months_rank_desc'] = cc.groupby('SK_ID_CURR')['MONTHS_BALANCE'].rank(ascending=False, method='first')
    cc['months_rank_asc'] = cc.groupby('SK_ID_CURR')['MONTHS_BALANCE'].rank(ascending=True, method='first')
    
    # Mean balance of 3 most recent months:
    recent_3 = cc[cc['months_rank_desc'] <= 3].groupby('SK_ID_CURR')['AMT_BALANCE'].mean().rename('CC_BAL_RECENT_MEAN')
    # Mean balance of 3 oldest months:
    oldest_3 = cc[cc['months_rank_asc'] <= 3].groupby('SK_ID_CURR')['AMT_BALANCE'].mean().rename('CC_BAL_OLDEST_MEAN')
    
    # Combine them:
    trend_df = pd.concat([recent_3, oldest_3], axis=1)
    diff = trend_df['CC_BAL_RECENT_MEAN'] - trend_df['CC_BAL_OLDEST_MEAN']
    cc_trend = np.sign(diff).fillna(0).astype('int8').rename('CC_BALANCE_TREND')
    
    # Clean up trend-only columns to save memory
    cc.drop(columns=['months_rank_desc', 'months_rank_asc'], inplace=True)
    gc.collect()
    
    # 5. CC_MONTHS_COUNT, CC_DPD_MEAN
    # 6. CC_ATM_DRAWINGS_RATIO = sum(ATM_DRAWINGS) / sum(DRAWINGS)
    cc_agg = cc.groupby('SK_ID_CURR').agg(
        CC_UTILIZATION_MEAN=('utilization', 'mean'),
        CC_UTILIZATION_MAX=('utilization', 'max'),
        CC_DRAWINGS_MEAN=('AMT_DRAWINGS_CURRENT', 'mean'),
        CC_LIMIT_MEAN=('AMT_CREDIT_LIMIT_ACTUAL', 'mean'),
        CC_MIN_PAYMENT_MISS_RATIO=('missed_min_payment', 'mean'),
        CC_MONTHS_COUNT=('MONTHS_BALANCE', 'count'),
        CC_DPD_MEAN=('SK_DPD', 'mean'),
        CC_ATM_DRAWINGS_SUM=('AMT_DRAWINGS_ATM_CURRENT', 'sum'),
        CC_TOTAL_DRAWINGS_SUM=('AMT_DRAWINGS_CURRENT', 'sum')
    )
    
    # Calculate drawings limit ratio
    cc_agg['CC_DRAWINGS_LIMIT_RATIO'] = cc_agg['CC_DRAWINGS_MEAN'] / cc_agg['CC_LIMIT_MEAN'].replace(0, np.nan)
    # Calculate ATM drawings ratio
    cc_agg['CC_ATM_DRAWINGS_RATIO'] = cc_agg['CC_ATM_DRAWINGS_SUM'] / cc_agg['CC_TOTAL_DRAWINGS_SUM'].replace(0, np.nan)
    
    # Merge the trend
    cc_agg = cc_agg.join(cc_trend, how='left')
    
    # Drop intermediate columns
    cc_agg.drop(columns=['CC_DRAWINGS_MEAN', 'CC_LIMIT_MEAN', 'CC_ATM_DRAWINGS_SUM', 'CC_TOTAL_DRAWINGS_SUM'], inplace=True)
    
    # Reset index and downcast types
    cc_agg = cc_agg.reset_index()
    cc_agg['SK_ID_CURR'] = cc_agg['SK_ID_CURR'].astype('int32')
    cc_agg['CC_MONTHS_COUNT'] = cc_agg['CC_MONTHS_COUNT'].astype('int32')
    cc_agg['CC_BALANCE_TREND'] = cc_agg['CC_BALANCE_TREND'].astype('int8')
    
    for col in cc_agg.columns:
        if cc_agg[col].dtype == 'float64':
            cc_agg[col] = cc_agg[col].astype('float32')
            
    print(f"Final credit card aggregation shape: {cc_agg.shape}")
    print(cc_agg.head())
    
    # Ensure processed directory exists
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet
    cc_agg.to_parquet(config.CC_AGG, index=False)
    print(f"Saved aggregated credit card data to {config.CC_AGG}")
    print(f"Aggregation completed in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    aggregate_credit_card()
