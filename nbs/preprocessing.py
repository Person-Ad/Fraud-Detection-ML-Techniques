import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from typing import Tuple, List, Dict

# Feature engineering functions

def clean_data(X: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    X.replace([np.inf, -np.inf, -999], np.nan, inplace=True)
    X.fillna(X.mean(numeric_only=True), inplace=True)
    return X

def drop_low_information_columns(X: pd.DataFrame, threshold: float = 0.96) -> pd.DataFrame:
    high_missing = [col for col in X.columns if X[col].isnull().mean() > threshold]
    low_variance = [col for col in X.columns if X[col].value_counts(dropna=False, normalize=True).values[0] > threshold]
    cols_to_drop = list(set(high_missing + low_variance))
    return X.drop(columns=cols_to_drop, errors='ignore')

def encode_categorical_columns(X: pd.DataFrame) -> pd.DataFrame:
    cat_cols = X.select_dtypes(include=['object', 'category', 'O']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    return X

def group_rare_categories(X: pd.DataFrame, features: List[str], threshold: int = 500) -> pd.DataFrame:
    for col in features:
        if col in X.columns:
            freq = X[col].value_counts()
            rare = freq[freq < threshold].index
            X[col] = X[col].replace(rare, 'Rare')
    return X

def create_transaction_amount_ratios(X: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    for col in group_cols:
        if col in X.columns and 'TransactionAmt' in X.columns:
            means = X.groupby(col)['TransactionAmt'].mean()
            stds = X.groupby(col)['TransactionAmt'].std()
            X[f'TransactionAmt_to_mean_{col}'] = X['TransactionAmt'] / X[col].map(means)
            X[f'TransactionAmt_to_std_{col}'] = X['TransactionAmt'] / X[col].map(stds)
    return X

def fill_missing_values(X: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = reference.select_dtypes(include=[np.number]).columns.intersection(X.columns)
    X[numeric_cols] = X[numeric_cols].fillna(-999)
    
    cat_cols = reference.select_dtypes(include=['object']).columns.intersection(X.columns)
    X[cat_cols] = X[cat_cols].fillna('missing')
    return X

def create_time_features(X: pd.DataFrame) -> pd.DataFrame:
    if 'TransactionDT' in X.columns:
        X['TransactionDT_days'] = X['TransactionDT'] / (24 * 60 * 60)
        X['Transaction_hour'] = ((X['TransactionDT'] / 3600) % 24).astype(int)
        X['Transaction_weekday'] = ((X['TransactionDT'] / (3600 * 24)) % 7).astype(int)
        X['is_weekend'] = (X['Transaction_weekday'] >= 5).astype(int)
        X['is_nighttime'] = (X['Transaction_hour'].between(0, 5)).astype(int)
    return X

def drop_unused_columns(X: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ['TransactionID', 'id_34', 'id_07', 'id_08', 'id_18', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27']
    X = X.drop(columns=[col for col in drop_cols if col in X.columns], errors='ignore')
    return X

def log_transform_transaction_amt(X: pd.DataFrame) -> pd.DataFrame:
    if 'TransactionAmt' in X.columns:
        X['TransactionAmt_log'] = np.log1p(X['TransactionAmt'])
    return X

# Feature engineering function for one DataFrame with customizable config
def run_feature_engineering_single_df(X: pd.DataFrame, config: Dict) -> pd.DataFrame:
    print("🚧 Starting feature engineering pipeline...\n")

    # Apply configurable feature engineering steps
    if config.get('drop_low_information_columns', True):
        X = drop_low_information_columns(X, config.get('low_info_threshold', 0.96))
        print("✅ Low-information columns dropped")

    if config.get('drop_transaction_dt', True) and 'TransactionDT' in X.columns:
        X = X.drop(columns=['TransactionDT'])
        print("✅ TransactionDT dropped")

    if config.get('create_transaction_amount_ratios', False):
        X = create_transaction_amount_ratios(X, config.get('transaction_columns', []))
        print("✅ Transaction amount ratios created")

    if config.get('clean_data', True):
        X = clean_data(X, X)
        print("✅ Data cleaned")

    if config.get('group_rare_categories', False):
        X = group_rare_categories(X, config.get('rare_category_columns', []), threshold=config.get('rare_category_threshold', 500))
        print("✅ Rare categories grouped")

    if config.get('encode_categorical_columns', True):
        X = encode_categorical_columns(X)
        print("✅ Categorical columns encoded")

    if config.get('fill_missing_values', True):
        X = fill_missing_values(X, X)
        print("✅ Missing values filled")

    if config.get('create_time_features', True):
        X = create_time_features(X)
        print("✅ Time features created")

    if config.get('drop_unused_columns', True):
        X = drop_unused_columns(X)
        print("✅ Unused columns dropped")

    if config.get('log_transform_transaction_amt', False):
        X = log_transform_transaction_amt(X)
        print("✅ Log transformation applied to TransactionAmt")

    print(f"🎯 Final shape: {X.shape}")
    return X

# Evaluation function to search for the best configuration
def evaluate_model(X_train: pd.DataFrame, X_val: pd.DataFrame, model, y_train: pd.Series, y_val: pd.Series) -> float:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred)

# Example function to search for the best config
def search_best_config(X_train: pd.DataFrame, X_val: pd.DataFrame, model, y_train: pd.Series, y_val: pd.Series) -> Dict:
    best_f1 = -1
    best_config = {}
    
    # Example configurations to test
   # Recommended configurations for experimentation
    configs = [
        # Config 1: Baseline + log-transform (similar to best so far)
        {
            'create_transaction_amount_ratios': False,
            'clean_data': True,
            'group_rare_categories': False,
            'encode_categorical_columns': True,
            'fill_missing_values': True,
            'create_time_features': False,
            'drop_unused_columns': False,
            'log_transform_transaction_amt': True
        },

        # Config 2: Add time features only
        {
            'create_transaction_amount_ratios': False,
            'clean_data': True,
            'group_rare_categories': False,
            'encode_categorical_columns': True,
            'fill_missing_values': True,
            'create_time_features': True,
            'drop_unused_columns': False,
            'log_transform_transaction_amt': True
        },

        # Config 3: Add ratios + log-transform, no rare grouping
        {
            'create_transaction_amount_ratios': True,
            'clean_data': True,
            'group_rare_categories': False,
            'encode_categorical_columns': True,
            'fill_missing_values': True,
            'create_time_features': False,
            'drop_unused_columns': False,
            'log_transform_transaction_amt': True
        },

        # Config 4: Minimal preprocessing
        {
            'create_transaction_amount_ratios': False,
            'clean_data': True,
            'group_rare_categories': False,
            'encode_categorical_columns': True,
            'fill_missing_values': True,
            'create_time_features': False,
            'drop_unused_columns': False,
            'log_transform_transaction_amt': False
        },

        # Config 5: Everything but rare category grouping
        {
            'create_transaction_amount_ratios': True,
            'clean_data': True,
            'group_rare_categories': False,
            'encode_categorical_columns': True,
            'fill_missing_values': True,
            'create_time_features': True,
            'drop_unused_columns': True,
            'log_transform_transaction_amt': True
        },
    ]


    
    for config in configs:
        print(f"🔍 Evaluating config: {config}")
        X_train_processed = run_feature_engineering_single_df(X_train.copy(), config)
        X_val_processed = run_feature_engineering_single_df(X_val.copy(), config)
        
        f1 = evaluate_model(X_train_processed, X_val_processed, model, y_train, y_val)
        print(f"✅ F1-score: {f1}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_config = config
    
    print(f"🏆 Best config: {best_config} with F1-score: {best_f1}")
    return best_config

# Usage
# model = your_model_here
# best_config = search_best_config(X_train, X_val, model, y_train, y_val)


# ✅ Feature Engineering Insights:
# - Log-transforming `TransactionAmt` consistently improves performance — should be kept.
# - Avoid grouping rare categories — it likely removes useful signal and hurts model quality.
# - Time-based features (`Transaction_hour`, `is_weekend`, etc.) do not provide clear benefit here — consider skipping them.
# - Creating transaction amount ratios shows neutral effect — optional based on model complexity.
# - Best performance comes from a **simple yet focused** config: clean data, encode categoricals, fill missing values, and log-transform `TransactionAmt`.
