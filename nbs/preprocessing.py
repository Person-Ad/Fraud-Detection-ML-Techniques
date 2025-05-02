from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_data(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for X in [X_train, X_val, X_test]:
        X.replace([np.inf, -np.inf, -999], np.nan, inplace=True)
        X.fillna(X.mean(numeric_only=True), inplace=True)
    return X_train, X_val, X_test

def encode_categorical_columns(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'O']).columns
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_val[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        for X in [X_train, X_val, X_test]:
            if col in X.columns:
                X[col] = le.transform(X[col].astype(str))
    return X_train, X_val, X_test

def group_rare_categories(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, features: List[str], threshold: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for col in features:
        if col in X_train.columns:
            freq = X_train[col].value_counts()
            rare = freq[freq < threshold].index
            for X in [X_train, X_val, X_test]:
                if col in X.columns:
                    X[col] = X[col].replace(rare, 'Rare')
    return X_train, X_val, X_test

def create_transaction_amount_ratios(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, group_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for col in group_cols:
        if col in X_train.columns and 'TransactionAmt' in X_train.columns:
            means = X_train.groupby(col, observed=True)['TransactionAmt'].mean()
            stds = X_train.groupby(col, observed=True)['TransactionAmt'].std()
            
            for X in [X_train, X_val, X_test]:
                if col in X.columns:
                    X[f'TransactionAmt_to_mean_{col}'] = X['TransactionAmt'] / X[col].map(means)
                    X[f'TransactionAmt_to_std_{col}'] = X['TransactionAmt'] / X[col].map(stds)
    return X_train, X_val, X_test

def fill_missing_values(X: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = reference.select_dtypes(include=[np.number]).columns.intersection(X.columns)
    X[numeric_cols] = X[numeric_cols].fillna(-1)
    
    cat_cols = reference.select_dtypes(include=['object']).columns.intersection(X.columns)
    X[cat_cols] = X[cat_cols].fillna('missing')
    return X

def create_time_features(X: pd.DataFrame) -> pd.DataFrame:
    if 'TransactionDT' in X.columns:
        X = X.copy()  
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
        X = X.copy()  
        X['TransactionAmt_log'] = np.log1p(X['TransactionAmt'])
    return X

def run_feature_engineering(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("🚧 Starting feature engineering pipeline...\n")

    X_train, X_val, X_test = create_transaction_amount_ratios(X_train, X_val, X_test, ['card1', 'card4'])
    print("✅ Transaction amount ratios created")

    X_train, X_val, X_test = clean_data(X_train, X_val, X_test)
    print("✅ Data cleaned")
    
    X_train, X_val, X_test = group_rare_categories(
        X_train, X_val, X_test,
        features=['P_emaildomain', 'R_emaildomain', 'id_30', 'id_31', 'id_33', 'card2', 'card5']
    )
    print("✅ Rare categories grouped")
    
    X_train, X_val, X_test = encode_categorical_columns(X_train, X_val, X_test)
    print("✅ Categorical columns encoded")
    
    for name, X in zip(['Train', 'Validation', 'Test'], [X_train, X_val, X_test]):
        X = fill_missing_values(X, X_train)
        X = create_time_features(X)
        print(f"✅ {name} set processed")
        if name == 'Train':
            X_train = X
        elif name == 'Validation':
            X_val = X
        else:
            X_test = X
    
    X_train = drop_unused_columns(X_train)
    X_val = drop_unused_columns(X_val)
    X_test = drop_unused_columns(X_test)
    print("✅ Unused columns dropped")

    print("\n🎯 Final Shapes:")
    print(f"📐 X_train shape: {X_train.shape}")
    print(f"📐 X_val shape:   {X_val.shape}")
    print(f"📐 X_test shape:  {X_test.shape}")
    print(f"🔍 Columns in train but not in test: {set(X_train.columns) - set(X_test.columns)}")
    print(f"🔍 Columns in test but not in train: {set(X_test.columns) - set(X_train.columns)}")
    
    return X_train, X_val, X_test
