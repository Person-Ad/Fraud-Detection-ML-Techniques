import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import run_feature_engineering_single_df, search_best_config
from evaluation import evaluate_logistic_regression, evaluate_random_forest_estimators, evaluate_model, evaluate_multiple_models, evaluate_svm
import os
import pickle


# Configuration
DATA_PATH = "data/"
RESULTS_PATH = "results/"
CONFIG = {
    'create_transaction_amount_ratios': False,
    'clean_data': True,
    'group_rare_categories': False,
    'encode_categorical_columns': True,
    'fill_missing_values': True,
    'create_time_features': False,
    'drop_unused_columns': False,
    'log_transform_transaction_amt': True,
    'standardize_numeric': True,
    'apply_pca': False,
    'n_pca_components': 100
}

# Create results directories
os.makedirs(os.path.join(RESULTS_PATH, "plots"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, "metrics"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, "models"), exist_ok=True)

def load_data(data_path):
    """Load and split dataset."""
    train_tx = pd.read_csv(data_path + "train_transaction.csv")
    train_identity = pd.read_csv(data_path + "train_identity.csv")
    train_all_cols = pd.merge(train_tx, train_identity, on='TransactionID', how='left')
    X =  train_all_cols.drop(columns=['isFraud', 'TransactionID'])
    y = train_all_cols['isFraud']
    return X, y

def preprocess_data(X, config):
    """Apply preprocessing pipeline."""
    return run_feature_engineering_single_df(X, config)

def save_results(results, filename, result_type="metrics"):
    """Save results to file."""
    if result_type == "metrics":
        with open(os.path.join(RESULTS_PATH, "metrics", filename), 'w') as f:
            f.write(str(results))
    elif result_type == "model":
        with open(os.path.join(RESULTS_PATH, "models", filename), 'wb') as f:
            pickle.dump(results, f)

def main():
    # Step 1: Load and split data
    print("Loading data...")
    X, y = load_data(DATA_PATH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Step 2: Preprocess data
    print("Preprocessing data...")
    X_train_processed = preprocess_data(X_train.copy(), CONFIG)
    X_val_processed = preprocess_data(X_val.copy(), CONFIG)

    
    print("Evaluating Logistic Regression...")
    lr_results = evaluate_logistic_regression(
        X_train_processed, y_train, X_val_processed, y_val,
        C_range=[0.001, 0.1, 5, 100],
        penalty_types=['l1', 'l2'],
        cv=None,
        subsample_fraction=0.2
    )
    
    # Evaluate the best Logistic Regression model in detail
    best_lr = lr_results['best_model']
    lr_detailed_results = evaluate_model(
        model=best_lr,
        X_val=X_val_processed,
        y_val=y_val,
        model_name="Logistic Regression",
        class_names=['Non-Fraud', 'Fraud']
    )
    save_results(lr_detailed_results, "lr_metrics.txt")
    save_results(best_lr, "logistic_regression_model.pkl", result_type="model")

    
    print("Evaluating SVM...")
    svm_results = evaluate_svm(
        X_train_processed, y_train, X_val_processed, y_val,
        C_range=[0.01, 0.1, 1, 10],
        kernel_types=['linear', 'rbf'],
        cv=None,
        subsample_fraction=0.1
    )

    # Evaluate the best SVM model in detail
    best_svm = svm_results['best_model']
    svm_detailed_results = evaluate_model(
        model=best_svm,
        X_val=X_val_processed,
        y_val=y_val,
        model_name="SVM",
        class_names=['Non-Fraud', 'Fraud']
    )
    save_results(svm_detailed_results, "svm_metrics.txt")
    save_results(best_svm, "svm_model.pkl", result_type="model")


    
    # Step 4: Evaluate Random Forest
    print("Evaluating Random Forest...")
    rf_results = evaluate_random_forest_estimators(
        X_train_processed, y_train, X_val_processed, y_val,
        estimator_range=[50, 100, 200],
        cv=5
    )
    
    # Evaluate the best Random Forest model in detail
    best_rf = rf_results['best_model']
    rf_detailed_results = evaluate_model(
        model=best_rf,
        X_val=X_val_processed,
        y_val=y_val,
        model_name="Random Forest",
        class_names=['Non-Fraud', 'Fraud']
    )
    save_results(rf_detailed_results, "rf_metrics.txt")
    save_results(best_rf, "random_forest_model.pkl", result_type="model")
    
    # Step 5: Compare models
    print("Comparing models...")
    models = {
        "Logistic Regression": lr_results['best_model'],
        "Random Forest": rf_results['best_model'],
        "SVM": svm_results['best_model']
    }
    comparison_results = evaluate_multiple_models(
        models=models,
        X_val=X_val_processed,
        y_val=y_val,
        threshold=0.5
    )
    save_results(comparison_results, "model_comparison.txt")
    
    print("Pipeline completed! Results saved in", RESULTS_PATH)

if __name__ == "__main__":
    main()