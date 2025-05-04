from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
import numpy as np
import copy

def cross_validate_model(model, X_train, y_train, X_test, y_test, epochs=4):
    """
    Perform stratified k-fold cross-validation and test evaluation.

    Args:
        model: Initialized classifier (e.g., xgb.XGBClassifier(...))
        X_train, y_train: Training data (preprocessed)
        X_test, y_test: Final hold-out test data
        epochs: Number of CV folds (default: 4)
    """
    y_preds = np.zeros(X_test.shape[0])
    y_oof = np.zeros(X_train.shape[0])

    kf = StratifiedKFold(n_splits=epochs, shuffle=True, random_state=99)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"\n🚀 Fold {fold + 1}/{epochs}")

        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model_fold = copy.deepcopy(model)  # clone model with same params
        model_fold.fit(X_tr, y_tr)

        val_preds = model_fold.predict_proba(X_val)[:, 1]
        y_oof[val_idx] = val_preds

        test_preds = model_fold.predict_proba(X_test)[:, 1]
        y_preds += test_preds / epochs

        val_binary = (val_preds > 0.5).astype(int)
        print(f"Fold {fold + 1} - Recall: {recall_score(y_val, val_binary)}")
        print(f"Fold {fold + 1} - Precision: {precision_score(y_val, val_binary)}")
        print(f"Fold {fold + 1} - F1 Score: {f1_score(y_val, val_binary)}")

    print("\n📊 Final OOF Evaluation:")
    final_binary = (y_oof > 0.5).astype(int)
    print(f"OOF Recall: {recall_score(y_train, final_binary)}")
    print(f"OOF Precision: {precision_score(y_train, final_binary)}")
    print(f"OOF F1 Score: {f1_score(y_train, final_binary)}")

    y_test_preds = (y_preds > 0.5).astype(int)
    print("\n🧪 Final Test Evaluation:")
    print(f"Test Recall: {recall_score(y_test, y_test_preds)}")
    print(f"Test Precision: {precision_score(y_test, y_test_preds)}")
    print(f"Test F1 Score: {f1_score(y_test, y_test_preds)}")
    print(f"Test ROC AUC: {roc_auc_score(y_test, y_preds)}")
