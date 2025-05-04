from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score,
    classification_report, log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Any, Tuple, Union, Optional
from sklearn.utils import resample
import os


def evaluate_random_forest_estimators(X_train, y_train, X_val, y_val, estimator_range, 
                                     max_depth_range=None, cv=None):
    """
    Evaluate Random Forest performance across different numbers of estimators
    and optionally different max_depth values.
    
    Parameters:
    -----------
    X_train : array-like
        Training feature data
    y_train : array-like
        Training target data
    X_val : array-like
        Validation feature data
    y_val : array-like
        Validation target data
    estimator_range : list
        List of n_estimators values to evaluate
    max_depth_range : list, optional
        List of max_depth values to evaluate (creates a grid search if provided)
    cv : int, optional
        Number of cross-validation folds (if None, no CV is performed)
        
    Returns:
    --------
    Dict containing scores and best parameters
    """
    results = {}
    best_auc = 0
    best_params = {}
    
    if max_depth_range is None:
        # Only vary n_estimators
        scores = []
        
        for n in estimator_range:
            rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)

            y_val_pred_proba = rf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_val_pred_proba)
            scores.append(auc)
            print(f"n_estimators={n} => AUC: {auc:.4f}")
            
            # Track best parameters
            if auc > best_auc:
                best_auc = auc
                best_params = {'n_estimators': n}
        
        # Plot results
        plt.figure(figsize=(10, 5))
        plt.plot(estimator_range, scores, marker='o')
        plt.title('Random Forest AUC vs. n_estimators')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Validation AUC')
        plt.grid(True)
        plt.show()
        
        results['estimator_scores'] = scores
        
    else:
        # Grid search over n_estimators and max_depth
        grid_scores = np.zeros((len(max_depth_range), len(estimator_range)))
        
        for i, depth in enumerate(max_depth_range):
            for j, n in enumerate(estimator_range):
                rf = RandomForestClassifier(
                    n_estimators=n, 
                    max_depth=depth,
                    random_state=42, 
                    n_jobs=-1
                )
                rf.fit(X_train, y_train)
                
                y_val_pred_proba = rf.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_val_pred_proba)
                grid_scores[i, j] = auc
                print(f"n_estimators={n}, max_depth={depth} => AUC: {auc:.4f}")
                
                # Track best parameters
                if auc > best_auc:
                    best_auc = auc
                    best_params = {'n_estimators': n, 'max_depth': depth}
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(grid_scores, annot=True, fmt='.4f', 
                   xticklabels=estimator_range, 
                   yticklabels=max_depth_range)
        plt.title('Random Forest AUC')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Max Depth')
        plt.tight_layout()
        plt.show()
        
        results['grid_scores'] = grid_scores
    
    # Cross-validation if requested
    if cv is not None:
        from sklearn.model_selection import cross_val_score
        cv_results = {}
        
        # Use best parameters if grid search was done
        n_estimators = best_params.get('n_estimators', estimator_range[-1])
        max_depth = best_params.get('max_depth', None)
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"\nCross-validation AUC with {cv} folds:")
        print(f"Mean: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")
        print(f"CV Scores: {cv_scores}")
        
        cv_results['scores'] = cv_scores
        cv_results['mean'] = cv_scores.mean()
        cv_results['std'] = cv_scores.std()
        results['cv_results'] = cv_results
    
    results['best_auc'] = best_auc
    results['best_params'] = best_params
    
    # Feature importance for best model
    best_n_estimators = best_params.get('n_estimators', estimator_range[-1])
    best_max_depth = best_params.get('max_depth', None)
    
    best_rf = RandomForestClassifier(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        random_state=42,
        n_jobs=-1
    )
    best_rf.fit(X_train, y_train)
    
    # Plot feature importance
    feature_importances = best_rf.feature_importances_
    if hasattr(X_train, 'columns'):  # If X_train is a DataFrame
        features = X_train.columns
    else:
        features = [f'Feature {i}' for i in range(X_train.shape[1])]
    
    # Sort feature importances
    sorted_idx = np.argsort(feature_importances)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.show()
    
    results['feature_importances'] = dict(zip(features, feature_importances))
    results['best_model'] = best_rf
    
    return results

def evaluate_model(model, X_val, y_val, X_test=None, y_test=None, threshold=0.5, 
                  model_name="Model", is_lightgbm=False, class_names=None):
    results = {}
    datasets = [('Validation', X_val, y_val)]
    
    if X_test is not None and y_test is not None:
        datasets.append(('Test', X_test, y_test))
    
    # Set default class names if not provided
    if class_names is None:
        class_names = ['Negative (0)', 'Positive (1)']
    
    # Function to get predictions based on model type
    def get_predictions(model, X, is_lightgbm):
        if is_lightgbm:
            # Handle different LightGBM model formats
            if hasattr(model, 'predict'):
                # Try different LightGBM model variations
                if hasattr(model, 'best_iteration_'):
                    y_proba = model.predict(X, num_iteration=model.best_iteration_)
                elif hasattr(model, 'best_iteration'):
                    y_proba = model.predict(X, num_iteration=model.best_iteration)
                else:
                    # If best_iteration is not available, don't specify it
                    y_proba = model.predict(X)
            else:
                # Fall back to generic predict method
                y_proba = model.predict(X)
        else:
            # For sklearn models with predict_proba
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X)[:, 1]
            else:
                # For models with only predict method
                y_proba = model.predict(X)
        
        # Ensure y_proba is the right shape for binary classification
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1]  # Take the probability of class 1
        
        y_pred = (y_proba >= threshold).astype(int)
        return y_proba, y_pred
    
    all_probas = []
    all_true = []
    
    # Evaluate on each dataset
    for dataset_name, X, y in datasets:
        y_proba, y_pred = get_predictions(model, X, is_lightgbm)
        all_probas.append(y_proba)
        all_true.append(y)
        
        # Calculate metrics
        metrics = {
            'auc': roc_auc_score(y, y_proba),
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'log_loss': log_loss(y, y_proba),
            'brier_score': brier_score_loss(y, y_proba),
            'average_precision': average_precision_score(y, y_proba),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }
        
        # Print main metrics
        print(f"\n{dataset_name} Metrics:")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Log Loss: {metrics['log_loss']:.4f}")
        print(f"Brier Score: {metrics['brier_score']:.4f}")
        print(f"Average Precision: {metrics['average_precision']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])
        
        # Store results
        results[dataset_name.lower()] = {
            k: v for k, v in metrics.items() if k != 'proba'
        }
    
    # VISUALIZATIONS
    
    # 1. ROC Curve
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red']
    
    for i, (dataset_name, X, y) in enumerate(datasets):
        y_proba = all_probas[i]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{dataset_name} (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 2. Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    
    for i, (dataset_name, X, y) in enumerate(datasets):
        y_proba = all_probas[i]
        precision, recall, _ = precision_recall_curve(y, y_proba)
        avg_prec = average_precision_score(y, y_proba)
        
        plt.plot(recall, precision, color=colors[i], lw=2,
                label=f'{dataset_name} (AP = {avg_prec:.4f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 3. Confusion Matrix Heatmap
    for i, (dataset_name, X, y) in enumerate(datasets):
        cm = results[dataset_name.lower()]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix - {model_name} ({dataset_name})')
        plt.tight_layout()
        plt.show()
    
    # 4. Calibration Curve (Reliability Diagram)
    plt.figure(figsize=(10, 8))
    
    for i, (dataset_name, X, y) in enumerate(datasets):
        y_proba = all_probas[i]
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, y_proba, n_bins=10)
        
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                color=colors[i], label=f"{dataset_name}")
    
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve - {model_name}')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 5. Probability Distribution by Class
    # for i, (dataset_name, X, y) in enumerate(datasets):
    #     y_proba = all_probas[i]
        
    #     plt.figure(figsize=(10, 6))
    #     for j, cls in enumerate([0, 1]):
    #         plt.hist(y_proba[y == cls], bins=25, alpha=0.5, 
    #                 label=f'Class {cls} ({class_names[j]})')
        
    #     plt.axvline(x=threshold, color='r', linestyle='--', 
    #                label=f'Threshold: {threshold}')
    #     plt.xlabel('Predicted Probability')
    #     plt.ylabel('Count')
    #     plt.title(f'Probability Distribution by Class - {model_name} ({dataset_name})')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.show()
    
    # 6. Threshold Analysis
    thresholds = np.linspace(0.01, 0.99, 99)
    threshold_metrics = {'threshold': thresholds}
    
    # First dataset only (validation)
    y = all_true[0]
    y_proba = all_probas[0]
    
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precisions.append(precision_score(y, y_pred, zero_division=0))
        recalls.append(recall_score(y, y_pred, zero_division=0))
        f1_scores.append(f1_score(y, y_pred, zero_division=0))
        accuracies.append(accuracy_score(y, y_pred))
    
    threshold_metrics['precision'] = precisions
    threshold_metrics['recall'] = recalls
    threshold_metrics['f1'] = f1_scores
    threshold_metrics['accuracy'] = accuracies
    
    # Plot threshold impact
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.axvline(x=threshold, color='r', linestyle='--', 
               label=f'Current Threshold: {threshold}')
    
    # Find optimal F1 threshold
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_f1_idx]
    plt.axvline(x=best_f1_threshold, color='g', linestyle='--',
               label=f'Best F1 Threshold: {best_f1_threshold:.2f}')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Threshold Impact Analysis - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    results['threshold_analysis'] = threshold_metrics
    results['optimal_threshold'] = {
        'f1': best_f1_threshold,
        'f1_score': f1_scores[best_f1_idx]
    }
    
    # 7. Feature Importance (if the model supports it)
    if hasattr(model, 'feature_importances_'):
        # Get feature names if available
        if hasattr(X_val, 'columns'):
            feature_names = X_val.columns
        else:
            feature_names = [f'Feature {i}' for i in range(X_val.shape[1])]
        
        # Sort feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot top 30 features or fewer if less available
        n_top_features = min(30, len(feature_names))
        top_indices = indices[:n_top_features]
        
        plt.figure(figsize=(12, 10))
        plt.title(f'Top {n_top_features} Feature Importances - {model_name}')
        plt.barh(range(n_top_features), importances[top_indices], align='center')
        plt.yticks(range(n_top_features), [feature_names[i] for i in top_indices])
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        results['feature_importances'] = {
            feature_names[i]: importances[i] for i in range(len(feature_names))
        }
    
    return results


def evaluate_multiple_models(models, X_val, y_val, X_test=None, y_test=None, threshold=0.5):
    """
    Compare multiple models side by side.
    
    Parameters:
    -----------
    models : dict
        Dictionary of {model_name: model_object}
    X_val : array-like
        Validation feature data
    y_val : array-like
        Validation target data
    X_test : array-like, optional
        Test feature data
    y_test : array-like, optional
        Test target data
    threshold : float, default=0.5
        Classification threshold
        
    Returns:
    --------
    Dict containing comparison results
    """
    comparison = {}
    metrics_to_compare = ['auc', 'accuracy', 'precision', 'recall', 'f1', 
                          'log_loss', 'brier_score', 'average_precision']
    
    for dataset_name, X, y in [('validation', X_val, y_val)] + \
                           ([('test', X_test, y_test)] if X_test is not None else []):
        dataset_results = {metric: [] for metric in metrics_to_compare}
        dataset_results['model_name'] = []
        
        # Collect metrics for all models
        for model_name, model in models.items():
            is_lightgbm = 'lightgbm' in str(type(model)).lower()
            
            if is_lightgbm:
                y_proba = model.predict(X, num_iteration=model.best_iteration_)
            else:
                y_proba = model.predict_proba(X)[:, 1]
                
            y_pred = (y_proba >= threshold).astype(int)
            
            dataset_results['model_name'].append(model_name)
            dataset_results['auc'].append(roc_auc_score(y, y_proba))
            dataset_results['accuracy'].append(accuracy_score(y, y_pred))
            dataset_results['precision'].append(precision_score(y, y_pred, zero_division=0))
            dataset_results['recall'].append(recall_score(y, y_pred, zero_division=0))
            dataset_results['f1'].append(f1_score(y, y_pred, zero_division=0))
            dataset_results['log_loss'].append(log_loss(y, y_proba))
            dataset_results['brier_score'].append(brier_score_loss(y, y_proba))
            dataset_results['average_precision'].append(average_precision_score(y, y_proba))
            
        # Convert to DataFrame for easier manipulation
        results_df = pd.DataFrame(dataset_results)
        comparison[dataset_name] = results_df
        
        # Print comparison table
        print(f"\n{dataset_name.capitalize()} Results:")
        print(results_df.set_index('model_name').round(4))
        
        # Plot comparison bar charts
        plt.figure(figsize=(15, 10))
        
        for i, metric in enumerate(metrics_to_compare):
            plt.subplot(3, 3, i+1)
            sns.barplot(x='model_name', y=metric, data=results_df)
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.xticks(rotation=45)
            plt.ylim(max(0, results_df[metric].min() - 0.1), 
                    min(1, results_df[metric].max() + 0.1))
        
        plt.tight_layout()
        plt.suptitle(f'Model Comparison - {dataset_name.capitalize()}', y=1.02, fontsize=16)
        plt.show()
        
        # ROC Curve comparison
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models.items():
            is_lightgbm = 'lightgbm' in str(type(model)).lower()
            
            if is_lightgbm:
                y_proba = model.predict(X, num_iteration=model.best_iteration_)
            else:
                y_proba = model.predict_proba(X)[:, 1]
                
            fpr, tpr, _ = roc_curve(y, y_proba)
            auc = roc_auc_score(y, y_proba)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc:.4f})')
        
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve Comparison - {dataset_name.capitalize()}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Precision-Recall curve comparison
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models.items():
            is_lightgbm = 'lightgbm' in str(type(model)).lower()
            
            if is_lightgbm:
                y_proba = model.predict(X, num_iteration=model.best_iteration_)
            else:
                y_proba = model.predict_proba(X)[:, 1]
                
            precision, recall, _ = precision_recall_curve(y, y_proba)
            avg_prec = average_precision_score(y, y_proba)
            plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {avg_prec:.4f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve Comparison - {dataset_name.capitalize()}')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return comparison

def evaluate_logistic_regression(X_train, y_train, X_val, y_val, 
                                C_range=None, penalty_types=None, cv=None,
                                results_path="results/plots", subsample_fraction=1.0):
    """
    Evaluate Logistic Regression with optimized performance.
    
    Parameters:
    -----------
    X_train : array-like
        Training feature data
    y_train : array-like
        Training target data
    X_val : array-like
        Validation feature data
    y_val : array-like
        Validation target data
    C_range : list, optional
        List of inverse regularization strength values (default: [0.01, 0.1, 1, 10])
    penalty_types : list, optional
        List of penalty types (default: ['l1', 'l2'])
    cv : int, optional
        Number of cross-validation folds (if None, no CV is performed)
    results_path : str, optional
        Directory to save plots (default: "results/plots")
    subsample_fraction : float, optional
        Fraction of training data to use for grid search (default: 1.0)
        
    Returns:
    --------
    Dict containing scores and best parameters
    """
    if C_range is None:
        C_range = [0.01, 0.1, 1, 10]
    if penalty_types is None:
        penalty_types = ['l1', 'l2']
    
    results = {}
    best_auc = 0
    best_params = {}
    
    os.makedirs(results_path, exist_ok=True)
    
    # Subsample training data
    if subsample_fraction < 1.0:
        print(f"Subsampling {subsample_fraction*100:.1f}% of training data for grid search...")
        X_train_sub, y_train_sub = resample(X_train, y_train, 
                                          n_samples=int(len(X_train) * subsample_fraction),
                                          stratify=y_train, random_state=42)
    else:
        X_train_sub, y_train_sub = X_train, y_train
    
    # Grid search with GridSearchCV
    param_grid = {'C': C_range, 'penalty': penalty_types}
    lr = LogisticRegression(solver='saga', random_state=42, max_iter=5000, tol=1e-3, class_weight='balanced')
    grid_search = GridSearchCV(
        lr, param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=1
    )
    print("Starting grid search with GridSearchCV...")
    grid_search.fit(X_train_sub, y_train_sub)
    
    # Extract grid scores
    grid_scores = np.zeros((len(param_grid['penalty']), len(param_grid['C'])))
    for i, penalty in enumerate(param_grid['penalty']):
        for j, C in enumerate(param_grid['C']):
            idx = np.where((grid_search.cv_results_['param_C'] == C) & 
                          (grid_search.cv_results_['param_penalty'] == penalty))[0]
            if len(idx) > 0:
                grid_scores[i, j] = grid_search.cv_results_['mean_test_score'][idx[0]]
    
    # Evaluate best model on validation set
    best_params = grid_search.best_params_
    best_auc = roc_auc_score(y_val, grid_search.best_estimator_.predict_proba(X_val)[:, 1])
    print(f"Best parameters: {best_params}, Validation AUC: {best_auc:.4f}")
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(grid_scores, annot=True, fmt='.4f', 
                xticklabels=param_grid['C'], 
                yticklabels=param_grid['penalty'])
    plt.title('Logistic Regression AUC (GridSearchCV)')
    plt.xlabel('C (Inverse Regularization Strength)')
    plt.ylabel('Penalty Type')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "logistic_regression_auc_heatmap.png"))
    plt.show()
    plt.close()
    
    results['grid_scores'] = grid_scores
    
    # Cross-validation on full data
    if cv is not None:
        print(f"\nPerforming {cv}-fold cross-validation with best parameters: {best_params}")
        lr = LogisticRegression(
            C=best_params['C'],
            penalty=best_params['penalty'],
            solver='saga',
            random_state=42,
            max_iter=5000,
            tol=1e-3,
            class_weight='balanced'
        )
        cv_scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"Cross-validation AUC with {cv} folds:")
        print(f"Mean: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")
        print(f"CV Scores: {cv_scores}")
        
        results['cv_results'] = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
    
    # Train best model on full data
    print("\nTraining best Logistic Regression model on full data...")
    best_lr = LogisticRegression(
        C=best_params['C'],
        penalty=best_params['penalty'],
        solver='saga',
        random_state=42,
        max_iter=5000,
        tol=1e-3,
        class_weight='balanced'
    )
    best_lr.fit(X_train, y_train)
    
    # Feature importance (coefficients)
    if hasattr(X_train, 'columns'):
        features = X_train.columns
    else:
        features = [f'Feature {i}' for i in range(X_train.shape[1])]
    
    coefficients = best_lr.coef_[0]
    sorted_idx = np.argsort(np.abs(coefficients))
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), coefficients[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.title('Logistic Regression Feature Coefficients')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "logistic_regression_coefficients.png"))
    plt.show()
    plt.close()
    
    results['feature_importances'] = dict(zip(features, coefficients))
    results['best_auc'] = best_auc
    results['best_params'] = best_params
    results['best_model'] = best_lr
    
    print("Logistic Regression evaluation completed!")
    return results

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_svm(X_train, y_train, X_val, y_val, 
                C_range=None, kernel_types=None, cv=None,
                results_path="results/plots", subsample_fraction=1.0):
    """
    Evaluate Support Vector Machine (SVM) performance across different C values and kernel types.
    
    Parameters:
    -----------
    X_train : array-like
        Training feature data
    y_train : array-like
        Training target data
    X_val : array-like
        Validation feature data
    y_val : array-like
        Validation target data
    C_range : list, optional
        List of regularization parameter values (default: [0.01, 0.1, 1, 10])
    kernel_types : list, optional
        List of kernel types (default: ['linear', 'rbf'])
    cv : int, optional
        Number of cross-validation folds (if None, no CV is performed)
    results_path : str, optional
        Directory to save plots (default: "results/plots")
    subsample_fraction : float, optional
        Fraction of training data to use for grid search (default: 1.0)
        
    Returns:
    --------
    Dict containing scores and best parameters
    """
    if C_range is None:
        C_range = [0.01, 0.1, 1, 10]
    if kernel_types is None:
        kernel_types = ['linear', 'rbf']
    
    results = {}
    best_auc = 0
    best_params = {}
    
    os.makedirs(results_path, exist_ok=True)
    
    # Subsample training data
    if subsample_fraction < 1.0:
        print(f"Subsampling {subsample_fraction*100:.1f}% of training data for grid search...")
        X_train_sub, y_train_sub = resample(X_train, y_train, 
                                            n_samples=int(len(X_train) * subsample_fraction),
                                            stratify=y_train, random_state=42)
    else:
        X_train_sub, y_train_sub = X_train, y_train
    
    # Grid search with GridSearchCV
    param_grid = {'C': C_range, 'kernel': kernel_types}
    svm = SVC(probability=True, random_state=42, max_iter=-1)  # probability=True for predict_proba
    grid_search = GridSearchCV(
        svm, param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=1
    )
    print("Starting grid search with GridSearchCV...")
    grid_search.fit(X_train_sub, y_train_sub)
    
    # Extract grid scores
    grid_scores = np.zeros((len(param_grid['kernel']), len(param_grid['C'])))
    for i, kernel in enumerate(param_grid['kernel']):
        for j, C in enumerate(param_grid['C']):
            idx = np.where((grid_search.cv_results_['param_C'] == C) & 
                           (grid_search.cv_results_['param_kernel'] == kernel))[0]
            if len(idx) > 0:
                grid_scores[i, j] = grid_search.cv_results_['mean_test_score'][idx[0]]
    
    # Evaluate best model on validation set
    best_params = grid_search.best_params_
    best_auc = roc_auc_score(y_val, grid_search.best_estimator_.predict_proba(X_val)[:, 1])
    print(f"Best parameters: {best_params}, Validation AUC: {best_auc:.4f}")
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(grid_scores, annot=True, fmt='.4f', 
                xticklabels=param_grid['C'], 
                yticklabels=param_grid['kernel'])
    plt.title('SVM AUC (GridSearchCV)')
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Kernel Type')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "svm_auc_heatmap.png"))
    plt.show()
    plt.close()
    
    results['grid_scores'] = grid_scores
    
    # Cross-validation on full data
    if cv is not None:
        print(f"\nPerforming {cv}-fold cross-validation with best parameters: {best_params}")
        svm = SVC(
            C=best_params['C'],
            kernel=best_params['kernel'],
            probability=True,
            random_state=42,
            max_iter=-1
        )
        cv_scores = cross_val_score(svm, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"Cross-validation AUC with {cv} folds:")
        print(f"Mean: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")
        print(f"CV Scores: {cv_scores}")
        
        results['cv_results'] = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
    
    # Train best model on full data
    print("\nTraining best SVM model on full data...")
    best_svm = SVC(
        C=best_params['C'],
        kernel=best_params['kernel'],
        probability=True,
        random_state=42,
        max_iter=-1
    )
    best_svm.fit(X_train, y_train)
    
    # Feature importance (coefficients for linear kernel, or approximation for non-linear)
    if hasattr(X_train, 'columns'):
        features = X_train.columns
    else:
        features = [f'Feature {i}' for i in range(X_train.shape[1])]
    
    if best_params['kernel'] == 'linear':
        coefficients = best_svm.coef_[0]
    else:
        # Approximate feature importance for non-linear kernels using permutation importance
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(best_svm, X_val, y_val, n_repeats=10, random_state=42)
        coefficients = perm_importance.importances_mean
    
    sorted_idx = np.argsort(np.abs(coefficients))
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), coefficients[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.title('SVM Feature Importance (Coefficients or Permutation Importance)')
    plt.xlabel('Coefficient/Permutation Importance Value')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "svm_feature_importance.png"))
    plt.show()
    plt.close()
    
    results['feature_importances'] = dict(zip(features, coefficients))
    results['best_auc'] = best_auc
    results['best_params'] = best_params
    results['best_model'] = best_svm
    
    print("SVM evaluation completed!")
    return results