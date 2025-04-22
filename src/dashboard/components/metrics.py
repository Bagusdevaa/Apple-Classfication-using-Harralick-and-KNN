"""
Metrics utility functions for the dashboard
"""
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report as sk_classification_report
from src.preprocessing.preprocessing import preprocess_data
from src.classifier.knn import calculate_knn_results, calculate_kfold_cv_score

@pd.api.extensions.register_dataframe_accessor("metrics")
class MetricsAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        
    def calculate_confusion_matrix(self, y_true, y_pred, num_classes=5):
        """Calculate confusion matrix from true and predicted labels"""
        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            conf_matrix[t, p] += 1
        return conf_matrix

def preprocess_and_calculate_results(feature_dataframes, k_values):
    """
    Preprocess data and calculate KNN results for all d, theta combinations
    """
    summary_data = []
    all_results = {}

    for (d, theta), df in feature_dataframes.items():
        X_train, X_test, y_train, y_test = preprocess_data(df, drop_columns=['label','homogeneity','sum_variance', 'ASM','IMC2', 'image'])
        results = calculate_knn_results(X_train, X_test, y_train, y_test, k_values)
        all_results[(d, theta)] = results

        best_k = max(results, key=lambda k: results[k]['accuracy'])
        best_accuracy = results[best_k]['accuracy']
        summary_data.append({'d': d, 'theta': theta, 'best_k': best_k, 'accuracy': best_accuracy})

    return pd.DataFrame(summary_data), all_results

def calculate_classification_report(y_test, y_pred):
    """Calculate classification report from true and predicted labels"""
    return sk_classification_report(y_test, y_pred, output_dict=True)

def calculate_cross_validation_results(feature_dataframes, summary_df, n_splits=5):
    """
    Perform K-fold cross validation on the best k value for each (d, theta) combination
    """
    cv_results = {}
    
    for _, row in summary_df.iterrows():
        d = row['d']
        theta = row['theta']
        best_k = int(row['best_k'])  # Convert best_k to int
        
        # Get data for this combination
        df = feature_dataframes[(d, theta)]
        
        # Preprocess all data (don't split into train/test yet as CV will do that)
        X = df.drop(columns=['label', 'homogeneity', 'sum_variance', 'ASM', 'IMC2', 'image'])
        y = df['label']
        
        # Perform cross-validation
        cv_result = calculate_kfold_cv_score(X, y, best_k, n_splits=n_splits)
        cv_results[(d, theta, best_k)] = cv_result
    
    return cv_results