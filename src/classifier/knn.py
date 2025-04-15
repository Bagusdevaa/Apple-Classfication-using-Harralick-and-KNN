import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score

def calculate_knn_results(X_train, X_test, y_train, y_test, k_values):
    """
    Train and evaluate KNN for different k values.
    """
    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)  # Get precision, recall, F1-score
        results[k] = {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': report
        }
    return results

## ----- K-Fold Cross Validation Functions -----
def calculate_kfold_cv_score(X, y, k_value, n_splits=5, random_state=42):
    """
    Perform K-fold cross validation for a specific k value in KNN.
    
    Parameters:
    -----------
    X : array-like 
        Features data
    y : array-like
        Target values
    k_value : int
        The k value for KNN (number of neighbors)
    n_splits : int, default=5
        Number of folds for cross validation
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    dict: Dictionary with cross validation results
    """
    knn = KNeighborsClassifier(n_neighbors=k_value)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Calculate accuracy scores
    cv_scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
    
    # Create detailed results for each fold
    fold_results = []
    fold_indices = list(cv.split(X))
    
    for i, (train_idx, test_idx) in enumerate(fold_indices):
        X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
        y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train and predict
        knn.fit(X_fold_train, y_fold_train)
        y_fold_pred = knn.predict(X_fold_test)
        
        # Calculate metrics
        fold_accuracy = accuracy_score(y_fold_test, y_fold_pred)
        fold_cm = confusion_matrix(y_fold_test, y_fold_pred)
        fold_report = classification_report(y_fold_test, y_fold_pred, output_dict=True)
        
        fold_results.append({
            'fold': i+1,
            'accuracy': fold_accuracy,
            'confusion_matrix': fold_cm,
            'classification_report': fold_report
        })
    
    return {
        'cv_scores': cv_scores,
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std(),
        'fold_results': fold_results
    }
## ----- K-Fold Cross Validation Functions END -----

# # Fungsi untuk menghitung jarak (Euclidean Distance)
# def euclidean_distance(point1, point2):
#     point1 = np.array(point1, dtype=np.float64)
#     point2 = np.array(point2, dtype=np.float64)
#     return np.sqrt(np.sum((point1 - point2) ** 2))

# # Kelas KNN
# class KNN:
#     def __init__(self, k=3):
#         self.k = k
#         self.data = None  # DataFrame
#         self.labels = None  # Pandas Series

#     # Metode untuk melatih model
#     def fit(self, data, labels):
#         self.data = data.reset_index(drop=True)  # Pastikan index sesuai
#         self.labels = labels.reset_index(drop=True)

#     # Metode untuk memprediksi kelas dari titik baru
#     def predict(self, new_points):
#         predictions = []
#         new_points = new_points.to_numpy(dtype=np.float64)  # Konversi menjadi numpy array

#         for new_point in new_points:
#             # Hitung jarak dari titik baru ke semua titik dalam data
#             distances = []
#             for i in range(len(self.data)): 
#                 distance = euclidean_distance(self.data.iloc[i].values, new_point)
#                 distances.append((distance, self.labels.iloc[i]))
            
#             # Urutkan berdasarkan jarak
#             distances.sort(key=lambda x: x[0])
            
#             # Ambil K tetangga terdekat
#             neighbors = distances[:self.k]
            
#             # Lakukan voting untuk menentukan kelas
#             votes = {}
#             for _, label in neighbors:
#                 votes[label] = votes.get(label, 0) + 1
            
#             # Temukan kelas dengan jumlah suara terbanyak
#             most_common = max(votes.items(), key=lambda x: x[1])[0]
            
#             predictions.append(most_common)
        
#         return predictions