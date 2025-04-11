import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

## load dataset
BASE_DIR = os.path.abspath(os.path.join("..", ".."))  # Naik dua level ke root proyek
# path ke dataset
dataset_path = os.path.join(BASE_DIR, "dataset") # Agar folder dataset dapat dibaca
# Kombinasi d dan theta
kombinasiFeature = [[1, 2, 3], [0, 45, 90, 135]]  # kombinasi jarak dan sudut matriks GLCM

# Dictionary untuk menyimpan kombinasi dari setiap dataset
# CARA AKSES feature_dataframes[(d, theta)] untuk mendapatkan dataframe
feature_dataframes = {}

# for d in kombinasiFeature[0]:
#     for theta in kombinasiFeature[1]:
#         data_fitur = []  # List untuk menyimpan fitur semua gambar untuk kombinasi ini
#         img = [index for index in range(500)]  # Index untuk data

#         df = pd.read_csv(f"{dataset_path}/ExtractResult/harralick/features_d{d}_theta{theta}.csv")
#         df['image'] = img
        
#         feature_dataframes[(d, theta)] = df
#         print(f"Data untuk d={d}, theta={theta} berhasil dihitung dan disimpan ke dalam memori!")


st.set_page_config(layout="wide")
for d in kombinasiFeature[0]:
    for theta in kombinasiFeature[1]:
        try:
            # Load the dataset for the given (d, theta) combination
            df = pd.read_csv(f"{dataset_path}/ExtractResult/harralick/features_d{d}_theta{theta}.csv")
            df['image'] = [f"img-{i}" for i in range(len(df))]  # Add image IDs
            feature_dataframes[(d, theta)] = df
        except FileNotFoundError:
            st.warning(f"Dataset for d={d}, theta={theta} not found. Skipping...")

# Function to calculate accuracy for different k values
def calculate_knn_results(df, k_values):
    numeric_features = df.drop(columns=['label', 'image'], axis=1).columns
    scaler = MinMaxScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    X = df.drop(columns=['label','homogeneity','sum_variance', 'ASM','IMC2', 'image'], axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[k] = {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    return results

# Streamlit Dashboard Title
st.title("Apple Ripeness Classification Dashboard")

# Tabs for Summary and Detailed Results
tab1, tab2 = st.tabs(["Summary Report", "Detailed Results"])

# Summary Report
with tab1:
    st.header("Summary Report")

    # Calculate the best k for each (d, theta) combination
    k_values = range(2, 26)
    summary_data = []
    all_results = {}

    for (d, theta), df in feature_dataframes.items():
        results = calculate_knn_results(df, k_values)
        all_results[(d, theta)] = results

        # Find the best k for this combination
        best_k = max(results, key=lambda k: results[k]['accuracy'])
        best_accuracy = results[best_k]['accuracy']
        summary_data.append({'d': d, 'theta': theta, 'best_k': best_k, 'accuracy': best_accuracy})

    # Display summary table
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df)

    # Highlight the best combination overall
    best_combination = summary_df.loc[summary_df['accuracy'].idxmax()]
    st.write(f"**Best Combination:** d={best_combination['d']}, theta={best_combination['theta']}, k={best_combination['best_k']}, Accuracy={best_combination['accuracy']:.2f}")

# Detailed Results
with tab2:
    st.header("Detailed Results")

    # Select a combination (d, theta)
    d_theta = st.selectbox("Select (d, theta) combination", list(all_results.keys()))
    results = all_results[d_theta]

    # Plot accuracy vs. k
    accuracies = {k: result['accuracy'] for k, result in results.items()}
    fig, ax = plt.subplots()
    ax.plot(list(accuracies.keys()), list(accuracies.values()), marker='o')
    ax.set_xlabel("k (Number of Neighbors)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs. k for d={d_theta[0]}, theta={d_theta[1]}")
    st.pyplot(fig)

    # Show confusion matrix for the best k
    best_k = max(results, key=lambda k: results[k]['accuracy'])
    st.write(f"**Best k:** {best_k}, Accuracy: {results[best_k]['accuracy']:.2f}")

    cm = results[best_k]['confusion_matrix']
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    st.pyplot(fig)
# # Streamlit Dashboard
# st.title("Apple Classification Dashboard using KNN")

# tab1, tab2 = st.tabs(["EDA", "KNN Classification"])

# # EDA Section
# with tab1:
#     st.header("Exploratory Data Analysis (EDA)")
    
#     # Select a combination (d, theta) for EDA
#     d_theta = st.selectbox("Select (d, theta) combination for EDA", list(feature_dataframes.keys()), help="Select the combination of distance (d) and angle (theta) for GLCM features.")