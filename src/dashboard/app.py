import sys
import os
from pathlib import Path

# DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__)) # Get the directory of the current file
# SRC_DIR = os.path.dirname(DASHBOARD_DIR) # Get the parent directory (src)

# # Get and add the root directory of the project to sys.path
# BASE_DIR = os.path.dirname(SRC_DIR)
# sys.path.append(BASE_DIR)
# import sys


import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from src.features_extraction.harralick import load_harralick_features
from src.preprocessing.preprocessing import preprocess_data
from src.classifier.knn import calculate_knn_results
from src.utils.metrics import plot_confusion_matrix

# TEST_DIR = Path(__file__).resolve().parent.parent.parent
# st.write(f"TEST_DIR: {TEST_DIR}") # Debugging line to check the SRC_DIR

# Adjust the matplotlib style for Streamlit
plt.rcParams.update({
    'axes.facecolor': '#00172B', # background color for plot area
    'axes.edgecolor': '#FFF',    # Color for axes border
    'axes.labelcolor': '#FFF',   # Color for axes labels
    'xtick.color': '#FFF',       # Color for x-axis ticks
    'ytick.color': '#FFF',       # Color for y-axis ticks
    'text.color': '#FFF',        # Color for text
    'figure.facecolor': '#00172B',  # Background color for figure
    'legend.facecolor': '#0083B8',  # Background color for legend
    'legend.edgecolor': '#FFF',     # color for legend border
})

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Report Dashboard", page_icon=":bar_chart:")

# Tambahkan log untuk debugging path
# st.write("Debugging Path:")
# st.write(f"BASE_DIR: {BASE_DIR}") # Debugging line to check the BASE_DIR
# st.write(f"DASHBOARD_DIR: {DASHBOARD_DIR}")
# st.write(f"SRC_DIR: {SRC_DIR}")
# st.write(f"BASE_DIR: {BASE_DIR}")

# Update BASE_DIR to use Pathlib for dynamic path resolution
BASE_DIR = Path(__file__).resolve().parent.parent.parent / 'dataset'
if not BASE_DIR.exists():
    st.error(f"Dataset directory not found: {BASE_DIR}")
else:
    st.write(f"Dataset directory found: {BASE_DIR}")

## Load Dataset
kombinasiFeature = [[1, 2, 3], [0, 45, 90, 135]]
feature_dataframes = load_harralick_features(BASE_DIR, kombinasiFeature)

st.title("Apple Ripeness Classification Dashboard")
st.write(f"""This dashboard displays the analysis results of the author's thesis. 
         The Summary Report tab contains the conclusion of the best KNN algorithm value on the combination of dataset hyperparameters. 
         The Detailed Results tab contains several visualizations of the results of this study.\n
        NOTE:
        Label 0 = 20% apple ripeness level
        Label 1 = 40% apple ripeness level
        Label 2 = 60% apple ripeness level
        Label 3 = 80% apple ripeness level
        Label 4 = 100% apple ripeness level""")

# Tabs for Summary and Detailed Results
tab1, tab2 = st.tabs(["Summary Report", "Detailed Results"])

### ----- Summary Report -----
with tab1:
    st.header("Summary Report")

    k_values = range(3, 26, 2) # Odd values for k
    summary_data = []
    all_results = {}

    for (d, theta), df in feature_dataframes.items():
        X_train, X_test, y_train, y_test = preprocess_data(df, drop_columns=['label','homogeneity','sum_variance', 'ASM','IMC2', 'image'])
        results = calculate_knn_results(X_train, X_test, y_train, y_test, k_values)
        all_results[(d, theta)] = results

        best_k = max(results, key=lambda k: results[k]['accuracy'])
        best_accuracy = results[best_k]['accuracy']
        summary_data.append({'d': d, 'theta': theta, 'best_k': best_k, 'accuracy': best_accuracy})

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df)
    best_combination = summary_df.loc[summary_df['accuracy'].idxmax()]
    st.write(f"**Best Combination:** d={best_combination['d']}, theta={best_combination['theta']}, k={best_combination['best_k']}, Accuracy={best_combination['accuracy']:.2f}")
### ----- Summary Report End -----

### ----- Detailed Results -----
with tab2:
    st.header("Detailed Results")
    st.write(f"""Base on the best combination from the summary report, as we can see the best combination is\n  
            d={best_combination['d']}, theta={best_combination['theta']}, k={best_combination['best_k']}, Accuracy={best_combination['accuracy']:.2f}, 
            \nbut you can select the other (d, theta) combination to see the detailed results.""")
    
    # Select a combination (d, theta)
    d_theta = st.selectbox("Select (d, theta) combination", list(all_results.keys()))
    results = all_results[d_theta]
    col1, col2= st.columns([7, 3])
    ## ----- RAW 1 VISUALIZATION -----
    # ----- Plot accuracy vs. k -----
    with col1:
        # Plot accuracy vs. k
        st.subheader("Accuracy vs. k")
        accuracies = {k: result['accuracy'] for k, result in results.items()}
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(list(accuracies.keys()), list(accuracies.values()), marker='o')
        ax.set_xlabel("k (Number of Neighbors)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs. k for d={d_theta[0]}, theta={d_theta[1]}")
        st.pyplot(fig)
        best_k = max(results, key=lambda k: results[k]['accuracy'])
        st.write(f"**Best k:** {best_k}, Accuracy: {results[best_k]['accuracy']:.2f}")
    # ----- Plot accuracy vs. k End -----

    # ----- Confusion Matrix -----
    with col2:
        # Show confusion matrix for the best k
        st.subheader("Confusion Matrix")
        cm = results[best_k]['confusion_matrix']
        fig, ax = plt.subplots(figsize=(6, 4)) 
        ax.set_facecolor('#00172B')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title(f"Confusion Matrix for k={best_k}", color='#FFF')
        ax.set_xlabel("Predicted Label", color='#FFF')
        ax.set_ylabel("True Label", color='#FFF')
        ax.tick_params(colors='#FFF')  # Warna ticks
        st.pyplot(fig)
    # ----- Confusion Matrix End -----
    ## ----- RAW 1 VISUALIZATION END -----


    ## ----- RAW 2 VISUALIZATION -----
    # Precision, Recall, and F1-Score Visualization
    st.subheader("Precision, Recall, and F1-Score")
    raw2col1, raw2col2 = st.columns(2)
    # ----- classification report -----
    with raw2col1:  
        # Get classification report for the best k
        classification_report = results[best_k]['classification_report']

        # Convert classification report to a DataFrame for easier visualization
        report_df = pd.DataFrame(classification_report).transpose()
        # Display the classification report as a table
        st.write("**Classification Report**")
        st.dataframe(report_df)
    # ----- classification report end -----

    # ----- Plot precision, recall, and F1-score as a bar chart -----
    with raw2col2:
        st.write("**Bar Chart of Precision, Recall, and F1-Score**")
        metrics = ['precision', 'recall', 'f1-score']
        classes = [label for label in classification_report.keys() if isinstance(label, str) and label.isdigit()]

        # Prepare data for the bar chart
        metric_values = {metric: [classification_report[cls][metric] for cls in classes] for metric in metrics}
        df_metrics = pd.DataFrame(metric_values, index=classes)

        # Plot the bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        df_metrics.plot(kind='bar', ax=ax, color=['#E694FF', '#0083B8', '#FFF']) 
        ax.set_facecolor('#00172B')  # Background color
        ax.set_title(f"Precision, Recall, and F1-Score for k={best_k}", color='#FFF')
        ax.set_xlabel("Classes", color='#FFF')
        ax.set_ylabel("Score", color='#FFF')
        ax.legend(title="Metrics", loc='lower center', facecolor='#0083B8', edgecolor='#FFF', title_fontsize=10, fontsize=9)
        st.pyplot(fig)
    # ----- Plot precision, recall, and F1-score as a bar chart end -----
    ## ----- RAW 2 VISUALIZATION END -----


    ## ----- RAW 3 VISUALIZATION -----
    # ----- Nearest Neighbors Visualization -----
    st.subheader("Nearest Neighbors Visualization")

    # Use a form to avoid refreshing the entire page
    with st.form("test_sample_form"):
        test_sample_idx = st.slider("Select Test Sample Index", 0, len(results[best_k]['y_test']) - 1, 0)
        submit_button = st.form_submit_button("Update Visualization")

    if submit_button:
        # Perform PCA for visualization

        X_train, X_test, y_train, y_test = preprocess_data(feature_dataframes[d_theta], drop_columns=['label', 'homogeneity', 'sum_variance', 'ASM', 'IMC2', 'image'])
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train)
        X_test_2d = pca.transform(X_test)

        # Get neighbors for the selected test sample
        knn = KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(X_train, y_train)
        distances, indices = knn.kneighbors(X_test)
        neighbors_idx = indices[test_sample_idx]

        # Plot the nearest neighbors
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_facecolor('#00172B')
        # Define label colors
        label_colors = {0: 'yellow', 1: 'orange', 2: 'green', 3: 'blue', 4: 'red'}
        label_names = {0: '20% Ripeness', 1: '40% Ripeness', 2: '60% Ripeness', 3: '80% Ripeness', 4: '100% Ripeness'}
        default_color = 'gray'

        # Plot training data with color coding based on labels
        y_train_array = np.array(y_train)
        for label in np.unique(y_train_array):
            ax.scatter(
                X_train_2d[y_train_array == label, 0],
                X_train_2d[y_train_array == label, 1],
                c=label_colors.get(label, '#FFF'), 
                label=f"Training Data ({label_names.get(label, 'Unknown')})",
                alpha=0.6
            )

        # Highlight test sample
        ax.scatter(
            X_test_2d[test_sample_idx, 0],
            X_test_2d[test_sample_idx, 1],
            c='#E694FF',  # Color for test sample
            label='Test Sample',
            edgecolor='#FFF',
            s=150
        )

        # Highlight the nearest neighbors
        for neighbor_idx in neighbors_idx:
            ax.scatter(
                X_train_2d[neighbor_idx, 0],
                X_train_2d[neighbor_idx, 1],
                c=label_colors.get(y_train_array[neighbor_idx], default_color),
                edgecolor='black',
                s=100,
                label=f"Neighbor ({label_names.get(y_train_array[neighbor_idx], 'Unknown')})"
            )

        # Add legend and labels
        ax.legend(loc='best', bbox_to_anchor=(1.05, 1), title="Legend", facecolor='#0083B8', edgecolor='#FFF')
        ax.set_title(f"Visualization of {best_k} Nearest Neighbors for Test Sample {test_sample_idx + 1}", color='#FFF')
        ax.set_xlabel('PCA Component 1', color='#FFF')
        ax.set_ylabel('PCA Component 2', color='#FFF')
        st.pyplot(fig)
        # ----- Nearest Neighbors Visualization End -----
        ## ----- RAW 3 VISUALIZATION END -----
        ### ----- Detail Results END -----


## ----- HIDE STREAMLIT STYLE -----
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
## ----- HIDE STREAMLIT STYLE END -----