import sys
import os
from pathlib import Path

## ----- Path Configuration -----
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__)) # Get the directory of the current file
SRC_DIR = os.path.dirname(DASHBOARD_DIR) # Get the parent directory (src)

# Get and add the root directory of the project to sys.path
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.append(BASE_DIR)
## ----- Path Configuration END -----

## ----- Imports -----
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import plotly.express as px
import plotly.graph_objects as go
from src.preprocessing.preprocessing import preprocess_data
from src.classifier.knn import calculate_knn_results, calculate_kfold_cv_score
from src.utils.metrics import plot_confusion_matrix
## ----- Imports END -----

## ----- Matplotlib Style Configuration -----
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
## ----- Matplotlib Style Configuration END -----

## ----- Streamlit Configuration -----
# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Report Dashboard", page_icon=":bar_chart:")

## ----- Load Haralick features for all (d, theta) combinations -----
def load_harralick_features(dataset_path, kombinasiFeature):
    """
    Load Haralick features for all (d, theta) combinations.
    """
    feature_dataframes = {}
    for d in kombinasiFeature[0]:
        for theta in kombinasiFeature[1]:
            try:
                df = pd.read_csv(dataset_path/'ExtractResult'/'harralick'/f'features_d{d}_theta{theta}.csv')
                df['image'] = [f"img-{i}" for i in range(len(df))]
                feature_dataframes[(d, theta)] = df
            except FileNotFoundError:
                print(f"Dataset for d={d}, theta={theta} not found. Skipping...")
    return feature_dataframes
## ----- Load Haralick features for all (d, theta) combinations END-----

## ----- Path and Data Loading -----
# Update BASE_DIR to use Pathlib for dynamic path resolution
BASE_DIR = Path(__file__).resolve().parent.parent.parent / 'dataset'

@st.cache_data
def load_feature_data():
    kombinasiFeature = [[1, 2, 3], [0, 45, 90, 135]]
    return load_harralick_features(BASE_DIR, kombinasiFeature)
## ----- Path and Data Loading END -----

## ----- Data Processing Functions -----
@st.cache_data
def preprocess_and_calculate_results(feature_dataframes, k_values):
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

@st.cache_data
def calculate_classification_report(y_test, y_pred):
    return classification_report(y_test, y_pred, output_dict=True)

@st.cache_data
def perform_pca(X_train, X_test, n_components=2):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

## ----- Cross Validation Functions -----
@st.cache_data
def calculate_cross_validation_results(feature_dataframes, summary_df, n_splits=5):
    """
    Perform K-fold cross validation on the best k value for each (d, theta) combination
    """
    cv_results = {}
    
    for _, row in summary_df.iterrows():
        d = row['d']
        theta = row['theta']
        best_k = int(row['best_k'])  # Convert best_k to int to fix the error
        
        # Get data for this combination
        df = feature_dataframes[(d, theta)]
        
        # Preprocess all data (don't split into train/test yet as CV will do that)
        X = df.drop(columns=['label', 'homogeneity', 'sum_variance', 'ASM', 'IMC2', 'image'])
        y = df['label']
        
        # Perform cross-validation
        cv_result = calculate_kfold_cv_score(X, y, best_k, n_splits=n_splits)
        cv_results[(d, theta, best_k)] = cv_result
    
    return cv_results
## ----- Cross Validation Functions END -----

## ----- Main Application -----
# Load Dataset
feature_dataframes = load_feature_data()

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
tab1, tab2, tab3 = st.tabs(["Summary Report", "Detailed Results", "Cross Validation"])

## ----- Summary Report Tab -----
with tab1:
    st.header("Summary Report")
    tab1col1, tab1col2 = st.columns([4,6])
    with tab1col1:

        k_values = range(3, 26, 2) # Odd values for k
        summary_df, all_results = preprocess_and_calculate_results(feature_dataframes, k_values)

        st.dataframe(summary_df)
        best_combination = summary_df.loc[summary_df['accuracy'].idxmax()]
        st.write(f"**Best Combination:** d={best_combination['d']}, theta={best_combination['theta']}, k={best_combination['best_k']}, Accuracy={best_combination['accuracy']:.2f}")
    with tab1col2:
        # Interactive Bar Chart
        fig = px.bar(summary_df, x='d', y='accuracy', color='theta', title='Accuracy by d and theta', text='accuracy')
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_title='d', yaxis_title='Accuracy', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
## ----- Summary Report Tab END -----

## ----- Detailed Results Tab -----
with tab2:
    st.header("Detailed Results")
    st.write(f"""Base on the best combination from the summary report, as we can see the best combination is""")
    col_d,col_theta,col_k,col_accuracy = st.columns([1,1,1,1]) 
    with col_d:
        st.write(f"```\nd: \n{best_combination['d']}", help="Best d value")
    with col_theta:
        st.write(f"```\ntheta: \n{best_combination['theta']}", help="Best theta value")
    with col_k:
        st.write(f"```\nk: \n{best_combination['best_k']}", help="Best k value")
    with col_accuracy:
        st.write(f"```\nAccuracy: \n{best_combination['accuracy']:.2f}", help="Best accuracy value") 
 
    st.write(f"""but you can select the other (d, theta) and K combination to see the detailed results.""")

    ## ----- Parameter Selection -----
    # Select a combination (d, theta)
    d_theta = st.selectbox("Select (d, theta) combination", list(all_results.keys()), help="Choose a combination of d and theta to visualize the results")
    results = all_results[d_theta]

    # Interactive Line Chart for Accuracy vs. k
    accuracies = {k: result['accuracy'] for k, result in results.items()}
    line_fig = px.line(x=list(accuracies.keys()), y=list(accuracies.values()), markers=True, title=f"Accuracy vs. k for d={d_theta[0]}, theta={d_theta[1]}")
    line_fig.update_layout(xaxis_title='k (Number of Neighbors)', yaxis_title='Accuracy', template='plotly_white')
    st.plotly_chart(line_fig, use_container_width=True)

    # Show the best k from algorithm and allow user to select a custom k
    best_k = max(results, key=lambda k: results[k]['accuracy'])
    st.write(f"**Best k:** {best_k}, Accuracy: {results[best_k]['accuracy']:.2f}")
    # Let user select a custom k value
    available_k_values = list(results.keys())
    selected_k = st.selectbox(
        "Select k value for visualization", 
        available_k_values,
        index=available_k_values.index(best_k),  # Default to best_k
        help="Choose a k value to use for the KNN visualization below"
    )
    st.write(f"Selected k: {selected_k}, Accuracy: {results[selected_k]['accuracy']:.2f}")
    ## ----- Parameter Selection END -----
    
    ## ----- Results Metrics Display -----
    # Get test and prediction data for selected_k
    y_test = results[selected_k]['y_test']
    y_pred = results[selected_k]['y_pred']
    
    # Create 3 columns for visualizations
    col_cm, col_report, col_bar = st.columns(3)
    
    # Column 1: Confusion Matrix
    with col_cm:
        st.subheader("Confusion Matrix")
        
        # Calculate confusion matrix
        conf_matrix = np.zeros((5, 5), dtype=int)  # 5 classes (0-4)
        for t, p in zip(y_test, y_pred):
            conf_matrix[t, p] += 1
        
        # Create heatmap using Plotly
        labels = ['20%', '40%', '60%', '80%', '100%']
        conf_fig = px.imshow(
            conf_matrix,
            x=labels,
            y=labels,
            color_continuous_scale='Blues',
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            title=f"Confusion Matrix for k={selected_k}",
            text_auto=True
        )
        
        # Update layout for better readability
        conf_fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='#00172B',
            paper_bgcolor='#00172B',
            height=400,
            xaxis=dict(
                title_font=dict(size=12),
                tickfont=dict(size=10),
                tickvals=[0, 1, 2, 3, 4],
                ticktext=labels
            ),
            yaxis=dict(
                title_font=dict(size=12),
                tickfont=dict(size=10),
                tickvals=[0, 1, 2, 3, 4],
                ticktext=labels
            ),
            coloraxis_showscale=True,
            margin=dict(l=50, r=50, t=80, b=50),
            title_font=dict(size=14)
        )
        
        st.plotly_chart(conf_fig, use_container_width=True)
    
    # Column 2: Classification Report
    with col_report:
        st.subheader("Classification Report")
        
        # Add some space before the classification report to align it better
        st.write("")
        st.write("")
        st.write("")
        
        # Classification Report
        class_report = pd.DataFrame(calculate_classification_report(y_test, y_pred)).transpose()
        st.dataframe(class_report)
    
    # Column 3: Bar Chart for Precision, Recall, F1-Score
    with col_bar:
        st.subheader("Bar Chart")
        st.write("")
        metrics_df = pd.DataFrame(class_report)
        metrics_df = metrics_df[['precision', 'recall', 'f1-score']].iloc[:-3]  # Exclude support and averages
        bar_fig = px.bar(metrics_df, barmode='group',title=f"Prec, Rec, F1-Score by Class",)
        bar_fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis_title='Class',
            yaxis_title='Score',
            legend_title='Variable',
            margin=dict(l=50, r=50, t=30, b=50)
        )
        st.plotly_chart(bar_fig, use_container_width=True)
    ## ----- Results Metrics Display END -----
    
    ## ----- Nearest Neighbor Visualization -----
    st.subheader("Nearest Neighbor Visualization")
    
    # Use a form to avoid refreshing the entire page
    with st.form("test_sample_form"):
        test_sample_idx = st.slider("Select Test Sample Index", 0, 99, 27)
        submit_button = st.form_submit_button("Update Visualization")

    # Always show visualization, update on submit
    # Perform PCA for visualization
    X_train, X_test, y_train, y_test = preprocess_data(feature_dataframes[d_theta], drop_columns=['label', 'homogeneity', 'sum_variance', 'ASM', 'IMC2', 'image'])
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    # Get neighbors for the selected test sample
    knn = KNeighborsClassifier(n_neighbors=selected_k)
    knn.fit(X_train, y_train)
    distances, indices = knn.kneighbors(X_test)
    neighbors_idx = indices[test_sample_idx]

    ## ----- PCA Visualization Data Preparation -----
    # Define label colors and names
    label_colors = {0: 'yellow', 1: 'orange', 2: 'green', 3: 'blue', 4: 'red'}
    label_names = {0: '20% Ripeness', 1: '40% Ripeness', 2: '60% Ripeness', 3: '80% Ripeness', 4: '100% Ripeness'}
    
    # Create a DataFrame for plotly visualization
    plot_data = []
    
    # Add training data
    y_train_array = np.array(y_train)
    for i, (x, y, label) in enumerate(zip(X_train_2d[:, 0], X_train_2d[:, 1], y_train_array)):
        is_neighbor = i in neighbors_idx
        neighbor_text = f"Neighbor ({label_names.get(label, 'Unknown')})" if is_neighbor else None
        
        if is_neighbor:
            marker_size = 15
            border_width = 2
            legend_group = f"Neighbor ({label_names.get(label, 'Unknown')})"
        else:
            marker_size = 8
            border_width = 0
            legend_group = f"Training Data ({label_names.get(label, 'Unknown')})"
            
        plot_data.append({
            'PCA Component 1': x, 
            'PCA Component 2': y,
            'Label': str(label),
            'Category': 'Training Data',
            'Ripeness': label_names.get(label, 'Unknown'),
            'Size': marker_size,
            'BorderWidth': border_width,
            'LegendGroup': legend_group,
            'IsNeighbor': is_neighbor,
            'NeighborText': neighbor_text
        })
    
    # Add test sample
    plot_data.append({
        'PCA Component 1': X_test_2d[test_sample_idx, 0],
        'PCA Component 2': X_test_2d[test_sample_idx, 1],
        'Label': str(y_test.iloc[test_sample_idx]),
        'Category': 'Test Sample',
        'Ripeness': label_names.get(y_test.iloc[test_sample_idx], 'Unknown'),
        'Size': 20,
        'BorderWidth': 2,
        'LegendGroup': 'Test Sample',
        'IsNeighbor': False,
        'NeighborText': None
    })
    
    plot_df = pd.DataFrame(plot_data)
    ## ----- PCA Visualization Data Preparation END -----
    
    ## ----- Plotly Visualization -----
    # Create Plotly figure
    fig = px.scatter(
        plot_df, 
        x='PCA Component 1', 
        y='PCA Component 2',
        color='Ripeness',
        symbol='Category',
        size='Size',
        title=f"Visualization of {selected_k} Nearest Neighbors for Test Sample {test_sample_idx}",
        color_discrete_map={
            '20% Ripeness': 'yellow',
            '40% Ripeness': 'orange',
            '60% Ripeness': 'green',
            '80% Ripeness': 'blue',
            '100% Ripeness': 'red',
        },
        symbol_map={
            'Training Data': 'circle',
            'Test Sample': 'diamond',
        },
        hover_data={
            'Label': True,
            'Category': True,
            'Ripeness': True,
            'IsNeighbor': False,
            'Size': False,
            'BorderWidth': False,
            'LegendGroup': False,
            'NeighborText': True
        }
    )
    
    # Customize the figure
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#00172B',
        paper_bgcolor='#00172B',
        legend_title_text='Legend',
        height=600,
        title={
            'text': f"Visualization of {selected_k} Nearest Neighbors for Test Sample {test_sample_idx}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    # Customize markers for each point individually (but don't add to legend)
    for i, row in plot_df.iterrows():
        fig.add_scatter(
            x=[row['PCA Component 1']],
            y=[row['PCA Component 2']],
            mode='markers',
            marker=dict(
                color=label_colors.get(int(row['Label']), 'gray'),
                size=row['Size'],
                line=dict(width=row['BorderWidth'], color='black'),
                symbol='diamond' if row['Category'] == 'Test Sample' else 'circle'
            ),
            legendgroup=row['LegendGroup'],
            name=row['LegendGroup'],
            showlegend=False,  # Don't show in legend to avoid duplicates
            hoverinfo='text',
            hovertext=f"{row['Category']}<br>Label: {row['Label']}<br>Ripeness: {row['Ripeness']}<br>{row['NeighborText'] if row['NeighborText'] else ''}"
        )
    
    ## ----- Legend Customization -----
    # Count occurrences of each label in neighbors
    neighbor_label_counts = {}
    for idx in neighbors_idx:
        label = y_train_array[idx]
        if label in neighbor_label_counts:
            neighbor_label_counts[label] += 1
        else:
            neighbor_label_counts[label] = 1
            
    # Add only the labels with neighbors to legend (in descending order of occurrence)
    for label, count in sorted(neighbor_label_counts.items(), key=lambda x: x[1], reverse=True):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    color=label_colors.get(label, 'gray'),
                    size=15,
                    line=dict(width=2, color='black'),
                    symbol='circle'
                ),
                name=f"Neighbor ({label_names.get(label, 'Unknown')}) - {count}/{selected_k}",
                legendgroup=f"Neighbor ({label_names.get(label, 'Unknown')})",
                showlegend=True
            )
        )
    
    # Add test sample to legend
    test_label = y_test.iloc[test_sample_idx]
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                color=label_colors.get(test_label, 'gray'),
                size=20,
                line=dict(width=2, color='black'),
                symbol='diamond'
            ),
            name=f"Test Sample ({label_names.get(test_label, 'Unknown')})",
            legendgroup='Test Sample',
            showlegend=True
        )
    )
    ## ----- Legend Customization END -----
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display test sample prediction info
    true_label = y_test.iloc[test_sample_idx]
    pred_label = y_pred[test_sample_idx]
    st.write(f"**Test Sample #{test_sample_idx}:**")
    st.write(f"- True Label: {true_label} ({label_names.get(true_label, 'Unknown')})")
    st.write(f"- Predicted Label: {pred_label} ({label_names.get(pred_label, 'Unknown')})")
    st.write(f"- Prediction {'Correct' if true_label == pred_label else 'Incorrect'}")
    ## ----- Nearest Neighbor Visualization END -----
## ----- Detailed Results Tab END -----

## ----- Cross Validation Tab -----
with tab3:
    st.header("K-Fold Cross Validation Results")
    
    # Define number of folds for cross-validation
    n_splits = st.slider("Number of folds for cross-validation", min_value=3, max_value=10, value=5)
    
    # Calculate cross-validation results for the best k value of each (d, theta) combination
    cv_results = calculate_cross_validation_results(feature_dataframes, summary_df, n_splits)
    
    # Create a summary dataframe for CV results
    cv_summary_data = []
    for (d, theta, k), result in cv_results.items():
        cv_summary_data.append({
            'd': d,
            'theta': theta,
            'k': k,
            'mean_cv_accuracy': result['mean_cv_score'],
            'std_cv_accuracy': result['std_cv_score'],
        })
    
    cv_summary_df = pd.DataFrame(cv_summary_data)
    
    # Display the CV summary
    st.subheader("Cross Validation Summary")
    st.write(f"Results of {n_splits}-fold cross validation for the best k value of each (d, theta) combination:")
    st.dataframe(cv_summary_df.style.format({
        'mean_cv_accuracy': "{:.4f}",
        'std_cv_accuracy': "{:.4f}"
    }))
    
    # Find the best combination based on CV
    best_cv_combo = cv_summary_df.loc[cv_summary_df['mean_cv_accuracy'].idxmax()]
    st.write(f"""
    **Best Combination (Cross Validation):**
    - d = {best_cv_combo['d']}
    - theta = {best_cv_combo['theta']}
    - k = {best_cv_combo['k']}
    - Mean CV Accuracy = {best_cv_combo['mean_cv_accuracy']:.4f}
    - Standard Deviation = {best_cv_combo['std_cv_accuracy']:.4f}
    """)
    
    # Visualizations
    # 1. Bar chart of mean CV accuracy by (d, theta) combination
    cv_bar_fig = px.bar(
        cv_summary_df, 
        x=['d', 'theta'],
        y='mean_cv_accuracy',
        error_y='std_cv_accuracy',
        color='d',
        barmode='group',
        title="Cross Validation Accuracy by (d, theta) Combination",
        labels={'mean_cv_accuracy': 'Mean CV Accuracy', 'std_cv_accuracy': 'Standard Deviation'}
    )
    cv_bar_fig.update_layout(template='plotly_white')
    st.plotly_chart(cv_bar_fig, use_container_width=True)
    
    # 2. Detailed view of selected combination
    st.subheader("Detailed Cross Validation Results")
    
    # Select a (d, theta, k) combination to view detailed results
    cv_combo_options = [(d, theta, k) for d, theta, k in cv_results.keys()]
    selected_combo_str = st.selectbox(
        "Select a combination to view detailed fold results:",
        options=[f"d={d}, theta={theta}, k={k}" for d, theta, k in cv_combo_options],
        index=cv_combo_options.index((best_cv_combo['d'], best_cv_combo['theta'], best_cv_combo['k']))
    )
    
    # Parse the selected combo string back to tuple
    parts = selected_combo_str.split(', ')
    selected_d = int(float(parts[0].split('=')[1]))
    selected_theta = int(float(parts[1].split('=')[1]))
    selected_k = int(float(parts[2].split('=')[1]))
    selected_combo = (selected_d, selected_theta, selected_k)
    
    # Get the detailed results for the selected combination
    selected_cv_result = cv_results[selected_combo]
    
    # Display fold accuracies
    fold_accuracies = [fold_result['accuracy'] for fold_result in selected_cv_result['fold_results']]
    fold_df = pd.DataFrame({
        'Fold': [f"Fold {i+1}" for i in range(len(fold_accuracies))],
        'Accuracy': fold_accuracies
    })
    
    # Bar chart of fold accuracies
    fold_fig = px.bar(
        fold_df,
        x='Fold',
        y='Accuracy',
        title=f"Accuracy by Fold for d={selected_combo[0]}, theta={selected_combo[1]}, k={selected_combo[2]}",
        color='Accuracy',
        color_continuous_scale='Viridis'
    )
    fold_fig.add_hline(
        y=selected_cv_result['mean_cv_score'], 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Mean: {selected_cv_result['mean_cv_score']:.4f}"
    )
    fold_fig.update_layout(template='plotly_white', yaxis_range=[0, 1])
    st.plotly_chart(fold_fig, use_container_width=True)
    
    ## ----- Detailed K-fold Metrics -----
    st.subheader("Detailed Metrics for Each Fold")
    
    # Add metrics tabs for detailed analysis
    metric_tabs = st.tabs(["Confusion Matrices", "Class Metrics", "Fold Comparison"])
    
    # Tab 1: Confusion Matrices for each fold
    with metric_tabs[0]:
        # Create grid of confusion matrices
        n_cols = min(3, n_splits)  # Max 3 columns
        n_rows = (n_splits + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure for all confusion matrices
        labels = ['20%', '40%', '60%', '80%', '100%']
        for row in range(n_rows):
            cols = st.columns(n_cols)
            for col_idx in range(n_cols):
                fold_idx = row * n_cols + col_idx
                if fold_idx < n_splits:
                    with cols[col_idx]:
                        fold_result = selected_cv_result['fold_results'][fold_idx]
                        st.write(f"**Fold {fold_idx+1}** (Acc: {fold_result['accuracy']:.4f})")
                        
                        # Create heatmap for this fold's confusion matrix
                        fold_cm_fig = px.imshow(
                            fold_result['confusion_matrix'],
                            x=labels,
                            y=labels,
                            color_continuous_scale='Blues',
                            labels=dict(x="Predicted", y="True", color="Count"),
                            text_auto=True
                        )
                        fold_cm_fig.update_layout(
                            height=300,
                            width=300,
                            margin=dict(l=40, r=40, t=40, b=40)
                        )
                        st.plotly_chart(fold_cm_fig)
    
    # Tab 2: Class-specific metrics for each fold
    with metric_tabs[1]:
        # Extract class metrics from each fold
        class_metrics_data = []
        for fold_idx, fold_result in enumerate(selected_cv_result['fold_results']):
            for class_label, metrics in fold_result['classification_report'].items():
                if class_label.isdigit():  # Only include actual classes, not averages
                    class_metrics_data.append({
                        'Fold': fold_idx + 1,
                        'Class': int(class_label),
                        'Class Name': label_names.get(int(class_label), f"Class {class_label}"),
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1-Score': metrics['f1-score'],
                        'Support': metrics['support']
                    })
        
        class_metrics_df = pd.DataFrame(class_metrics_data)
        
        # Interactive selection for class
        selected_metric = st.selectbox(
            "Select metric to visualize:", 
            ["Precision", "Recall", "F1-Score"], 
            index=2  # Default to F1-Score
        )
        
        # Create grouped bar chart for the selected metric across folds and classes
        class_metric_fig = px.bar(
            class_metrics_df,
            x="Class Name",
            y=selected_metric,
            color="Fold",
            barmode="group",
            title=f"{selected_metric} by Class and Fold",
            labels={selected_metric: selected_metric, "Class Name": "Ripeness Level"},
            color_continuous_scale="Viridis"
        )
        class_metric_fig.update_layout(template='plotly_white', yaxis_range=[0, 1])
        st.plotly_chart(class_metric_fig, use_container_width=True)
        
        # Show detailed metrics table with expandable sections by class
        st.subheader("Detailed Class Metrics by Fold")
        
        # Group by class for expandable sections
        for class_label in sorted(class_metrics_df['Class'].unique()):
            class_name = label_names.get(class_label, f"Class {class_label}")
            with st.expander(f"{class_name} (Class {class_label}) Metrics"):
                class_data = class_metrics_df[class_metrics_df['Class'] == class_label]
                st.dataframe(
                    class_data[['Fold', 'Precision', 'Recall', 'F1-Score', 'Support']]
                    .set_index('Fold')
                    .style.format({
                        'Precision': "{:.4f}",
                        'Recall': "{:.4f}",
                        'F1-Score': "{:.4f}",
                        'Support': "{:.0f}"
                    })
                )
                
                # Add radar chart for this class across folds
                radar_fig = go.Figure()
                
                for fold_idx, fold_data in class_data.groupby('Fold'):
                    radar_fig.add_trace(go.Scatterpolar(
                        r=[fold_data['Precision'].values[0], fold_data['Recall'].values[0], fold_data['F1-Score'].values[0]],
                        theta=['Precision', 'Recall', 'F1-Score'],
                        fill='toself',
                        name=f'Fold {fold_idx}'
                    ))
                
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=True,
                    title=f"Metric Comparison for {class_name}",
                    height=400
                )
                st.plotly_chart(radar_fig)
    
    # Tab 3: Fold Comparison - Compare metrics across folds
    with metric_tabs[2]:
        st.write("### Compare Metrics Across Folds")
        
        # Calculate aggregate metrics for each fold
        fold_aggregate_metrics = []
        for fold_idx, fold_result in enumerate(selected_cv_result['fold_results']):
            # Get the "macro avg" or "weighted avg" from classification report
            macro_metrics = fold_result['classification_report'].get('macro avg', {})
            weighted_metrics = fold_result['classification_report'].get('weighted avg', {})
            
            fold_aggregate_metrics.append({
                'Fold': fold_idx + 1,
                'Accuracy': fold_result['accuracy'],
                'Macro Precision': macro_metrics.get('precision', 0),
                'Macro Recall': macro_metrics.get('recall', 0),
                'Macro F1-Score': macro_metrics.get('f1-score', 0),
                'Weighted Precision': weighted_metrics.get('precision', 0),
                'Weighted Recall': weighted_metrics.get('recall', 0),
                'Weighted F1-Score': weighted_metrics.get('f1-score', 0),
            })
        
        fold_metrics_df = pd.DataFrame(fold_aggregate_metrics)
        
        # Display the metrics table
        st.write("#### Aggregate Metrics by Fold")
        st.dataframe(
            fold_metrics_df.set_index('Fold')
            .style.format({
                'Accuracy': "{:.4f}",
                'Macro Precision': "{:.4f}",
                'Macro Recall': "{:.4f}",
                'Macro F1-Score': "{:.4f}",
                'Weighted Precision': "{:.4f}",
                'Weighted Recall': "{:.4f}",
                'Weighted F1-Score': "{:.4f}"
            })
        )
        
        # Create parallel coordinates plot for comparing folds
        parallel_fig = px.parallel_coordinates(
            fold_metrics_df,
            color="Accuracy",
            dimensions=['Fold', 'Accuracy', 'Macro Precision', 'Macro Recall', 
                        'Macro F1-Score', 'Weighted F1-Score'],
            labels={
                'Fold': 'Fold Number',
                'Accuracy': 'Accuracy',
                'Macro Precision': 'Macro Precision',
                'Macro Recall': 'Macro Recall',
                'Macro F1-Score': 'Macro F1-Score',
                'Weighted F1-Score': 'Weighted F1-Score'
            },
            color_continuous_scale='Viridis',
            title="Parallel Coordinates Plot of Metrics Across Folds"
        )
        st.plotly_chart(parallel_fig, use_container_width=True)
        
        # Show boxplots of metrics distribution across folds
        metrics_long_df = pd.melt(
            fold_metrics_df, 
            id_vars=['Fold'],
            value_vars=['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score', 
                        'Weighted Precision', 'Weighted Recall', 'Weighted F1-Score'],
            var_name='Metric',
            value_name='Value'
        )
        
        box_fig = px.box(
            metrics_long_df,
            x="Metric",
            y="Value",
            points="all",
            title="Distribution of Metrics Across Folds",
        )
        box_fig.update_layout(template='plotly_white')
        st.plotly_chart(box_fig, use_container_width=True)
        
        # Add statistical summary of metrics
        st.write("#### Statistical Summary of Metrics Across Folds")
        metrics_summary = fold_metrics_df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        st.dataframe(
            metrics_summary
            .style.format({
                'mean': "{:.4f}",
                'std': "{:.4f}",
                'min': "{:.4f}",
                '25%': "{:.4f}",
                '50%': "{:.4f}",
                '75%': "{:.4f}",
                'max': "{:.4f}"
            })
        )
## ----- Cross Validation Tab END -----

## ----- HIDE STREAMLIT STYLE -----
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
## ----- HIDE STREAMLIT STYLE END -----
## ----- Main Application END -----