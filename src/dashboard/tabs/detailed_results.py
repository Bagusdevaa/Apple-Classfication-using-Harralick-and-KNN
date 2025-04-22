"""
Detailed results tab module for the dashboard
"""
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from src.preprocessing.preprocessing import preprocess_data
from src.dashboard.components.metrics import calculate_classification_report
from src.dashboard.components.plots import (
    plot_accuracy_line,
    plot_confusion_matrix,
    plot_class_metrics,
    plot_neighbors_visualization
)
from src.dashboard.components.widgets import display_best_combination_metrics, display_test_sample_info

def render_detailed_results_tab(feature_dataframes, all_results, best_combination):
    """Render the detailed results tab content"""
    st.header("Detailed Results")
    
    st.write(f"""Base on the best combination from the summary report, as we can see the best combination is""")
    
    # Display best combination metrics
    display_best_combination_metrics(best_combination)
    
    st.write(f"""but you can select the other (d, theta) and K combination to see the detailed results.""")

    ## ----- Parameter Selection -----
    # Select a combination (d, theta)
    d_theta = st.selectbox("Select (d, theta) combination", list(all_results.keys()), help="Choose a combination of d and theta to visualize the results")
    results = all_results[d_theta]

    # Interactive Line Chart for Accuracy vs. k
    line_fig = plot_accuracy_line(results, d_theta)
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
            
        # Plot confusion matrix
        conf_fig = plot_confusion_matrix(conf_matrix)
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
        
        # Plot metrics
        bar_fig = plot_class_metrics(calculate_classification_report(y_test, y_pred))
        st.plotly_chart(bar_fig, use_container_width=True)
    ## ----- Results Metrics Display END -----
    
    ## ----- Nearest Neighbor Visualization -----
    st.subheader("Nearest Neighbor Visualization")
    
    # Use a form to avoid refreshing the entire page
    with st.form("test_sample_form"):
        test_sample_idx = st.slider("Select Test Sample Index", 0, len(y_test)-1, 27)
        submit_button = st.form_submit_button("Update Visualization")

    # Get data for visualization
    X_train, X_test, y_train, y_test = preprocess_data(
        feature_dataframes[d_theta], 
        drop_columns=['label', 'homogeneity', 'sum_variance', 'ASM', 'IMC2', 'image']
    )
    
    # Create visualization
    fig = plot_neighbors_visualization(X_train, X_test, y_train, y_test, test_sample_idx, selected_k)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display test sample prediction info
    display_test_sample_info(y_test, y_pred, test_sample_idx)
    ## ----- Nearest Neighbor Visualization END -----
    
    return results