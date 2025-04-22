"""
Cross validation tab module for the dashboard
"""
import streamlit as st
import pandas as pd
from src.dashboard.components.metrics import calculate_cross_validation_results
from src.dashboard.components.plots import (
    plot_cv_bar,
    plot_fold_accuracies,
    plot_class_metrics_by_fold,
    plot_radar_chart,
    plot_parallel_coordinates,
    plot_metrics_boxplot,
    plot_confusion_matrix,
    LABEL_NAMES
)
from src.dashboard.components.widgets import display_fold_metrics, display_class_metrics_by_fold

def render_cv_tab(feature_dataframes, summary_df):
    """Render the cross validation tab content"""
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
    cv_bar_fig = plot_cv_bar(cv_summary_df)
    st.plotly_chart(cv_bar_fig, use_container_width=True)
    
    # 2. Detailed view of selected combination
    st.subheader("Detailed Cross Validation Results")
    
    # Select a (d, theta, k) combination to view detailed results
    cv_combo_options = [(d, theta, k) for d, theta, k in cv_results.keys()]
    selected_combo, selected_cv_result = display_fold_metrics(cv_combo_options, cv_results, best_cv_combo)
    
    # Display fold accuracies
    fold_accuracies = [fold_result['accuracy'] for fold_result in selected_cv_result['fold_results']]
    
    # Bar chart of fold accuracies
    fold_fig = plot_fold_accuracies(fold_accuracies, selected_cv_result['mean_cv_score'])
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
                        fold_cm_fig = plot_confusion_matrix(fold_result['confusion_matrix'])
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
                        'Class Name': LABEL_NAMES.get(int(class_label), f"Class {class_label}"),
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
        class_metric_fig = plot_class_metrics_by_fold(class_metrics_df, selected_metric)
        st.plotly_chart(class_metric_fig, use_container_width=True)
        
        # Show detailed metrics table with expandable sections by class
        st.subheader("Detailed Class Metrics by Fold")
        
        # Display metrics for each class
        for class_label, class_data in display_class_metrics_by_fold(class_metrics_data, LABEL_NAMES):
            # Add radar chart for this class across folds
            radar_fig = plot_radar_chart(class_data)
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
        parallel_fig = plot_parallel_coordinates(fold_metrics_df)
        st.plotly_chart(parallel_fig, use_container_width=True)
        
        # Show boxplots of metrics distribution across folds
        box_fig = plot_metrics_boxplot(fold_metrics_df)
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
        
    return cv_summary_df