"""
UI widget components for the dashboard
"""
import streamlit as st
from pathlib import Path
import pandas as pd
from src.dashboard.components.plots import LABEL_NAMES, LABEL_COLORS

def display_best_combination_metrics(best_combination):
    """Display best combination metrics in columns"""
    col_d, col_theta, col_k, col_accuracy = st.columns([1, 1, 1, 1]) 
    with col_d:
        st.write(f"```\nd: \n{best_combination['d']}", help="Best d value")
    with col_theta:
        st.write(f"```\ntheta: \n{best_combination['theta']}", help="Best theta value")
    with col_k:
        st.write(f"```\nk: \n{best_combination['best_k']}", help="Best k value")
    with col_accuracy:
        st.write(f"```\nAccuracy: \n{best_combination['accuracy']:.2f}", help="Best accuracy value")

def display_test_sample_info(y_test, y_pred, test_sample_idx):
    """Display test sample prediction info"""
    true_label = y_test.iloc[test_sample_idx]
    pred_label = y_pred[test_sample_idx]
    st.write(f"**Test Sample #{test_sample_idx}:**")
    st.write(f"- True Label: {true_label} ({LABEL_NAMES.get(true_label, 'Unknown')})")
    st.write(f"- Predicted Label: {pred_label} ({LABEL_NAMES.get(pred_label, 'Unknown')})")
    st.write(f"- Prediction {'Correct ✅' if true_label == pred_label else 'Incorrect ❌'}")

def display_fold_metrics(cv_combo_options, cv_results, best_cv_combo=None):
    """
    Display fold metrics selector and metrics table
    Returns the selected combination
    """
    # Find index of the best combination in the list of options
    best_index = 0
    if best_cv_combo is not None:
        best_combo = (best_cv_combo['d'], best_cv_combo['theta'], best_cv_combo['k'])
        for i, combo in enumerate(cv_combo_options):
            if combo == best_combo:
                best_index = i
                break
                
    # Select a (d, theta, k) combination to view detailed results
    selected_combo_str = st.selectbox(
        "Select a combination to view detailed fold results:",
        options=[f"d={d}, theta={theta}, k={k}" for d, theta, k in cv_combo_options],
        index=best_index
    )
    
    # Parse the selected combo string back to tuple
    parts = selected_combo_str.split(', ')
    selected_d = int(float(parts[0].split('=')[1]))
    selected_theta = int(float(parts[1].split('=')[1]))
    selected_k = int(float(parts[2].split('=')[1]))
    selected_combo = (selected_d, selected_theta, selected_k)
    
    # Get the detailed results for the selected combination
    selected_cv_result = cv_results[selected_combo]
    
    return selected_combo, selected_cv_result

def display_class_metrics_by_fold(class_metrics_data, label_names):
    """Display class metrics by fold with expandable sections"""
    # Group by class for expandable sections
    for class_label in sorted(set([item['Class'] for item in class_metrics_data])):
        class_name = label_names.get(class_label, f"Class {class_label}")
        with st.expander(f"{class_name} (Class {class_label}) Metrics"):
            class_data = pd.DataFrame([item for item in class_metrics_data if item['Class'] == class_label])
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
            
            # Yield the class data for potential further use (like plotting)
            yield class_label, class_data

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