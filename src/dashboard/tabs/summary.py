"""
Summary tab module for the dashboard
"""
import streamlit as st
import pandas as pd
from src.dashboard.components.metrics import preprocess_and_calculate_results
from src.dashboard.components.plots import plot_accuracy_bar

def render_summary_tab(feature_dataframes):
    """Render the summary tab content"""
    st.header("Summary Report")
    
    tab1col1, tab1col2 = st.columns([4, 6])
    
    with tab1col1:
        # Calculate KNN results for all combinations
        k_values = range(3, 26, 2)  # odd values for k
        summary_df, all_results = preprocess_and_calculate_results(feature_dataframes, k_values)
        
        # Display summary dataframe
        st.dataframe(summary_df)
        
        # Find and display best combination
        best_combination = summary_df.loc[summary_df['accuracy'].idxmax()]
        st.write(f"**Best Combination:** d={best_combination['d']}, theta={best_combination['theta']}, k={best_combination['best_k']}, Accuracy={best_combination['accuracy']:.2f}")
    
    with tab1col2:
        # Interactive bar chart
        fig = plot_accuracy_bar(summary_df)
        st.plotly_chart(fig, use_container_width=True)
    
    return summary_df, all_results, best_combination