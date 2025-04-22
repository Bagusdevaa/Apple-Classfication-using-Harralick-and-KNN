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
import matplotlib.pyplot as plt
from pathlib import Path

# Import local modules using absolute imports instead of relative imports
from src.dashboard.components.widgets import load_harralick_features
from src.dashboard.tabs.summary import render_summary_tab
from src.dashboard.tabs.detailed_results import render_detailed_results_tab
from src.dashboard.tabs.cv_tab import render_cv_tab
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

## ----- Path and Data Loading -----
# Update BASE_DIR to use Pathlib for dynamic path resolution
BASE_DIR = Path(__file__).resolve().parent.parent.parent / 'dataset'

@st.cache_data
def load_feature_data():
    kombinasiFeature = [[1, 2, 3], [0, 45, 90, 135]]
    return load_harralick_features(BASE_DIR, kombinasiFeature)
## ----- Path and Data Loading END -----

## ----- Main Application -----
def main():
    """Main application entry point"""
    
    # App title and description
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

    # Load Dataset
    feature_dataframes = load_feature_data()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Summary Report", "Detailed Results", "Cross Validation"])
    
    # Render each tab
    with tab1:
        summary_df, all_results, best_combination = render_summary_tab(feature_dataframes)
    
    with tab2:
        results = render_detailed_results_tab(feature_dataframes, all_results, best_combination)
    
    with tab3:
        cv_summary_df = render_cv_tab(feature_dataframes, summary_df)
    
    ## ----- HIDE STREAMLIT STYLE -----
    hide_st_style = """
                <style>
                footer {visibility: hidden;}
                
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    ## ----- HIDE STREAMLIT STYLE END -----

if __name__ == "__main__":
    main()
## ----- Main Application END -----