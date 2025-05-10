# Apple Classification using Haralick Features and KNN

## Features
- **Preprocessing**: Includes resize, convert to grayscale, and preparing images for feature extraction.
- **Feature Extraction**: Utilizes Haralick texture features with `d=(1,2,3)` and `theta=(0,45,90,135)` combination.
- **Classification**: Implements KNN for classifying apple ripeness levels with different K values to each dataset combination.
- **Dashboard**: Interactive Streamlit dashboard for visualizing results and exploring data.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Bagusdevaa/Apple-Classfication-using-Harralick-and-KNN.git
   cd Apple-Classfication-using-Harralick-and-KNN
   ```
2. Set up a Python virtual environment:
   ```bash
   python -m venv skripsi
   skripsi\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running the Dashboard
1. Navigate to the `src/dashboard` directory.
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the provided URL in your browser to access the dashboard.

### Notebooks
- Explore the Jupyter notebooks in the `notebooks/` directory for preprocessing, feature extraction, and classification.

## Dataset
The dataset is organized into folders based on ripeness levels (20%, 40%, 60%, 80%, 100%). Each folder contains images of apples at the corresponding ripeness level.

## Results
- The dashboard provides visualizations of classification results, including accuracy, precision, recall, f1-score, confusion matrices, and Nearest Neighbors Visualizatoin.
- The best KNN hyperparameters and K value of KNN are determined through experimentation and displayed in the dashboard.

## License
This project is for educational purposes and is not licensed for commercial use.
