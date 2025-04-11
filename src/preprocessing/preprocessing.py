import cv2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def general_preprocessing(file_path):
    '''
    FUNGSI GENERAL PREPROCESSING UNTUK MELAKUKAN PREPROCESSING SECARA UMUM SEBELUM MASUK KE SPESIFIK PREPROCESSING
    '''
    try:
        # Load the image
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Image not loaded: {file_path}")
        
        # Convert to grayscale (example preprocessing step)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize the image (example preprocessing step)
        image = cv2.resize(image, (128, 128))
        
        return image
    except Exception as e:
        print(f"Error in preprocessing {file_path}: {e}")
        return None
    

def preprocess_data(df, drop_columns):
    """
    Preprocess the dataset by normalizing numeric features and splitting into train/test sets.
    """
    numeric_features = df.drop(columns=drop_columns, axis=1).columns
    scaler = MinMaxScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    X = df.drop(columns=drop_columns)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test