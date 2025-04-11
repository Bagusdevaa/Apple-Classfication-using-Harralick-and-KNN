from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 

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