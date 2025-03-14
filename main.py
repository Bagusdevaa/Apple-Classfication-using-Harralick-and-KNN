import numpy as np

# Fungsi untuk menghitung jarak (Euclidean Distance)
def euclidean_distance(point1, point2):
    return sum((x2 - x1) ** 2 for x2, x1 in zip(point1, point2)) ** 0.5

# Kelas KNN
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.data = []
        self.labels = []

    # Metode untuk melatih model
    def fit(self, data, labels):
        self.data = data
        self.labels = labels

    # Metode untuk memprediksi kelas dari titik baru
    def predict(self, new_points):
        predictions = []
        for new_point in new_points:
            # Hitung jarak dari titik baru ke semua titik dalam data
            distances = []
            for i in range(len(self.data)): 
                distance = euclidean_distance(self.data[i], new_point)
                distances.append((distance, self.labels[i]))
            
            # Urutkan berdasarkan jarak
            distances.sort(key=lambda x: x[0])
            
            # Ambil K tetangga terdekat
            neighbors = distances[:self.k]
            
            # Lakukan voting untuk menentukan kelas
            votes = {}
            for _, label in neighbors:
                if label in votes:
                    votes[label] += 1
                else:
                    votes[label] = 1
            
            # Temukan kelas dengan suara terbanyak
            most_common = max(votes.items(), key=lambda x: x[1])
            
            predictions.append(most_common[0])
        
        return predictions




from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load fitur harralicknya

kombinasiFeature = [[1, 2, 3], [0, 45, 90, 135]]

list_harralick_dfs = []

for d in kombinasiFeature[0]:
    for theta in kombinasiFeature[1]:
        df = pd.read_csv(f"dataset/ExtractResult/harralick/features_d{d}_theta{theta}.csv")
        df = df.set_index('image')
        list_harralick_dfs.append(df)

# Inisialisasi list untuk menyimpan hasil evaluasi
hasil_evaluasi = []

# Looping pada setiap fitur dalam list_harralick_dfs
for i, df in enumerate(list_harralick_dfs):
    # Pisahkan data menjadi fitur (X) dan target (y)
    X = df.drop('label', axis=1) 
    y = df['label']

    # Pisahkan data menjadi training dan testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi list untuk menyimpan hasil evaluasi pada setiap fitur
    hasil_fitur = []

    # Looping pada variasi K dari 3 sampai 15
    for k in range(3, 16):
        # Buat model KNN dengan nilai K yang saat ini
        knn = KNN(k=k)

        # Latih model KNN pada data training
        knn.fit(X_train, y_train)

        # Prediksi target untuk setiap titik data testing
        y_pred = []
        for x in X_test.values:
            y_pred.append(knn.predict(x))

        # Evaluasi model KNN menggunakan metrik akurasi, presisi, recall, dan f1-score
        akurasi = accuracy_score(y_test, y_pred)
        presisi = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Simpan hasil evaluasi pada list
        hasil_fitur.append({
            'K': k,
            'Akurasi': akurasi,
            'Presisi': presisi,
            'Recall': recall,
            'F1-score': f1
        })

    # Simpan hasil evaluasi pada list utama
    hasil_evaluasi.append({
        'Fitur': i,
        'Hasil Evaluasi': hasil_fitur
    })