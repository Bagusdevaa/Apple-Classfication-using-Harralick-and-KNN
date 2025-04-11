import numpy as np
import pandas as pd

# Fungsi untuk menghitung jarak (Euclidean Distance)
def euclidean_distance(point1, point2):
    point1 = np.array(point1, dtype=np.float64)
    point2 = np.array(point2, dtype=np.float64)
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Kelas KNN
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.data = None  # DataFrame
        self.labels = None  # Pandas Series

    # Metode untuk melatih model
    def fit(self, data, labels):
        self.data = data.reset_index(drop=True)  # Pastikan index sesuai
        self.labels = labels.reset_index(drop=True)

    # Metode untuk memprediksi kelas dari titik baru
    def predict(self, new_points):
        predictions = []
        new_points = new_points.to_numpy(dtype=np.float64)  # Konversi menjadi numpy array

        for new_point in new_points:
            # Hitung jarak dari titik baru ke semua titik dalam data
            distances = []
            for i in range(len(self.data)): 
                distance = euclidean_distance(self.data.iloc[i].values, new_point)
                distances.append((distance, self.labels.iloc[i]))
            
            # Urutkan berdasarkan jarak
            distances.sort(key=lambda x: x[0])
            
            # Ambil K tetangga terdekat
            neighbors = distances[:self.k]
            
            # Lakukan voting untuk menentukan kelas
            votes = {}
            for _, label in neighbors:
                votes[label] = votes.get(label, 0) + 1
            
            # Temukan kelas dengan jumlah suara terbanyak
            most_common = max(votes.items(), key=lambda x: x[1])[0]
            
            predictions.append(most_common)
        
        return predictions