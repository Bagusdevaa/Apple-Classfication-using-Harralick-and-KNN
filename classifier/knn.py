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
    def predict(self, new_point):
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
        
        return most_common[0]  # Kembalikan kelas dengan suara terbanyak