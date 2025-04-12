import numpy as np
from skimage.feature import graycomatrix, graycoprops
import pandas as pd

# Fungsi untuk menghitung GLCM dan fitur Harralick
def calculate_additional_features(glcm):
    features = {}
    # Sum of squares: Variance
    features['sum_of_squares'] = np.sum((np.arange(glcm.shape[0]) - np.mean(glcm))**2 * glcm)

    # Sum average
    px_plus_py = np.sum(glcm, axis=0) + np.sum(glcm, axis=1)
    features['sum_average'] = np.sum(np.arange(len(px_plus_py)) * px_plus_py)

    # Sum variance
    features['sum_variance'] = np.sum((np.arange(len(px_plus_py)) - features['sum_average'])**2 * px_plus_py)

    # Sum entropy
    features['sum_entropy'] = -np.sum(px_plus_py * np.log2(px_plus_py + 1e-10))

    # Difference variance
    px_minus_py = np.abs(np.sum(glcm, axis=0) - np.sum(glcm, axis=1))
    features['difference_variance'] = np.var(px_minus_py)

    # Difference entropy
    features['difference_entropy'] = -np.sum(px_minus_py * np.log2(px_minus_py + 1e-10))

    # Information measure of correlation 1
    hxy = -np.sum(glcm * np.log2(glcm + 1e-10))
    hxy1 = -np.sum(np.sum(glcm, axis=0) * np.log2(np.sum(glcm, axis=0) + 1e-10))
    hxy2 = -np.sum(np.sum(glcm, axis=1) * np.log2(np.sum(glcm, axis=1) + 1e-10))
    features['IMC1'] = (hxy - hxy1) / max(hxy, hxy2)

    # Information measure of correlation 2
    hx = -np.sum(np.sum(glcm, axis=0) * np.log2(np.sum(glcm, axis=0) + 1e-10))
    hy = -np.sum(np.sum(glcm, axis=1) * np.log2(np.sum(glcm, axis=1) + 1e-10))
    features['IMC2'] = np.sqrt(1 - np.exp(-2 * (hxy - hx - hy)))

    return features

import streamlit as st
def load_harralick_features(dataset_path, kombinasiFeature):
    """
    Load Haralick features for all (d, theta) combinations.
    """
    feature_dataframes = {}
    for d in kombinasiFeature[0]:
        for theta in kombinasiFeature[1]:
            if dataset_path.exists():
                st.write(f"Dataset file found: {dataset_path}")
                df = pd.read_csv(dataset_path)  # Sesuaikan dengan format file Anda
                st.write("Dataset loaded successfully!")
                st.dataframe(df)
            else:
                st.error(f"Dataset file not found: {dataset_path}")
            try:
                df = pd.read_csv(f"{dataset_path}\\ExtractResult\\harralick\\features_d{d}_theta{theta}.csv")
                df['image'] = [f"img-{i}" for i in range(len(df))]
                feature_dataframes[(d, theta)] = df
            except FileNotFoundError:
                print(f"Dataset for d={d}, theta={theta} not found. Skipping...")
    return feature_dataframes

# class Harralick:
#   """
#   Kelas untuk menghitung matriks GLCM dan fitur tekstur.
#   """

#   def __init__(self, d, theta):
#     """
#     Inisialisasi objek GLCM.

#     Args:
#       d: Jarak antara dua piksel.
#       theta: Sudut (dalam derajat) antara dua piksel (0, 45, 90, 135).
#     """
#     self.d = d
#     self.theta = theta

#   def calculate_glcm(self, img):
#     """
#     Menghitung matriks GLCM untuk citra keabuan.

#     Args:
#       img: Citra keabuan dalam bentuk array NumPy.

#     Returns:
#       Matriks GLCM yang telah dinormalisasi.
#     """

#     rows, cols = img.shape
#     levels = np.max(img) + 1 # Jumlah tingkat keabuan
#     if levels > 256:  # Jika nilai maksimum gambar melebihi 256
#       levels = 256  # Ubah ukuran glcm_matrix menjadi 256
#     glcm_matrix = np.zeros((levels, levels), dtype=np.uint32)

#     # Konversi sudut ke radian
#     theta_rad = np.radians(self.theta)

#     # Hitung offset piksel berdasarkan jarak dan sudut
#     row_offset = int(self.d * np.sin(theta_rad))
#     col_offset = int(self.d * np.cos(theta_rad))

#     # Iterasi melalui piksel-piksel citra
#     for i in range(rows):
#       for j in range(cols):
#         # Hitung koordinat piksel tetangga
#         row_neighbor = i + row_offset
#         col_neighbor = j + col_offset

#         # Pastikan piksel tetangga masih dalam batas citra
#         if 0 <= row_neighbor < rows and 0 <= col_neighbor < cols:
#           # Dapatkan nilai keabuan piksel dan tetangganya
#           pixel_value = img[i, j]
#           neighbor_value = img[row_neighbor, col_neighbor]

#           # Increment nilai pada matriks GLCM
#           glcm_matrix[pixel_value, neighbor_value] += 1

#     # Normalisasi matriks GLCM
#     glcm_matrix = glcm_matrix / np.sum(glcm_matrix)

#     return glcm_matrix

#   def energy(self, glcm_matrix): # atau bisa juga Angular Second Moment (ASM)
#     """
#     Mengukur homogenitas citra. Tekstur yang homogen akan memiliki nilai ASM tinggi.
#     """
#     energy = np.sum(glcm_matrix ** 2)
#     return energy
  
#   def contrast(self, glcm_matrix):
#     """
#     Mengukur perbedaan intensitas antara piksel tetangga. 
#     Citra dengan tekstur kasar cenderung memiliki nilai kontras tinggi.
#     """
#     contrast = 0
#     levels = glcm_matrix.shape[0]
#     for i in range(levels):
#       for j in range(levels):
#         contrast += (i - j) ** 2 * glcm_matrix[i, j]
#     return contrast

#   def correlation(self, glcm_matrix):
#     """
#     Menunjukkan seberapa besar ketergantungan linear antara piksel-piksel dalam citra.
#     """
#     correlation = 0
#     levels = glcm_matrix.shape[0]
#     mean_x = np.sum(glcm_matrix * np.arange(levels))
#     mean_y = np.sum(glcm_matrix.T * np.arange(levels))
#     std_x = np.sqrt(np.sum(glcm_matrix * (np.arange(levels) - mean_x) ** 2))
#     std_y = np.sqrt(np.sum(glcm_matrix.T * (np.arange(levels) - mean_y) ** 2))
#     for i in range(levels):
#       for j in range(levels):
#         correlation += ((i - mean_x) * (j - mean_y) * glcm_matrix[i, j]) / (std_x * std_y)
#     return correlation

#   def variance(self, glcm_matrix):
#     """
#     Mengukur variasi nilai piksel di sekitar nilai rata-rata keabuan.
#     """
#     variance = 0
#     levels = glcm_matrix.shape[0]
#     mean = np.sum(glcm_matrix * np.arange(levels))
#     for i in range(levels):
#       for j in range(levels):
#         variance += (i - mean) ** 2 * glcm_matrix[i, j]
#     return variance

#   def homogeneity(self, glcm_matrix): # atau bisa juga Inverse Difference Moment (IDM)
#     """
#     Mengukur homogenitas lokal citra. 
#     Nilai IDM tinggi menandakan sedikit variasi intensitas di sekitar piksel.
#     """
#     homogeneity = 0
#     levels = glcm_matrix.shape[0]
#     for i in range(levels):
#       for j in range(levels):
#         homogeneity += glcm_matrix[i, j] / (1 + abs(i - j))
#     return homogeneity

#   def sum_average(self, glcm_matrix):
#     """
#     Tidak memiliki interpretasi fisik langsung, namun sensitif terhadap perubahan homogenitas citra.
#     """
#     sum_average = 0
#     levels = glcm_matrix.shape[0]
#     for i in range(levels):
#       for j in range(levels):
#         sum_average += (i + j) * glcm_matrix[i, j]
#     return sum_average
  
#   def sum_variance(self, glcm_matrix):
#     """
#     Mengukur homogenitas citra dan sensitif terhadap variasi di sekitar nilai rata-rata keabuan.
#     """
#     sum_variance = 0
#     levels = glcm_matrix.shape[0]
#     mean = np.sum(glcm_matrix * np.arange(levels))
#     for i in range(levels):
#       for j in range(levels):
#         sum_variance += (i + j - mean) ** 2 * glcm_matrix[i, j]
#     return sum_variance
  
#   def sum_entropy(self, glcm_matrix):
#     """
#     Mengukur keacakan atau ketidakpastian dalam citra.
#     """
#     sum_entropy = 0
#     levels = glcm_matrix.shape[0]
#     for i in range(levels):
#       for j in range(levels):
#         if glcm_matrix[i, j] != 0:
#           sum_entropy -= glcm_matrix[i, j] * np.log2(glcm_matrix[i, j])
#     return sum_entropy
  
#   def entropy(self, glcm_matrix):
#     """
#     Mirip dengan Sum Entropy, mengukur kompleksitas atau keacakan tekstur.
#     """
#     entropy = 0
#     for i in range(glcm_matrix.shape[0]):
#       for j in range(glcm_matrix.shape[1]):
#         if glcm_matrix[i, j] != 0:
#           entropy -= glcm_matrix[i, j] * np.log2(glcm_matrix[i, j])
#     return entropy
  
#   def difference_variance(self, glcm_matrix):
#     """
#     Mengukur variasi nilai piksel yang berbeda.
#     """
#     difference_variance = 0
#     levels = glcm_matrix.shape[0]
#     mean = np.sum(glcm_matrix * np.arange(levels))
#     for i in range(levels):
#       for j in range(levels):
#         difference_variance += (abs(i - j) - mean) ** 2 * glcm_matrix[i, j]
#     return difference_variance
  
#   def difference_entropy(self, glcm_matrix):
#     """
#     Mengukur keacakan atau ketidakpastian yang berhubungan dengan perbedaan nilai piksel.
#     """
#     difference_entropy = 0
#     levels = glcm_matrix.shape[0]
#     for i in range(levels):
#       for j in range(levels):
#         if glcm_matrix[i, j] != 0:
#           difference_entropy -= glcm_matrix[i, j] * np.log2(glcm_matrix[i, j])
#     return difference_entropy
  
#   def information_measure_of_correlation1(self, glcm_matrix):
#     """
#     Salah satu ukuran korelasi antara piksel-piksel.
#     """
#     information_measure_of_correlation1 = 0
#     levels = glcm_matrix.shape[0]
#     for i in range(levels):
#       for j in range(levels):
#         if glcm_matrix[i, j] != 0 and glcm_matrix[i, i] != 0 and glcm_matrix[j, j] != 0:
#           information_measure_of_correlation1 += glcm_matrix[i, j] * np.log2(glcm_matrix[i, j] / (glcm_matrix[i, i] * glcm_matrix[j, j]))
#     return information_measure_of_correlation1
  
#   def information_measure_of_correlation2(self, glcm_matrix):
#     """
#     Menghitung fitur information measure of correlation 2 dari matriks GLCM.
#     Ukuran korelasi lain yang mirip dengan IMC1.
#     """
#     information_measure_of_correlation2 = 0
#     levels = glcm_matrix.shape[0]
#     for i in range(levels):
#       for j in range(levels):
#         if glcm_matrix[i, j] != 0 and glcm_matrix[i, i] != 0 and glcm_matrix[j, j] != 0:
#           information_measure_of_correlation2 += glcm_matrix[i, j] * np.log2(glcm_matrix[i, j] / (glcm_matrix[i, i] * glcm_matrix[j, j]))
#     return information_measure_of_correlation2
  
#   def max_correlation_coefficient(self, glcm_matrix):
#     """
#     Menghitung fitur max correlation coefficient dari matriks GLCM.
#     Mengukur korelasi maksimum antara piksel-piksel.
#     """
#     max_correlation_coefficient = 0
#     levels = glcm_matrix.shape[0]
#     for i in range(levels):
#       for j in range(levels):
#         max_correlation_coefficient = max(max_correlation_coefficient, glcm_matrix[i, j])
#     return max_correlation_coefficient
  
#   def get_all_features(self, glcm_matrix):
#     """
#     Menghitung semua fitur dari matriks GLCM.
#     """
#     features = {
#       'energy': self.energy(glcm_matrix),
#       'contrast': self.contrast(glcm_matrix),
#       'correlation': self.correlation(glcm_matrix),
#       'variance': self.variance(glcm_matrix),
#       'homogeneity': self.homogeneity(glcm_matrix),
#       'sum_average': self.sum_average(glcm_matrix),
#       'sum_variance': self.sum_variance(glcm_matrix),
#       'sum_entropy': self.sum_entropy(glcm_matrix),
#       'entropy': self.entropy(glcm_matrix),
#       'difference_variance': self.difference_variance(glcm_matrix),
#       'difference_entropy': self.difference_entropy(glcm_matrix),
#       'information_measure_of_correlation1': self.information_measure_of_correlation1(glcm_matrix),
#       'information_measure_of_correlation2': self.information_measure_of_correlation2(glcm_matrix),
#       'max_correlation_coefficient': self.max_correlation_coefficient(glcm_matrix),
#     }
#     return features
  
# # Contoh penggunaan
# img = np.array([
#   [0, 0, 1, 1],
#   [0, 0, 1, 1],
#   [0, 2, 2, 2],
#   [2, 2, 3, 3]
# ])

# # Inisialisasi objek GLCM dengan d=1 dan theta=0
# glcm_calculator = Harralick(d=2, theta=45)

# # Hitung matriks GLCM
# glcm_matrix = glcm_calculator.calculate_glcm(img)

# # Hitung semua fitur
# features = glcm_calculator.get_all_features(glcm_matrix)

# # Tampilkan fitur
# for feature, value in features.items():
#   print(f'{feature}: {value}')