import numpy as np

class NaiveBayes:
    """
    Kelas untuk mengimplementasikan Naive Bayes.
    """

    def __init__(self):
        """
        Inisialisasi Naive Bayes.
        """
        self.prior = None
        self.mean = None
        self.std = None

    def fit(self, X, y):
        """
        Melatih model Naive Bayes.

        Args:
            X: Data latih (numpy array).
            y: Label kelas (numpy array).
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Hitung prior probability
        self.prior = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.prior[i] = np.sum(y == c) / n_samples

        # Hitung mean dan standard deviation untuk setiap fitur dan kelas
        self.mean = np.zeros((n_classes, n_features))
        self.std = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i, :] = np.mean(X_c, axis=0)
            self.std[i, :] = np.std(X_c, axis=0)

    def predict(self, X):
        """
        Memprediksi kelas untuk data baru.

        Args:
            X: Data baru (numpy array).

        Returns:
            Prediksi kelas (numpy array).
        """
        y_pred = [self._predict_sample(x) for x in X]
        return np.array(y_pred)

    def _predict_sample(self, x):
        """
        Memprediksi kelas untuk satu sampel data.

        Args:
            x: Satu sampel data (numpy array).

        Returns:
            Prediksi kelas.
        """
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.prior[i])
            likelihood = np.sum(np.log(self._gaussian_pdf(i, x)))
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def _gaussian_pdf(self, class_idx, x):
        """
        Menghitung probability density function (PDF) Gaussian.

        Args:
            class_idx: Indeks kelas.
            x: Sampel data (numpy array).

        Returns:
            Nilai PDF Gaussian.
        """
        mean = self.mean[class_idx]
        std = self.std[class_idx]
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

# Contoh penggunaan
X_train = np.array([
    [1, 2], [2, 3], [3, 1],  # Kelas 0
    [4, 3], [5, 3], [6, 2],  # Kelas 1
    [7, 1], [8, 2], [9, 3]   # Kelas 2
])
y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])  # Label kelas 0, 1, dan 2

# Data uji
X_test = np.array([[5, 2], [8, 1]])

# Inisialisasi Naive Bayes
nb = NaiveBayes()

# Latih model
nb.fit(X_train, y_train)

# Prediksi
y_pred = nb.predict(X_test)

# Tampilkan hasil prediksi
print("Prediksi:", y_pred)