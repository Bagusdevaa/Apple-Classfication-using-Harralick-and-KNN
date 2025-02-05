import numpy as np
from scipy import signal

class Gabor:
    """
    Kelas untuk menghitung filter Gabor dan mengekstrak fitur.
    """
    def __init__(self, sigma, theta, Lambda, gamma, frequency):
        """
        Inisialisasi objek Gabor.

        Args:
            sigma: Deviasi standar fungsi Gaussian.
            theta: Orientasi filter dalam radian.
            Lambda: Panjang gelombang faktor kosinus.
            gamma: Aspek rasio fungsi Gaussian.
            frequency: Frekuensi kernel Gabor.
        """
        self.sigma = sigma
        self.theta = theta
        self.Lambda = Lambda
        self.gamma = gamma
        self.frequency = frequency

    def calculate_kernel(self):
        """
        Menghitung kernel Gabor.

        Returns:
            Kernel Gabor.
        """
        kernel_size = int(3 * self.sigma)
        x = np.arange(-kernel_size, kernel_size + 1)
        y = np.arange(-kernel_size, kernel_size + 1)
        x, y = np.meshgrid(x, y)

        # Hitung kernel Gabor
        x_theta = x * np.cos(self.theta) + y * np.sin(self.theta)
        y_theta = -x * np.sin(self.theta) + y * np.cos(self.theta)
        kernel = np.exp(-(x_theta**2 + (self.gamma**2 * y_theta**2)) / (2 * self.sigma**2)) * \
                 np.cos(2 * np.pi * x_theta / self.Lambda)

        return kernel

    def apply_filter(self, img):
        """
        Mengaplikasikan filter Gabor pada citra.

        Args:
            img: Citra grayscale yang akan diaplikasikan filter Gabor.

        Returns:
            Citra yang telah diaplikasikan filter Gabor.
        """
        kernel = self.calculate_kernel()
        filtered_img = signal.convolve2d(img, kernel, mode='same')

        return filtered_img

    def get_features(self, filtered_img):
        """
        Menghitung fitur dari citra yang telah diaplikasikan filter Gabor.

        Args:
            filtered_img: Citra yang telah diaplikasikan filter Gabor.

        Returns:
            Array fitur (mean, std, skewness, kurtosis, energi, fase).
        """
        features = np.zeros(6, dtype=np.float32)

        features[0] = np.mean(filtered_img)
        features[1] = np.std(filtered_img)

        std = np.std(filtered_img)
        if std != 0:
            features[2] = np.mean((filtered_img - np.mean(filtered_img)) ** 3) / (std ** 3)
            features[3] = np.mean((filtered_img - np.mean(filtered_img)) ** 4) / (std ** 4) - 3
        else:
            features[2] = np.nan
            features[3] = np.nan

        features[4] = np.sum(filtered_img ** 2)
        features[5] = np.arctan2(np.imag(filtered_img), np.real(filtered_img)).mean()

        return features

# # Contoh penggunaan
# img = np.array([
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12],
#     [13, 14, 15, 16]
# ])

# gabor_filter = Gabor(sigma=1, theta=0, frequency=0.1, Lambda=10, gamma=0.5)
# filtered_img = gabor_filter.apply_filter(img)
# features = gabor_filter.get_features(filtered_img)

# print(features)