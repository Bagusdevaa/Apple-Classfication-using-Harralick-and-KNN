import numpy as np

class LBP:
  def __init__(self, radius=1, neighbors=8):
      self.radius = radius
      self.neighbors = neighbors

  def calculate_lbp(self, img):
      rows, cols = img.shape
      lbp_img = np.zeros((rows, cols), dtype=np.uint8)

      for i in range(rows):
          for j in range(cols):
              center_pixel = img[i, j]
              binary_string = ""

              for k in range(self.neighbors):
                  neighbor_x = i + int(self.radius * np.cos(2 * np.pi * k / self.neighbors))
                  neighbor_y = j + int(self.radius * np.sin(2 * np.pi * k / self.neighbors))

                  if 0 <= neighbor_x < rows and 0 <= neighbor_y < cols:
                      neighbor_pixel = img[neighbor_x, neighbor_y]
                      if neighbor_pixel >= center_pixel:
                          binary_string += "1"
                      else:
                          binary_string += "0"
                  else:
                      binary_string += "0"

              lbp_img[i, j] = int(binary_string, 2)

      return lbp_img

  def get_histogram(self, lbp_img):
      histogram = np.zeros(256, dtype=np.int32)

      for i in range(lbp_img.shape[0]):
          for j in range(lbp_img.shape[1]):
              histogram[lbp_img[i, j]] += 1

      return histogram

  def get_features(self, histogram):
      features = np.zeros(256, dtype=np.float32)

      for i in range(256):
          features[i] = histogram[i] / (histogram.sum())

      return features

  def get_uniform_pattern(self, lbp_img):
      uniform_pattern = np.zeros(10, dtype=np.int32)

      for i in range(lbp_img.shape[0]):
          for j in range(lbp_img.shape[1]):
              if self.is_uniform(lbp_img[i, j]):
                  uniform_pattern[self.get_uniform_index(lbp_img[i, j])] += 1

      return uniform_pattern

  def is_uniform(self, lbp_value):
      binary_string = bin(lbp_value)[2:].zfill(8)
      transitions = 0

      for i in range(7):
          if binary_string[i] != binary_string[i + 1]:
              transitions += 1

      return transitions <= 2

  def get_uniform_index(self, lbp_value):
      binary_string = bin(lbp_value)[2:].zfill(8)
      transitions = 0

      for i in range(7):
          if binary_string[i] != binary_string[i + 1]:
              transitions += 1

      if transitions == 0:
          return 0
      elif transitions == 1:
          return 1
      elif transitions == 2:
          return 2 + self.get_rotation_invariant(lbp_value)

  def get_rotation_invariant(self, lbp_value):
      binary_string = bin(lbp_value)[2:].zfill(8)
      rotation_invariant = 0

      for i in range(7):
          if binary_string[i] == '1':
              rotation_invariant += 1

      return rotation_invariant

  def get_rotation_invariant_features(self, lbp_img):
      rotation_invariant_features = np.zeros(10, dtype=np.float32)

      for i in range(lbp_img.shape[0]):
          for j in range(lbp_img.shape[1]):
              rotation_invariant_features[self.get_rotation_invariant(lbp_img[i, j])] += 1

      return rotation_invariant_features / (lbp_img.shape[0] * lbp_img.shape[1])

  def get_statistical_features(self, features):
      statistical_features = np.zeros(4, dtype=np.float32)

      statistical_features[0] = np.mean(features) # Mean
      statistical_features[1] = np.std(features) # Standard Deviation

      std = np.std(features)
      if std != 0:
          statistical_features[2] = np.mean((features - np.mean(features)) ** 3) / (std ** 3) # Skewness
          statistical_features[3] = np.mean((features - np.mean(features)) ** 4) / (std ** 4) - 3 # Kurtosis
      else:
          statistical_features[2] = np.nan # Skewness Jadi NaN
          statistical_features[3] = np.nan # Kurtosis jadi NaN

      return statistical_features

# # Contoh penggunaan
# img = np.array([
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12],
#     [13, 14, 15, 16]
# ])

# lbp_calculator = LBP(radius=1, neighbors=8)
# lbp_img = lbp_calculator.calculate_lbp(img)
# histogram = lbp_calculator.get_histogram(lbp_img)
# features = lbp_calculator.get_features(histogram)
# uniform_pattern = lbp_calculator.get_uniform_pattern(lbp_img)
# rotation_invariant_features = lbp_calculator.get_rotation_invariant_features(lbp_img)

# features_statistical = lbp_calculator.get_statistical_features(features)
# uniform_pattern_statistical = lbp_calculator.get_statistical_features(uniform_pattern)
# rotation_invariant_features_statistical = lbp_calculator.get_statistical_features(rotation_invariant_features)

# print(features_statistical)
# print(uniform_pattern_statistical)
# print(rotation_invariant_features_statistical)