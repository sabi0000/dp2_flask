import cv2
import numpy as np
import os
from scipy.stats import entropy

image_path = 'static/processed_image.png'
image = cv2.imread(image_path)

lower_threshold = int(os.getenv('lower_threshold'))
upper_threshold = int(os.getenv('upper_threshold'))

# Definovanie prahových hodnôt pre Canny edge detection
low_threshold = lower_threshold
high_threshold = upper_threshold

# Aplikácia Canny edge detection na rozmazaný obrázok
edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
num_edges = np.sum(edges > 0)
print(f"Number of edges: {num_edges}")

laplacian_var = cv2.Laplacian(edges, cv2.CV_64F).var()
print(f"Laplacian variance: {laplacian_var}")

edge_density = num_edges / (image.shape[0] * image.shape[1])
print(f"Edge density: {edge_density}")

histogram, _ = np.histogram(edges.flatten(), bins=256, range=[0, 256])
edge_entropy = entropy(histogram)
print(f"Edge entropy: {edge_entropy}")
