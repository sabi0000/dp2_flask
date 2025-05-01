import cv2
import numpy as np
import os

image_path = os.getenv('IMAGE_PATH')
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
 # Convert the image to grayscale
ray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

cv2.imwrite(blurred_image_path, blurred_image)


lower_threshold = 50
upper_threshold = 150
canny_filtered = cv2.Canny(blurred_image[start_y:end_y, start_x:end_x], lower_threshold, upper_threshold)

