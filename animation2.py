import cv2
import numpy as np
import os
from manim import *

class SobelFilterAnimation(Scene):
    def construct(self):
        gauss_size =int(os.getenv('GAUSS_SIZE'))
        sigma_size=int(os.getenv('SIGMA_SIZE'))
        # Načítanie obrázka
        # Retrieve the image path from the environment variable
        image_path = os.getenv('IMAGE_PATH')
        if not image_path:
            raise FileNotFoundError("No image path set in the environment variable.")

        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (gauss_size, gauss_size), sigma_size)

        blurred_image_path='blured_image.png'
        cv2.imwrite(blurred_image_path, blurred_image)


        # Create the original and processed (Sobel-filtered) images for Manim
        original_image_manim = ImageMobject(blurred_image_path)
        original_image_manim.set_width(6)  # Set new width
        original_image_manim.set_height(4.5)  # Set new height
        original_image_manim.shift(LEFT * 4)

        # Initially use the same image for the processed one (will update it progressively)
        sobel_image_manim = ImageMobject(blurred_image_path)
        sobel_image_manim.set_width(6)  # Set new width
        sobel_image_manim.set_height(4.5)  # Set new height
        sobel_image_manim.shift(RIGHT * 4)

        self.clear()

        # Show the original and processed images side by side
        self.play(FadeIn(original_image_manim), FadeIn(sobel_image_manim))

        # Create a visual representation of the Sobel kernel
        kernel_visual = Square(side_length=1.5, color=YELLOW)
        kernel_visual.move_to((sobel_image_manim.get_corner(UL)) + RIGHT * 0.4 + DOWN * 0.4)
        self.play(FadeIn(kernel_visual))

        # Define step sizes for kernel movement
        step_x = 1.5
        step_y = 1.5

        # Image dimensions in terms of the Manim visual object
        img_width = 6
        img_height = 4.5

        for j in range(int(img_height // step_y)):
            for i in range(int(img_width // step_x)):
                # Alternate kernel movement
                if j % 2 == 0:  # Even row: left to right
                    self.move_kernel_and_apply_sobel(kernel_visual, sobel_image_manim, blurred_image, i, j, step_x, step_y)
                else:  # Odd row: right to left
                    self.move_kernel_and_apply_sobel(kernel_visual, sobel_image_manim, blurred_image, int(img_width // step_x) - 1 - i, j, step_x, step_y)

        self.wait(2)

    def move_kernel_and_apply_sobel(self, kernel_visual, sobel_image_manim, blurred_image, i, j, step_x, step_y):
        ksize = int(os.getenv('KERNEL_SIZE'))
        """ Move the kernel and apply Sobel filter to the corresponding area """
        # Calculate the new position for the kernel in image coordinates
        new_x = i * step_x
        new_y = j * step_y

        # Animate kernel movement
        self.play(kernel_visual.animate.move_to((sobel_image_manim.get_corner(UL) + RIGHT * 0.75 + DOWN * 0.75) + RIGHT * new_x + DOWN * new_y), run_time=0.5)

        # Define the size of the kernel in pixels based on the Manim image size
        kernel_size_in_image = int(1.5 * 100)

        # Apply Sobel filter only inside the kernel's region
        start_x = int(new_x * 100)  # Convert to pixel coordinates
        start_y = int(new_y * 100)
        end_x = start_x + kernel_size_in_image
        end_y = start_y + kernel_size_in_image

        # Ensure the region is within bounds
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(blurred_image.shape[1], end_x)
        end_y = min(blurred_image.shape[0], end_y)

        # Create a mask for the region inside the kernel
        mask = np.zeros((end_y - start_y, end_x - start_x), dtype=np.uint8)
        mask[:, :] = 1  # Fill the mask with ones

        # Apply Sobel filter within the kernel's region only
        sobel_x = cv2.Sobel(blurred_image[start_y:end_y, start_x:end_x], cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(blurred_image[start_y:end_y, start_x:end_x], cv2.CV_64F, 0, 1, ksize=ksize)
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)
        sobel_normalized = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
        sobel_filtered = np.uint8(sobel_normalized)

        # Update the blurred image only in the kernel region
        blurred_image[start_y:end_y, start_x:end_x] = np.where(mask == 1, sobel_filtered, blurred_image[start_y:end_y, start_x:end_x])

  
        # Save the progressively filtered image
        sobel_image_path='sobel.png'
        cv2.imwrite(sobel_image_path, blurred_image)

        # Update the right-side image with the progressively filtered region
        sobel_image_updated = ImageMobject(sobel_image_path)
        sobel_image_updated.set_width(6).shift(RIGHT * 4)

        # Show the filtered image transition
        self.play(Transform(sobel_image_manim, sobel_image_updated), run_time=0.5)