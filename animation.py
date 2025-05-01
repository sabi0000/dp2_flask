import cv2
import numpy as np
import os
from manim import *

class BinaryFilterAnimation(Scene):
    def construct(self):
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
        blurred_image = cv2.GaussianBlur(gray_image, (45, 45), 0)

        # Apply Sobel filter to detect edges
        sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=1)
        sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=1)
        sobel_combined = cv2.magnitude(sobelx, sobely)

        # Normalize Sobel result to 0-255
        sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the Sobel result to a binary image
        _, binary_edges = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)

        # Save the binary image (use a valid path for Windows)
        binary_image_path = 'binary_image.png'
        cv2.imwrite(binary_image_path, binary_edges)

        # Load the saved binary image into Manim
        original_image_manim = ImageMobject(binary_image_path)
        original_image_manim.set_width(6)  # Ensure width is sufficient
        original_image_manim.set_height(4.5)
        original_image_manim.move_to(LEFT * 4)  # Adjust the position further left

        # Use the same image for the processed one initially
        contours_image_manim = ImageMobject(binary_image_path)
        contours_image_manim.set_width(6)
        contours_image_manim.set_height(4.5)
        contours_image_manim.move_to(RIGHT * 4)  # Position on the right

        self.clear()

        # Show the original and processed images side by side
        self.play(FadeIn(original_image_manim), FadeIn(contours_image_manim))

        # Create a visual representation of the Sobel kernel
        kernel_visual = Square(side_length=1.55, color=YELLOW)
        kernel_visual.move_to((contours_image_manim.get_corner(UL)) + RIGHT * 0.4 + DOWN * 0.4)
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
                    self.move_kernel_and_apply_sobel(kernel_visual, contours_image_manim, binary_edges, i, j, step_x, step_y)
                else:  # Odd row: right to left
                    self.move_kernel_and_apply_sobel(kernel_visual, contours_image_manim, binary_edges, int(img_width // step_x) - 1 - i, j, step_x, step_y)

        self.wait(2)

    def move_kernel_and_apply_sobel(self, kernel_visual, contours_image_manim, binary_edges, i, j, step_x, step_y):
        """ Move the kernel and apply Sobel filter to the corresponding area """
        # Calculate the new position for the kernel in image coordinates
        new_x = i * step_x
        new_y = j * step_y

        # Animate kernel movement
        self.play(kernel_visual.animate.move_to((contours_image_manim.get_corner(UL) + RIGHT * 0.75 + DOWN * 0.75) + RIGHT * new_x + DOWN * new_y), run_time=0.5)

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
        end_x = min(binary_edges.shape[1], end_x)
        end_y = min(binary_edges.shape[0], end_y)

        # Create a mask for the region inside the kernel
        mask = np.zeros_like(binary_edges, dtype=np.uint8)  # Ensure mask is the same size as binary_edges
        mask[start_y:end_y, start_x:end_x] = 1  # Fill the mask only for the kernel region

        # Find contours on the binary edges image
        contours, _ = cv2.findContours(binary_edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank image to draw the contours
        contour_image = np.zeros_like(binary_edges)

        # Draw the contours on the blank image
        cv2.drawContours(contour_image, contours, -1, (255), 1)  # Draw with white color (255) and thickness 1

        # Combine the original binary edges and the new contours in the masked region
        binary_edges[mask == 1] = contour_image[mask == 1]

        # Save the progressively updated image (use a valid path for Windows)
        progressive_contours_path = 'contours.png'
        cv2.imwrite(progressive_contours_path, binary_edges)

        # Update the right-side image with the progressively filtered region
        binary_image_updated = ImageMobject(progressive_contours_path)
        binary_image_updated.set_width(6).shift(RIGHT * 4)

        # Show the filtered image transition
        self.play(Transform(contours_image_manim, binary_image_updated), run_time=0.5)
