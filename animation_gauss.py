import cv2
import numpy as np
import os
from manim import *

class GaussAnimation(Scene):
    def construct(self):
        image_path = os.getenv('IMAGE_PATH')
        if not image_path:
            raise FileNotFoundError("No image path set in the environment variable.")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

       
        gray_image = image

        cv2.imwrite(image_path, gray_image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Create the original and processed (Sobel-filtered) images for Manim
        original_image_manim = ImageMobject(image_rgb)
        original_image_manim.set_width(6)  # Set new width
        original_image_manim.set_height(4.5)  # Set new height
        original_image_manim.shift(LEFT * 4)

        # Initially use the same image for the processed one (will update it progressively)
        gauss_image_manim = ImageMobject(image_path)
        gauss_image_manim.set_width(6)  # Set new width
        gauss_image_manim.set_height(4.5)  # Set new height
        gauss_image_manim.shift(RIGHT * 4)

        self.clear()

        # Show the original and processed images side by side
        self.play(FadeIn(original_image_manim), FadeIn(gauss_image_manim))

        # Create a visual representation of the Sobel kernel
        kernel_visual = Square(side_length=1.5, color=YELLOW)
        kernel_visual.move_to((gauss_image_manim.get_corner(UL)) + RIGHT * 0.4 + DOWN * 0.4)
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
                    self.move_kernel_and_apply_sobel(kernel_visual, gauss_image_manim, gray_image, i, j, step_x, step_y)
                else:  # Odd row: right to left
                    self.move_kernel_and_apply_sobel(kernel_visual, gauss_image_manim, gray_image, int(img_width // step_x) - 1 - i, j, step_x, step_y)

        self.wait(2)

    def move_kernel_and_apply_sobel(self, kernel_visual, gauss_image_manim, gray_image, i, j, step_x, step_y,):
        gauss_size = int(os.getenv('GAUSS_SIZE'))
        sigma_size = int(os.getenv('SIGMA_SIZE'))

        # Calculate the new position for the kernel in image coordinates
        new_x = i * step_x
        new_y = j * step_y

        # Animate kernel movement
        self.play(kernel_visual.animate.move_to((gauss_image_manim.get_corner(UL) + RIGHT * 0.75 + DOWN * 0.75) + RIGHT * new_x + DOWN * new_y), run_time=0.5)

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
        end_x = min(gray_image.shape[1], end_x)
        end_y = min(gray_image.shape[0], end_y)

        # Create a mask for the region inside the kernel
        mask = np.zeros((end_y - start_y, end_x - start_x), dtype=np.uint8)
        mask[:, :] = 1  # Fill the mask with ones

        # Aplikuj Gaussov filter na farebný obrázok
        gauss_updated = cv2.GaussianBlur(gray_image, (gauss_size, gauss_size), sigma_size)

        # Extrahuj relevantné časti
        gauss_updated_region = gauss_updated[start_y:end_y, start_x:end_x]
        original_region = gray_image[start_y:end_y, start_x:end_x]

        # Vytvor masku 3-kanálovú (farebnú), aby sme mohli kombinovať
        mask_3ch = np.stack([mask]*3, axis=-1)

        # Aplikuj rozmazanie iba na maskované časti (vo farebnom priestore)
        gray_image[start_y:end_y, start_x:end_x] = np.where(mask_3ch == 1, gauss_updated_region, original_region)

        # Save the progressively filtered image
        gauss_updated_image_path = 'gauss_updated_image.png'
        cv2.imwrite(gauss_updated_image_path, gray_image)

        # Update the right-side image with the progressively filtered region
        canny_image_updated = ImageMobject(gauss_updated_image_path)
        canny_image_updated.set_width(6).shift(RIGHT * 4)

        # Show the filtered image transition
        self.play(Transform(gauss_image_manim, canny_image_updated), run_time=0.5)
