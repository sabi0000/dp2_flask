import cv2
import numpy as np
import random
import os
import sys
from manim import *

config.pixel_width = 1080
config.pixel_height = 1920
config.frame_rate = 30

class ImageDecompositionAnimationUser (Scene):

    def construct(self):
        
        image_path = os.getenv('IMAGE_PATH')
        if not image_path:
            raise FileNotFoundError("No image path set in the environment variable.")
        
        # Read the image
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if original_image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        original_image_manim = ImageMobject(image_path).scale(1.8)
        original_image_manim.move_to(UP * 8)
        self.clear()
        self.play(FadeIn(original_image_manim))

        height, width, _ = original_image.shape

        tile_configs = [
            {"tile_px": 100, "tile_size": 1.5, "y_offset": -2.5, "max_per_row": 10},
            {"tile_px": 75,  "tile_size": 1.0, "y_offset": 2.8,  "max_per_row": 20},
            {"tile_px": 50,  "tile_size": 0.5, "y_offset": 6.8,  "max_per_row": 25},
        ]
        output_dir = "pictures/tiles"
        os.makedirs(output_dir, exist_ok=True)

        for config_item in tile_configs:
            tile_px = config_item["tile_px"]
            tile_size = config_item["tile_size"]
            y_offset = config_item["y_offset"]
            max_per_row = config_item["max_per_row"]

            spacing = 0.1
            step_x = tile_size + spacing
            step_y = tile_size + spacing

            num_tiles_x = width // tile_px
            num_tiles_y = height // tile_px

            tile_objects = []
            positions = []

            index = 0
            for i in range(num_tiles_y):
                for j in range(num_tiles_x):
                    sx = j * tile_px
                    sy = i * tile_px
                    ex = sx + tile_px
                    ey = sy + tile_px

                    tile = original_image[sy:ey, sx:ex]
                    tile_path = os.path.join(output_dir, f"tile_{tile_px}_{i}_{j}.png")
                    cv2.imwrite(tile_path, tile)

                    tile_manim = ImageMobject(tile_path)
                    tile_manim.set_width(tile_size)
                    tile_manim.set_height(tile_size)
                    tile_objects.append(tile_manim)

                    row = index // max_per_row
                    col = index % max_per_row
                    pos = (
                        DOWN * (y_offset + row * step_y)
                        + RIGHT * (col * step_x - (max_per_row * step_x) / 2)
                    )
                    positions.append(pos)
                    index += 1

            # Shuffle positions and tile objects
            random.shuffle(tile_objects)
            random.shuffle(positions)

            for tile_manim, pos in zip(tile_objects, positions):
                tile_manim.move_to(pos)
                self.play(FadeIn(tile_manim), run_time=0.05)

        self.wait(2)