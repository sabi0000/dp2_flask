from manim import *

class ConvolutionVisualization(Scene):
    def construct(self):
        # Define the original 5x5 image (represented by colored squares for simplicity)
        original_image = VGroup(*[
            Square(side_length=1, color=WHITE, fill_opacity=1).shift(LEFT*3 + UP*2 + RIGHT*i + DOWN*j)
            for i in range(5) for j in range(5)
        ])

        # Apply different colors to simulate pixel values (for simplicity)
        for i, square in enumerate(original_image):
            if i % 2 == 0:
                square.set_fill(BLUE)
            else:
                square.set_fill(RED)

        # Create a 3x3 filter (kernel) in the form of a smaller grid
        filter_matrix = VGroup(*[
            Square(side_length=1, color=YELLOW, fill_opacity=0.5).shift(LEFT*2 + UP*2 + RIGHT*i + DOWN*j)
            for i in range(3) for j in range(3)
        ])

        # Position the filter over the original image (top-left corner)
        filter_matrix.shift(LEFT*0.5 + UP*0.5)

        # Create a second image for the result of the convolution (filtered image)
        filtered_image = VGroup(*[
            Square(side_length=1, color=WHITE, fill_opacity=0.7).shift(LEFT*3 + UP*2 + RIGHT*i + DOWN*j)
            for i in range(5) for j in range(5)
        ])
        
        # Initialize the filtered image with a neutral color
        for square in filtered_image:
            square.set_fill(GRAY)

        # Add the original image and the filter to the scene
        self.play(Create(original_image))

        # Display the filter matrix on the left side
        self.play(Create(filter_matrix))

        # Show the original image on the left side and the filtered image (initially empty) on the right
        self.play(Create(filtered_image))

        # Move the filter across the original image
        for i in range(3):  # Sliding the filter in rows
            for j in range(3):  # Sliding the filter in columns
                # Move the filter
                self.play(filter_matrix.animate.move_to(LEFT*0.5 + UP*0.5 + RIGHT*i + DOWN*j))
                
                # Simulate applying the filter to this region (we will just change the color to represent the filter's effect)
                # In a real implementation, this would compute the convolution result
                
                # Show the updated result on the filtered image (for simplicity, we change colors)
                new_color = BLUE if (i + j) % 2 == 0 else RED
                self.play(filtered_image[(i + 1) * 5 + (j + 1)].animate.set_fill(new_color))

                self.wait(0.1)

        # Show final filtered image on the right side after convolution is complete
        self.play(
            filtered_image.animate.set_fill(BLUE),
            run_time=2
        )

        self.wait(2)
