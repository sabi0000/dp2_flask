from manim import *
import numpy as np

def relu(x):
    return np.maximum(0, x)

class ReLUActivation(Scene):
    def construct(self):
        neuron_radius = 0.5
        
        # Vytvorenie neurónu
        neuron = Circle(radius=neuron_radius, color=WHITE)
        neuron.move_to(ORIGIN + DOWN * 1.5)
        self.add(neuron)

        # Text pre ReLU funkciu
        relu_function_text = MathTex(r"\text{ReLU}(x) = \max(0, x)", font_size=44)
        relu_function_text.next_to(neuron, DOWN * 3)
        self.add(relu_function_text)

        # Graf ReLU funkcie s číselnými hodnotami na osiach
        ax = Axes(
            x_range=[-3, 3, 1],
            y_range=[-0.2, 3.2, 1],
            axis_config={"color": WHITE},
        ).scale(0.7).to_edge(UP)

        ax_labels = ax.get_axis_labels(x_label='x', y_label='\\text{ReLU}(x)')

        # Číselné značky
        x_numbers = ax.get_x_axis().add_numbers()
        y_numbers = ax.get_y_axis().add_numbers()

        self.add(ax, ax_labels, x_numbers, y_numbers)

        relu_curve = ax.plot(lambda x: relu(x), color=GREEN)
        self.add(relu_curve)

        # Zvýraznenie bodu na ReLU krivke
        point = Dot(ax.coords_to_point(-3, relu(-3)), color=YELLOW)
        self.add(point)

        # Animácia bodu pohybujúceho sa po ReLU krivke
        num_steps = 100
        for i in range(num_steps + 1):
            x = -3 + (6 / num_steps) * i
            y = relu(x)
            self.play(
                point.animate.move_to(ax.coords_to_point(x, y)),
                run_time=0.05,
                rate_func=smooth
            )
            # Zmena farby neurónu na základe aktuálnej hodnoty ReLU(x)
            # Pre ReLU môže ísť y od 0 do 3 — normalizuj do intervalu 0–1
            normalized_y = y / 3 if y <= 3 else 1
            neuron_color = interpolate_color(GREY, GREEN, normalized_y)
            self.play(
                neuron.animate.set_fill(color=neuron_color, opacity=normalized_y),
                run_time=0.1
            )

        self.wait(2)
