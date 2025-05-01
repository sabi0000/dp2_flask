from manim import *
import numpy as np

def tanh(x):
    return np.tanh(x)

class TanhActivation(Scene):
    def construct(self):
        neuron_radius = 0.5

        # Vytvorenie neurónu
        neuron = Circle(radius=neuron_radius, color=WHITE)
        neuron.move_to(ORIGIN + DOWN * 1.5)
        self.add(neuron)

        # Text pre tanh funkciu
        tanh_function_text = MathTex(r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}", font_size=44)
        tanh_function_text.next_to(neuron, DOWN * 3)
        self.add(tanh_function_text)

        # Graf tanh funkcie s číselnými hodnotami na osiach
        ax = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1.2, 1.2, 0.2],
            axis_config={"color": WHITE},
        ).scale(0.7).to_edge(UP)

        # Osy a popisky
        ax_labels = ax.get_axis_labels(x_label='x', y_label='\\tanh(x)')

        # Číselné značky
        x_numbers = ax.get_x_axis().add_numbers()
        y_numbers = ax.get_y_axis().add_numbers()

        self.add(ax, ax_labels, x_numbers, y_numbers)

        tanh_curve = ax.plot(lambda x: tanh(x), color=GREEN)
        self.add(tanh_curve)

        # Zvýraznenie bodu na tanh krivke
        point = Dot(ax.coords_to_point(-3, tanh(-3)), color=YELLOW)
        self.add(point)

        # Animácia bodu pohybujúceho sa po tanh krivke
        num_steps = 100
        for i in range(num_steps + 1):
            x = -3 + (6 / num_steps) * i
            y = tanh(x)
            self.play(
                point.animate.move_to(ax.coords_to_point(x, y)),
                run_time=0.05,
                rate_func=smooth
            )
            # Farba neurónu podľa hodnoty tanh(x) upravená z intervalu [-1, 1] na [0, 1]
            y_normalized = (y + 1) / 2
            neuron_color = interpolate_color(BLUE, RED, y_normalized)
            self.play(
                neuron.animate.set_fill(color=neuron_color, opacity=abs(y_normalized)),
                run_time=0.1
            )

        self.wait(2)
