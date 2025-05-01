from manim import *
import numpy as np

def softmax_3(x0):
    x = np.array([x0, 1, 0])  # fixujeme 3-prvkový vektor
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class SoftmaxActivation(Scene):
    def construct(self):
        neuron_radius = 0.5

        # Vytvorenie neurónu
        neuron = Circle(radius=neuron_radius, color=WHITE)
        neuron.move_to(ORIGIN + DOWN * 1.5)  # posunuté o trochu nižšie
        self.add(neuron)

        # Text pre softmax funkciu
        softmax_function_text = MathTex(
            r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}",
            font_size=44
        )
        softmax_function_text.next_to(neuron, DOWN * 1)
        self.add(softmax_function_text)

        # Vytvorenie väčších osí pre softmax priebeh
        ax = Axes(
            x_range=[-3, 5, 1],
            y_range=[0, 1.1, 0.2],
            x_length=8,  # zväčšená dĺžka osi x
            y_length=4,  # zväčšená dĺžka osi y
            axis_config={"color": WHITE}
        ).scale(1).to_edge(UP)  # zväčšený graf a posunutý vyššie

        ax_labels = ax.get_axis_labels(x_label="x_0", y_label="softmax(x_0)")
        x_numbers = ax.get_x_axis().add_numbers()
        y_numbers = ax.get_y_axis().add_numbers()

        self.add(ax, ax_labels, x_numbers, y_numbers)

        # Softmax krivka len pre prvý prvok
        softmax_curve = ax.plot(lambda x: softmax_3(x)[0], color=BLUE)
        self.add(softmax_curve)

        # Bod na krivke
        point = Dot(ax.coords_to_point(-3, softmax_3(-3)[0]), color=YELLOW)
        self.add(point)

        # Animácia bodu a zmena farby neurónu
        num_steps = 100
        for i in range(num_steps + 1):
            x_val = -3 + (8 / num_steps) * i  # od -3 po 5
            y_val = softmax_3(x_val)[0]
            self.play(
                point.animate.move_to(ax.coords_to_point(x_val, y_val)),
                run_time=0.03,
                rate_func=smooth
            )

            neuron_color = interpolate_color(BLUE, YELLOW, y_val)
            self.play(
                neuron.animate.set_fill(neuron_color, opacity=y_val),
                run_time=0.05
            )

        self.wait(2)
