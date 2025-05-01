from manim import *
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetworkActivation(Scene):
    def construct(self):
        neuron_radius = 0.5
        
        
        # Vytvorenie neurónu
        neuron = Circle(radius=neuron_radius, color=WHITE)
        neuron.move_to(ORIGIN+DOWN*1.5)
        self.add(neuron)
        
        # Text pre sigmoid funkciu
        sigmoid_function_text = MathTex(r"\sigma(x) = \frac{1}{1 + e^{-x}}", font_size=44)
        sigmoid_function_text.next_to(neuron, DOWN * 3)  # Posunúť pod neurón
        self.add(sigmoid_function_text)
        
        
       # Graf sigmoid funkcie s číselnými hodnotami na osiach
        ax = Axes(
            x_range=[-3, 3, 1],  # tretia hodnota určuje krok (tick) pre os x
            y_range=[-0.2, 1.2, 0.2],  # krok pre os y
            axis_config={"color": WHITE},
        ).scale(0.7).to_edge(UP)

        # Osy a popisky
        ax_labels = ax.get_axis_labels(x_label='x', y_label='\\sigma(x)')

        # Číselné značky (ticks)
        x_numbers = ax.get_x_axis().add_numbers()
        y_numbers = ax.get_y_axis().add_numbers()

        self.add(ax, ax_labels, x_numbers, y_numbers)
        
        sigmoid_curve = ax.plot(lambda x: sigmoid(x), color=BLUE)
        self.add(sigmoid_curve)
        
        # Zvýraznenie bodu na sigmoid krivke
        point = Dot(ax.coords_to_point(-3, sigmoid(-3)), color=YELLOW)
        self.add(point)

        # Animácia bodu pohybujúceho sa po sigmoid krivke
        num_steps = 100  # Počet krokov pre animáciu
        for i in range(num_steps + 1):
            x = -3 + (6 / num_steps) * i  # Rozsah x od -3 do 3
            y = sigmoid(x)
            self.play(
                point.animate.move_to(ax.coords_to_point(x, y)),
                run_time=0.05,  # Doba trvania pre každý krok
                rate_func=smooth
            )
            # Zmena farby neurónu na základe aktuálnej hodnoty sigmoid(x)
            neuron_color = interpolate_color(BLUE, YELLOW, y)  # y je hodnota sigmoid(x)
            self.play(
                neuron.animate.set_fill(color=neuron_color, opacity=y),
                run_time=0.1  # Doba trvania pre zmenu farby
            )
        



        
        self.wait(2)