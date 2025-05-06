import cv2
import numpy as np
import os
from manim import *

class NeuralNetwork(Scene):
    def construct(self):
        #hidden_layers = 2  # Užívateľ si môže vybrať hodnotu od 1 po 6
        #neurons_per_layer = 2  # Užívateľ si môže vybrať hodnotu od 2 po 8
        hidden_layers=int(os.getenv(('LAYERS')))
        neurons_per_layer=int(os.getenv(('NEURONS')))
        layers = [3] + [neurons_per_layer] * hidden_layers + [2]  # Automatické generovanie vrstiev

        neuron_radius = 0.3
        spacing = 1.2  
        layer_distance = 1.5  
        
        total_width = (len(layers) - 1) * layer_distance
        x_offset = -total_width / 2
        
        neurons = []
        activations = {} 
        
        for i, num_neurons in enumerate(layers):
            layer = []
            for j in range(num_neurons):
                neuron = Circle(radius=neuron_radius, color=WHITE)
                neuron.move_to(RIGHT * (i * layer_distance + x_offset) + UP * (j - num_neurons / 2) * spacing)
                layer.append(neuron)
                self.add(neuron)
                activations[neuron] = np.random.rand()  # Náhodná aktivácia
                
            neurons.append(layer)
        
        connections = []
        for i in range(len(neurons) - 1):
            for neuron1 in neurons[i]:
                for neuron2 in neurons[i + 1]:
                    connection = Line(neuron1.get_center(), neuron2.get_center(), color=GRAY)
                    connections.append(connection)
                    self.add(connection)
        
        self.wait(1)

        for i, layer in enumerate(neurons[:-1]):  # Neaktivujeme výstupné neuróny v tejto časti
            for neuron in layer:
                activation_value = activations[neuron]  
                neuron_color = interpolate_color(WHITE, YELLOW, activation_value)
                self.play(neuron.animate.set_fill(neuron_color, opacity=activation_value), run_time=0.3)
            self.wait(0.5)

            for connection in connections:
                if any(np.allclose(connection.get_start(), n.get_center()) for n in layer):
                    self.play(connection.animate.set_color(BLUE), run_time=0.2)

        # 🔥 Dynamická farba výstupných neurónov podľa počtu skrytých vrstiev a neurónov
        max_layers = 6
        max_neurons = 8
        intensity_factor = hidden_layers / max_layers  # Normalizácia od 0 do 1
        neuron_factor = neurons_per_layer / max_neurons
        output_values = [0.2 + 0.6 * intensity_factor * neuron_factor, 0.3 + 0.5 * intensity_factor * neuron_factor]  # Hodnoty pre opacity

        colors = [interpolate_color(WHITE, RED, value) for value in output_values]
        for neuron, color in zip(neurons[-1], colors):
            self.play(neuron.animate.set_fill(color, opacity=output_values[neurons[-1].index(neuron)]), run_time=0.5)
        
        output_text = Text(f"Výstup: Trieda {output_values.index(max(output_values))}", font_size=24)
        output_text.next_to(neurons[-1][-1], RIGHT)
        output_text.shift(RIGHT * -0.5 + UP * 2)
        self.add(output_text)

        # Simulácia overfittingu pri vyššom počte vrstiev a neurónov
        if len(layers) >= 5 or neurons_per_layer >= 7:
            warning_text = Text("Pozor! Overfitting", font_size=28, color=RED)
            warning_text.move_to(UP * 3)
            self.add(warning_text)
            for _ in range(5):
                self.play(warning_text.animate.set_opacity(0), run_time=0.2)
                self.play(warning_text.animate.set_opacity(1), run_time=0.2)
        
        self.wait(2)
