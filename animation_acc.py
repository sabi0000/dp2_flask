import numpy as np
from manim import *
import os

class AccuracyLossGraph(Scene):
    def construct(self):
        hidden_layers = int(os.getenv("LAYERS_M", 2))  # Default 2 layers
        neurons_per_layer = int(os.getenv("NEURONS_M", 4))  # Default 4 neurons
        epochs = int(os.getenv("EPOCHS_M", 4))  # Zmena na 50 epoch
        
        # Sigmoidová funkcia ako aktivačná funkcia
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        
        # Simulácia presnosti a straty na základe architektúry siete
        x_values = np.arange(1, epochs + 1)
        accuracy = 0.5 + 0.4 * sigmoid((x_values - 5) / 5) * (hidden_layers / 6) * (neurons_per_layer / 8)
        loss = np.clip(1.5 * np.exp(-x_values / (5 + hidden_layers)) + 0.1 * np.random.rand(epochs), 0, 2)  # Obmedzené na max. 2
        
        ax = Axes(
            x_range=[1, epochs, 2],
            y_range=[0, 2, 0.2],
            axis_config={"color": WHITE},
        ).add_coordinates()
        
        labels = ax.get_axis_labels(x_label="Epochs", y_label="Value")
        accuracy_graph = ax.plot_line_graph(x_values, accuracy, line_color=BLUE, add_vertex_dots=True, vertex_dot_radius=0.05)
        loss_graph = ax.plot_line_graph(x_values, loss, line_color=RED, add_vertex_dots=True, vertex_dot_radius=0.05)
        
        accuracy_label = Text("Accuracy", color=BLUE).next_to(ax, UP * 1).shift(LEFT * 1.5)
        loss_label = Text("Loss", color=RED).next_to(accuracy_label, RIGHT * 0.1)
        
        self.play(Create(ax), Write(labels))
        self.wait(1)
        self.play(Create(accuracy_graph), Write(accuracy_label))
        self.wait(1)
        self.play(Create(loss_graph), Write(loss_label))
        self.wait(2)

