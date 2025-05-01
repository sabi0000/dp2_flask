import numpy as np
from manim import *
import os

class ValidationAccuracyLossGraph(Scene):
    def construct(self):
        hidden_layers = int(os.getenv("LAYERS_M", 2))  # Default 2 layers
        neurons_per_layer = int(os.getenv("NEURONS_M", 4))  # Default 4 neurons
        epochs = int(os.getenv("EPOCHS_M", 4))  # Zmena na 50 epoch
        
        # Sigmoidová funkcia ako aktivačná funkcia
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        
        # Dynamický prah na základe počtu neurónov a vrstiev
        overfitting_threshold = 10 + (neurons_per_layer * hidden_layers) / 10  # Dynamicky nastaviteľný prah
        
        # Zaokrúhlenie na celé číslo pre použitie ako index
        overfitting_threshold = int(np.ceil(overfitting_threshold))
        
        # Simulácia validácie s overfittingom
        x_values = np.arange(1, epochs + 1)
        validation_accuracy = 0.5 + 0.3 * sigmoid((x_values - 5) / 5) * (hidden_layers / 6) * (neurons_per_layer / 8)
        validation_accuracy[overfitting_threshold:] -= 0.1 * (x_values[overfitting_threshold:] - overfitting_threshold) / 10  # Simulácia poklesu presnosti
        validation_loss = np.clip(1.2 * np.exp(-x_values / (6 + hidden_layers)) + 0.1 * np.random.rand(epochs), 0, 2)
        validation_loss[overfitting_threshold:] += 0.1 * (x_values[overfitting_threshold:] - overfitting_threshold) / 10  # Simulácia zvýšenia straty
        
        ax = Axes(
            x_range=[1, epochs, 2],
            y_range=[0, 2, 0.2],
            axis_config={"color": WHITE},
        ).add_coordinates()
        
        labels = ax.get_axis_labels(x_label="Epochs", y_label="Value")
        accuracy_graph = ax.plot_line_graph(x_values, validation_accuracy, line_color=GREEN, add_vertex_dots=True, vertex_dot_radius=0.05)
        loss_graph = ax.plot_line_graph(x_values, validation_loss, line_color=ORANGE, add_vertex_dots=True, vertex_dot_radius=0.05)
        
        accuracy_label = Text("Validation Accuracy", color=GREEN).next_to(ax, UP * 1).shift(LEFT * 1.5)
        loss_label = Text("Validation Loss", color=ORANGE).next_to(accuracy_label , RIGHT * 0.1)
        
        warning_text = Text("Overfitting Detected!", color=YELLOW).scale(1.2).next_to(ax, DOWN * 2)
        
        self.play(Create(ax), Write(labels))
        self.wait(1)
        self.play(Create(accuracy_graph), Write(accuracy_label))
        self.wait(1)
        self.play(Create(loss_graph), Write(loss_label))
        self.wait(2)
        
        # Zobrazí varovanie podľa dynamického prahu
        if overfitting_threshold < epochs:
            self.play(Write(warning_text), run_time=2)
        self.wait(2)