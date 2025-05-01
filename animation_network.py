from manim import *

class NeuralNetworkLearning(Scene):
    def construct(self):
        # Define the number of neurons in each layer
        input_layer_size = 3
        hidden_layer_size = 4
        output_layer_size = 1
        
        # Create layers of neurons
        input_layer = VGroup(*[Circle(radius=0.3, color=BLUE).set_fill(BLUE, opacity=0.6) for _ in range(input_layer_size)]).arrange(DOWN, buff=0.5)
        hidden_layer = VGroup(*[Circle(radius=0.3, color=GREEN).set_fill(GREEN, opacity=0.6) for _ in range(hidden_layer_size)]).arrange(DOWN, buff=0.5)
        output_layer = VGroup(*[Circle(radius=0.3, color=RED).set_fill(RED, opacity=0.6) for _ in range(output_layer_size)]).arrange(DOWN, buff=0.5)
        
        # Position layers
        input_layer.move_to(LEFT * 4)
        hidden_layer.move_to(ORIGIN)
        output_layer.move_to(RIGHT * 4)
        
        # Create synaptic connections (edges) with initial weights
        connections = VGroup()
        for input_neuron in input_layer:
            for hidden_neuron in hidden_layer:
                weight = np.random.uniform(-1, 1)
                color = GREEN if weight > 0 else RED
                thickness = abs(weight) * 4
                connections.add(Line(input_neuron.get_center(), hidden_neuron.get_center(), color=color, stroke_width=thickness))
        
        for hidden_neuron in hidden_layer:
            for output_neuron in output_layer:
                weight = np.random.uniform(-1, 1)
                color = GREEN if weight > 0 else RED
                thickness = abs(weight) * 4
                connections.add(Line(hidden_neuron.get_center(), output_neuron.get_center(), color=color, stroke_width=thickness))
        
        # Create activity levels as labels above each neuron
        activity_labels = VGroup()
        for neuron_group in [input_layer, hidden_layer, output_layer]:
            for neuron in neuron_group:
                activity_value = DecimalNumber(0, num_decimal_places=2).next_to(neuron, UP)
                activity_labels.add(activity_value)

        # Add headline
        headline = Text("Tréning neurónovej siete – Epochy", font_size=36).to_edge(UP)
        self.play(Write(headline))

        # Epoch counter
        epoch_text = Text("Epocha 0", font_size=28).next_to(headline, DOWN*18, buff=0.3)
        self.play(Write(epoch_text))
        
        # Add neurons, connections, and activity labels to the scene
        self.play(FadeIn(input_layer), FadeIn(hidden_layer), FadeIn(output_layer))
        self.play(Create(connections), FadeIn(activity_labels))
        
        # Simulate learning with changes in synaptic weights and activity levels
        for epoch in range(1, 4):  # Loop for a few "training iterations"
            # Update epoch text
            new_epoch_text = Text(f"Epocha {epoch}", font_size=28).next_to(headline, DOWN*18, buff=0.3)
            self.play(Transform(epoch_text, new_epoch_text))

            # Update activity levels with random values
            new_activity_values = [np.random.uniform(0, 1) for _ in activity_labels]
            self.play(*[activity_labels[i].animate.set_value(new_activity_values[i]) for i in range(len(activity_labels))])

            # Adjust weights
            new_connections = VGroup()
            for line in connections:
                new_weight = np.random.uniform(-1, 1)
                new_color = GREEN if new_weight > 0 else RED
                new_thickness = abs(new_weight) * 4
                new_line = Line(line.get_start(), line.get_end(), color=new_color, stroke_width=new_thickness)
                new_connections.add(new_line)
            
            # "Thinking" animation
            for neuron in hidden_layer:
                self.play(neuron.animate.set_fill(YELLOW, opacity=0.8), run_time=0.2)
                self.play(neuron.animate.set_fill(GREEN, opacity=0.6), run_time=0.2)
            
            # Update synaptic weights
            self.play(ReplacementTransform(connections, new_connections))
            connections = new_connections  # Update reference
        
        self.wait()
