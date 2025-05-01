from manim import *

class NeuralNetwork(Scene):
    def construct(self):
        # Definícia vrstiev neurónov
        layers = [3, 5, 4, 2]  # Pridaná jedna skrytá vrstva so 4 neurónmi
        neuron_radius = 0.3
        spacing = 1.2  # Vzdialenosť medzi neurónmi
        layer_distance = 2.5  # Vzdialenosť medzi vrstvami
        
        # Posunutie celej siete na stred
        total_width = (len(layers) - 1) * layer_distance
        x_offset = -total_width / 2
        
        # Vytvorenie neurónov
        neurons = []
        for i, num_neurons in enumerate(layers):
            layer = []
            for j in range(num_neurons):
                neuron = Circle(radius=neuron_radius, color=WHITE)
                neuron.move_to(RIGHT * (i * layer_distance + x_offset) + UP * (j - num_neurons / 2) * spacing)
                layer.append(neuron)
                self.add(neuron)
            neurons.append(layer)
        
        # Vytvorenie spojení medzi neurónmi
        connections = []
        for i in range(len(neurons) - 1):
            for neuron1 in neurons[i]:
                for neuron2 in neurons[i + 1]:
                    connection = Line(neuron1.get_center(), neuron2.get_center(), color=GRAY)
                    connections.append(connection)
                    self.add(connection)
        
        # Pridanie textového označenia pre vrstvy
        layer_labels = ["Vstupná vrstva", "Skrytá vrstva 1", "Skrytá vrstva 2", "Výstupná vrstva"]
        layer_offsets = [10.5, 21.0, 16.0, 7.5]  # Rôzne výšky pre každú vrstvu
        for i, (label, offset) in enumerate(zip(layer_labels, layer_offsets)):
            layer_text = Text(label, font_size=24)
            layer_text.next_to(neurons[i][0], UP * offset)  # Umiestnenie textu nad neuróny
            self.add(layer_text)

        # Animácia aktivácie neurónov
        self.wait(1)
        for i, layer in enumerate(neurons):
            for neuron in layer:
                self.play(neuron.animate.set_fill(YELLOW, opacity=0.7), run_time=0.3)
            self.wait(0.5)
        
        # Animácia šírenia signálu cez spojenia
        for connection in connections:
            self.play(connection.animate.set_color(BLUE), run_time=0.2)
        
        # Zvýraznenie výstupných neurónov podľa predpokladanej aktivity
        output_values = [0.2, 0.8]  # Simulované hodnoty výstupu
        colors = [interpolate_color(WHITE, RED, value) for value in output_values]
        for neuron, color in zip(neurons[-1], colors):
            self.play(neuron.animate.set_fill(color, opacity=1), run_time=0.5)
        
        # Pridanie textového výstupu
        output_text = Text(f"Výstup: Trieda {output_values.index(max(output_values))}", font_size=24)
        output_text.next_to(neurons[-1][-1], RIGHT * 2)
        self.add(output_text)
        
        self.wait(2)