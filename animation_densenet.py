from manim import *

class DenseNet121Dots(Scene):
    def create_layer(self, num_dots, center, radius=0.06, spacing=0.15, color=BLUE):
        dots = VGroup()
        for i in range(num_dots):
            dot = Dot(radius=radius, color=color)
            dot.move_to(center + DOWN * i * spacing)
            dots.add(dot)
        return dots

    def construct(self):
        # Definícia vrstiev: (názov, počet neurónov (bodiek), farba)
        layers = [
            ("Conv+Pool", 4, TEAL),
            ("Dense Block 1 (6)", 6, GREEN),
            ("Transition 1", 1, ORANGE),
            ("Dense Block 2 (12)", 12, GREEN),
            ("Transition 2", 1, ORANGE),
            ("Dense Block 3 (24)", 24, GREEN),
            ("Transition 3", 1, ORANGE),
            ("Dense Block 4 (16)", 16, GREEN),
            ("Global Avg Pool", 1, PURPLE),
            ("FC (1000)", 10, RED),
        ]

        all_layers = VGroup()
        x_shift = 0

        for (label, num_neurons, color) in layers:
            layer = self.create_layer(num_neurons, center=ORIGIN, color=color)
            layer.move_to(RIGHT * x_shift)
            text = Text(label, font_size=20).next_to(layer, UP)
            group = VGroup(layer, text)
            all_layers.add(group)
            x_shift += 1.0  # užší spacing kvôli množstvu vrstiev

        all_layers.move_to(ORIGIN)

        # Animácia vrstiev
        for layer in all_layers:
            self.play(FadeIn(layer), run_time=0.3)
            self.wait(0.05)

        # Šípky medzi vrstvami
        arrows = VGroup()
        for i in range(len(all_layers) - 1):
            start = all_layers[i][0].get_right()
            end = all_layers[i + 1][0].get_left()
            arrow = Arrow(start, end, buff=0.1, stroke_width=2)
            arrows.add(arrow)

        self.play(Create(arrows), run_time=1)
        self.wait(2)
