from manim import *
import random
class ResNet50Dots(Scene):
    def create_layer(self, num_dots, center, radius=0.06, spacing=0.15, color=BLUE):
        dots = VGroup()
        for i in range(num_dots):
            dot = Dot(radius=radius, color=color)
            dot.move_to(center + DOWN * i * spacing)
            dots.add(dot)
        return dots

    def create_residual_block(self, columns, center, radius=0.06, spacing=0.15, color=GREEN):
        block = VGroup()
        num_columns = len(columns)
        # Vypočítame posun pre zarovnanie na stred
        for i, num_dots in enumerate(columns):
            column_center = center + (i - (num_columns - 1) / 2) * 0.3  # Posun stĺpcov na stred
            column = self.create_layer(num_dots, center=column_center, radius=radius, spacing=spacing, color=color)
            # Posun stĺpcov vertikálne pre variabilitu
            vertical_offset = (num_columns - 1) * 0.9 - i * 0.3  # Pridanie variabilného posunu
            column.shift(UP * vertical_offset)
            block.add(column)
        return block

    def construct(self):
        # Definícia vrstiev: (názov, počet neurónov (bodiek), farba)
        layers = [
            ("Conv1 + Pool", 6, TEAL),             # Počiatočná konvolúcia + pooling
            ("Residual Block 1", [6, 6, 9], GREEN), # 1. Residual Block
            ("Residual Block 2", [8, 8, 12], GREEN),    # 2. Residual Block
            ("Residual Block 3", [12, 12, 16], GREEN),    # 3. Residual Block
            ("Residual Block 4", [16, 16, 20], GREEN),    # 4. Residual Block
            ("Fully Connected", 1, RED),           # Plne pripojené vrstvy
            ("Softmax", 1, PURPLE)                 # Softmax na výstup
        ]

        all_layers = VGroup()
        x_shift = 0

        # Vytvorenie vrstiev pomocou bodiek
        for (label, num_neurons, color) in layers:
            if isinstance(num_neurons, list):  # Pre reziduálny blok
                layer = self.create_residual_block(num_neurons, center=ORIGIN, color=color)
            else:
                layer = self.create_layer(num_neurons, center=ORIGIN, color=color)
            layer.move_to(RIGHT * x_shift)
            text = Text(label, font_size=20).next_to(layer, UP)
            group = VGroup(layer, text)
            all_layers.add(group)
            x_shift += 2  # Rozšírenie spacingu medzi vrstvami

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

        self.play(Create(arrows), run_time=2)
        self.wait(2)