from manim import *

class CNNArchitecture(Scene):
    def create_layer(self, num_dots, center, radius=0.15, spacing=0.25, color=BLUE):
        # Vytvorí stĺpec bodiek ako reprezentáciu vrstvy
        dots = VGroup()
        for i in range(num_dots):
            dot = Dot(radius=radius, color=color)
            dot.move_to(center + DOWN * i * spacing)
            dots.add(dot)
        return dots

    def construct(self):
        # Definuj štruktúru vrstiev: (typ, počet neurónov na vizualizáciu, farba)
        layers = [
            ("Input", 3, GREEN),
            ("Conv2D 1024", 20, BLUE),
            ("Conv2D 512", 16, BLUE),
            ("Conv2D 256", 12, BLUE),
            ("Conv2D 128", 8, BLUE),
            ("Flatten", 1, WHITE),
            ("Dense 64", 6, ORANGE),
            ("Output 7", 7, RED),
        ]

        all_layers = VGroup()
        x_shift = 0

        # Vytvor vrstvy ako skupiny bodiek
        for (label, num_neurons, color) in layers:
            layer = self.create_layer(num_neurons, center=ORIGIN, color=color)
            layer.move_to(RIGHT * x_shift)
            text = Text(label, font_size=24).next_to(layer, UP)
            group = VGroup(layer, text)
            all_layers.add(group)
            x_shift += 1.8

        # Posuň všetky vrstvy na stred
        all_layers.move_to(ORIGIN)

        # Zobraz vrstvy postupne
        for layer in all_layers:
            self.play(FadeIn(layer), run_time=0.5)
            self.wait(0.1)

        # Pridaj šípky medzi vrstvami
        arrows = VGroup()
        for i in range(len(all_layers) - 1):
            start = all_layers[i][0].get_right()
            end = all_layers[i+1][0].get_left()
            arrow = Arrow(start, end, buff=0.1, stroke_width=2)
            arrows.add(arrow)

        self.play(Create(arrows), run_time=1)
        self.wait(2)
