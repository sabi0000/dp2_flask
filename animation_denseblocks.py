from manim import *

class DenseBlock1Animation(Scene):
    def construct(self):
        # Parametre
        num_conv_layers = 6  # Počet konvolučných vrstiev
        radius = 0.12  # Veľkosť bodky
        spacing = 0.6  # Vzdialenosť medzi bodkami

        # Vytvorenie bodiek pre konvolučné vrstvy
        conv_dots = VGroup()
        for i in range(num_conv_layers):
            dot = Dot(radius=radius, color=BLUE)
            dot.move_to([0, i * spacing - (num_conv_layers - 1) * spacing / 2, 0])  # Rovnomerne rozmiestnené nad sebou
            conv_dots.add(dot)

        # Prechodová vrstva: 1x1 Conv + Pooling
        transition_dot1 = Dot(radius=radius, color=GREEN).next_to(conv_dots, RIGHT, buff=1)
        transition_dot2 = Dot(radius=radius, color=GREEN).next_to(transition_dot1, RIGHT, buff=1)

        # Pridanie textov k vrstvám
        conv_labels = VGroup()
        for i in range(num_conv_layers):
            label = Text(f"Conv {i+1}", font_size=18).next_to(conv_dots[i], DOWN)
            conv_labels.add(label)

        transition_label1 = Text("1×1 Conv", font_size=18).next_to(transition_dot1, DOWN)
        transition_label2 = Text("Pooling 2×2", font_size=18).next_to(transition_dot2, DOWN)

        # Zobrazenie bodiek a textov
        self.play(FadeIn(conv_dots), Write(conv_labels))
        self.wait(0.5)

        # Pridanie šípok medzi bodkami konvolučných vrstiev
        arrows = VGroup()
        for i in range(1, num_conv_layers):
            arrow = Arrow(conv_dots[i-1].get_right(), conv_dots[i].get_left(), buff=0.1, color=WHITE)
            arrows.add(arrow)

        # Zobrazenie šípok medzi konvolučnými vrstvami
        self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.1))
        self.wait(0.5)

        # Zobrazenie prechodovej vrstvy
        self.play(FadeIn(transition_dot1), FadeIn(transition_dot2))
        self.play(Write(transition_label1), Write(transition_label2))
        self.wait(0.5)

        # Pridanie šípok medzi konvolučnými vrstvami a prechodovou vrstvou
        transition_arrows = VGroup(
            Arrow(conv_dots[num_conv_layers-1].get_right(), transition_dot1.get_left(), buff=0.1, color=WHITE),
            Arrow(transition_dot1.get_right(), transition_dot2.get_left(), buff=0.1, color=WHITE)
        )

        self.play(LaggedStartMap(GrowArrow, transition_arrows, lag_ratio=0.1))
        self.wait(1)

        # Text na vysvetlenie DenseBlocku
        explanation_text = Text(
            "DenseBlock: 6 konvolučných vrstiev, následne 1×1 Conv + Pooling",
            font_size=24
        ).to_edge(UP)

        self.play(Write(explanation_text))
        self.wait(2)

        # Fade out pre čistý záver
        self.play(*[FadeOut(m) for m in self.mobjects])
