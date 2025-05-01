from manim import *

class ArtificialNeuron(Scene):
    def construct(self):
        # Vstupy a váhy
        input_labels = [MathTex(f"x_{i+1}") for i in range(3)]
        weight_labels = [MathTex(f"w_{i+1}") for i in range(3)]
        weight_circles = [Circle(radius=0.3, color=ORANGE) for _ in range(3)]

        # Rozmiestnenie vstupov, váh a kruhov pre váhy
        for i, (input_label, weight_label, weight_circle) in enumerate(zip(input_labels, weight_labels, weight_circles)):
            input_label.move_to(LEFT * 4 + UP * (1 - i))
            weight_circle.next_to(input_label, RIGHT, buff=0.5)
            weight_label.move_to(weight_circle.get_center())

        # Transfer funkcia (sčítanie)
        transfer_circle = Circle(radius=0.5, color=ORANGE).shift(RIGHT * 0.5)
        transfer_label = MathTex("\\sum").move_to(transfer_circle.get_center())

        # Neurón (ovál namiesto kruhu)
        neuron = Ellipse(width=2, height=1.5, color=BLUE).shift(RIGHT * 2.5)
        neuron_label = MathTex("\\sigma").move_to(neuron.get_center())

        # Spojenie vstupov a váh k transfer funkcii
        input_to_weight_arrows = [Arrow(input_label.get_right(), weight_circle.get_left(), buff=0.1) for input_label, weight_circle in zip(input_labels, weight_circles)]
        weight_to_transfer_arrows = [Arrow(weight_circle.get_right(), transfer_circle.get_left(), buff=0.1) for weight_circle in weight_circles]

        # Spojenie transfer funkcie s neurónom
        transfer_to_neuron_arrow = Arrow(transfer_circle.get_right(), neuron.get_left(), buff=0.1)

        # Výstup z neurónu
        output_arrow = Arrow(neuron.get_right(), RIGHT * 5, buff=0.1)
        output_label = MathTex("y").next_to(output_arrow, RIGHT)

        # Aktivačná funkcia (ReLU)
        relu_text = MathTex(r"\text{ReLU}(z)").move_to(DOWN * 2)
        relu_box = SurroundingRectangle(relu_text, color=YELLOW)

        # Animácie
        self.play(*[Write(label) for label in input_labels + weight_labels])
        self.play(*[Create(circle) for circle in weight_circles])
        self.play(Create(transfer_circle), Write(transfer_label))
        self.play(Create(neuron), Write(neuron_label))
        self.play(*[Create(arrow) for arrow in input_to_weight_arrows])
        self.play(*[Create(arrow) for arrow in weight_to_transfer_arrows])
        self.play(Create(transfer_to_neuron_arrow))
        self.play(Create(output_arrow), Write(output_label))
        self.play(Write(relu_text), Create(relu_box))

        # Zvýraznenie aktivácie
        self.play(relu_text.animate.set_color(YELLOW), neuron.animate.set_fill(BLUE, opacity=0.5))
        self.wait(1)

        # Vymazanie scény
        self.play(*[FadeOut(mob) for mob in self.mobjects])
