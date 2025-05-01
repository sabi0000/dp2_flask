from manim import *

class ArtificialNeuron(Scene):
    def construct(self):
        # Vstupy a váhy
        input_labels = [MathTex(f"x_{i+1}") for i in range(3)]
        weight_labels = [MathTex(f"w_{i+1}") for i in range(3)]
        
        # Rozmiestnenie vstupov a váh
        for i, (input_label, weight_label) in enumerate(zip(input_labels, weight_labels)):
            input_label.move_to(LEFT * 3 + UP * (1 - i))
            weight_label.next_to(input_label, LEFT)
        
        # Neurón - kruh
        neuron = Circle(radius=1, color=BLUE).shift(RIGHT * 2.5)
        neuron_label = MathTex("\\sigma").move_to(neuron.get_center())

        #Weights
        transfer_circle=Circle(radius=0.5, color=ORANGE).shift(RIGHT * 0.5)
        transfer_label = MathTex("\\sum").move_to(transfer_circle.get_center())

        # Spojenie vstupov a váh k neurónu
        input_to_transfer_arrows = [Arrow(input_label.get_right(), transfer_circle.get_left(), buff=0.1) for input_label in input_labels]
        #weight_texts = [MathTex("\\times").next_to(arrow, DOWN) for arrow in input_to_neuron_arrows]

        #Spojenie vah a neuronu
        input_to_neuron_arrows = [Arrow(transfer_circle.get_right(), neuron.get_left(), buff=0.1) for input_label in input_labels]
        
        # Výstup z neurónu
        output_arrow = Arrow(neuron.get_right(), RIGHT * 5, buff=0.1)
        output_label = MathTex("y").next_to(output_arrow, RIGHT)

        # Aktivačná funkcia (ReLU)
        relu_text = MathTex(r"\text{Aktivačná funkcia}").move_to(DOWN * 2)
        relu_box = SurroundingRectangle(relu_text, color=YELLOW)
        
        # Vytvorenie siete
        self.play(*[Write(label) for label in input_labels + weight_labels])
        self.play(Create(transfer_circle), Write(transfer_label))
        self.play(Create(neuron), Write(neuron_label))
        self.play(*[Create(arrow) for arrow in input_to_transfer_arrows])
        self.play(*[Create(arrow) for arrow in input_to_neuron_arrows])
        #self.play(*[Write(weight_text) for weight_text in weight_texts])
        
        # Spojenie výstupu s aktivačnou funkciou
        self.play(Create(output_arrow), Write(output_label))
        self.play(Write(relu_text), Create(relu_box))

        # Zvýraznenie ReLU aktivácie a výstupu
        self.play(relu_text.animate.set_color(YELLOW), neuron.animate.set_fill(BLUE, opacity=0.5))
        self.wait(1)

        # Vymazanie, aby scéna skončila
        self.play(*[FadeOut(mob) for mob in self.mobjects])
