from manim import *

class ConvolutionAnimation(Scene):
    def construct(self):
        # Finálny popis - pridaný na začiatku
        final_result = Text("Princíp konvolúcie", font_size=28).to_edge(UP)
        self.play(Write(final_result))
        self.wait(1)

        # Vstupná 4x4 matica
        input_data = [[i * 4 + j + 1 for j in range(4)] for i in range(4)]
        input_matrix = IntegerMatrix(input_data, left_bracket="(", right_bracket=")").scale(0.5)
        input_matrix.move_to(LEFT * 3)

        # 2x2 filter - posunutý bližšie k vstupnej matici
        kernel_data = [[1, -1], [0, 1]]
        kernel = IntegerMatrix(kernel_data, left_bracket="[", right_bracket="]").scale(0.5)
        kernel.move_to(RIGHT * 0.5)  # Posunúť filter bližšie k vstupnej matici

        # Popisy
        label_input = Text("Vstup", font_size=24).next_to(input_matrix, UP)
        label_kernel = Text("Filter", font_size=24).next_to(kernel, UP)

        # Pridanie na scénu
        self.play(Write(label_input), Write(label_kernel))
        self.play(Create(input_matrix), Create(kernel))
        self.wait(1)

        # Prázdna výstupná 3x3 matica - umiestnená vedľa filtra
        output_data = [[0 for _ in range(3)] for _ in range(3)]  # Inicializácia s nulami
        output_matrix_group = IntegerMatrix(output_data, left_bracket="(", right_bracket=")").scale(0.7)
        output_matrix_group.move_to(RIGHT * 3.5)  # Umiestnenie vedľa filtra

        label_output = Text("Výstup", font_size=24).next_to(output_matrix_group, UP)
        self.play(Write(label_output), FadeIn(output_matrix_group))
        self.wait(1)

        # Výpočty pre všetky pozície
        for i in range(3):  # rows
            for j in range(3):  # columns
                rects = VGroup()
                terms = []
                result = 0

                for ki in range(2):
                    for kj in range(2):
                        row = i + ki
                        col = j + kj
                        input_val = input_data[row][col]
                        kernel_val = kernel_data[ki][kj]
                        entry = input_matrix.get_entries()[row * 4 + col]
                        rect = SurroundingRectangle(entry, color=BLUE)
                        rects.add(rect)
                        terms.append(f"{input_val} \\cdot {kernel_val}")
                        result += input_val * kernel_val

                # Zobrazenie výpočtu
                equation = MathTex(" + ".join(terms) + f" = {result}").scale(0.7)
                equation.to_edge(DOWN)

                # Zvýraznenie
                self.play(Create(rects))
                self.play(FadeIn(equation))
                self.wait(0.5)

                # Zápis výsledku do výstupnej matice
                output_index = i * 3 + j
                new_entry = Integer(result).scale(0.7)
                new_entry.move_to(output_matrix_group.get_entries()[output_index].get_center())
                self.play(Transform(output_matrix_group.get_entries()[output_index], new_entry))
                self.wait(0.3)

                self.play(FadeOut(rects), FadeOut(equation))

        # Finálny popis zostáva na obrazovke
        self.wait(2)