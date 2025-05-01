from manim import *

class NeuronAnimation(Scene):
    def construct(self):
        # 🧠 Načítanie obrázka neurónu
        neuron_image = ImageMobject("neuron.png").scale(1)
        
        # ⚡ Cesta signálu (jednoduchá červená čiara)
        path = Line(start=LEFT * 3, end=RIGHT * 3, color=RED, stroke_width=4)

        # 🔥 Pohybujúci sa signál (bod)
        signal = Dot(radius=0.2, color=YELLOW).move_to(path.start)

        # 📌 Pridanie obrázka a trajektórie signálu
        self.add(neuron_image, path)

        # 🚀 Animácia signálu cez neurón
        self.play(MoveAlongPath(signal, path), run_time=2, rate_func=linear)

        self.wait()
