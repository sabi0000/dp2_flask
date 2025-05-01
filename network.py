from manim import *

class DisplayNetwork(Scene):
    def construct(self):
        # Definuj cestu k obrázku
        #image_path = "/content/drive/MyDrive/archive_skin/HAM10000_images_part_1/ISIC_0026503.jpg"
        image_path='static/Circle.png'
        #cv2.imwrite(blurred_image_path, blurred_image)
        # Načítanie obrázka
        image = ImageMobject(image_path)

        image.scale(1.5)

        # Umiestnenie obrázka do ľavej časti scény
        image.to_edge(LEFT)

        # Prvý stĺpec kruhov (8 kruhov)
        column_1 = VGroup()
        for i, label_text in enumerate(["x1", "x2", "x3", "x4","...", "x781", "x782", "x783", "x784"]):
            circle = Circle(radius=0.3, color=BLUE)
            label = Text(label_text, font_size=15).move_to(circle.get_center())
            column_1.add(VGroup(circle, label))  # Skupina kruhu a labelu
        column_1.arrange(DOWN, buff=0.3)  # Vertikálne usporiadanie s medzerou 0.5
        column_1.next_to(image, RIGHT, buff=1)  # Umiestnenie vedľa obrázka

        # Druhý stĺpec kruhov (7 kruhov)
            #column_2 = VGroup(*[Circle(radius=0.3, color=GREEN) for _ in range(7)])
        column_2 = VGroup()
        for i, label_text in enumerate(["B1", "B2", "B3", "B4", "B5", "B6", "B7"]):
            circle = Circle(radius=0.3, color=GREEN)
            label = Text(label_text, font_size=15).move_to(circle.get_center())
            column_2.add(VGroup(circle, label))
        column_2.arrange(DOWN, buff=0.3)  # Vertikálne usporiadanie s medzerou 0.5
        column_2.next_to(column_1, RIGHT, buff=0.8)  # Vedľa prvého stĺpca

        column_3 = VGroup(*[Circle(radius=0.3, color=ORANGE) for _ in range(5)])
        column_3.arrange(DOWN, buff=0.3)  # Vertikálne usporiadanie s medzerou 0.5
        column_3.next_to(column_2, RIGHT, buff=0.8)  # Vedľa prvého stĺpca

        column_4 = VGroup(*[Circle(radius=0.3, color=YELLOW) for _ in range(3)])
        column_4.arrange(DOWN, buff=0.3)  # Vertikálne usporiadanie s medzerou 0.5
        column_4.next_to(column_3, RIGHT, buff=0.8)  # Vedľa prvého stĺpca

        #numbers = VGroup(*[Text(str(num), font_size=24) for num in range(1, 10)])
        numbers = VGroup()
        for i, label_text in enumerate(["0,8", "0,2", "0,1", "0,3", "0,8", "0,1", "0,9","0,2","0,9"]):
            label = Text(label_text, font_size=15)
            numbers.add(VGroup(label))
        numbers.arrange(DOWN, buff=0.7)  # Vertikálne usporiadanie s medzerou 0.5
        numbers.next_to(column_1, RIGHT*0.4, buff=0.8)  # Medzi column_1 a column_2

        #numbers2 = VGroup(*[Text(str(num), font_size=24) for num in range(1, 7)])
        numbers2 = VGroup()
        for i, label_text in enumerate(["0,9", "0,1", "0,2", "0,7", "0,8", "0,1"]):
            label = Text(label_text, font_size=15)
            numbers2.add(VGroup(label))
        numbers2.arrange(DOWN*1, buff=0.6)  # Vertikálne usporiadanie s medzerou 0.5
        numbers2.next_to(column_2, RIGHT*0.4, buff=0.8)  # Medzi column_1 a column_2

            #numbers3 = VGroup(*[Text(str(num), font_size=24) for num in range(1, 6)])
        numbers3=VGroup()
        for i, label_text in enumerate(["0,6", "0,3", "0,7", "0,6", "0,3"]):
            label = Text(label_text, font_size=15)
            numbers3.add(VGroup(label))
        numbers3.arrange(DOWN, buff=0.7)  # Vertikálne usporiadanie s medzerou 0.5
        numbers3.next_to(column_3, RIGHT*0.4, buff=0.8)  # Medzi column_1 a column_2

            #numbers4 = VGroup(*[Text(str(num), font_size=24) for num in range(1, 4)])
        numbers4=VGroup()
        for i, label_text in enumerate(["0,04", "0,9", "0,06"]):
            label = Text(label_text, font_size=15)
            numbers4.add(VGroup(label))
        numbers4.arrange(DOWN, buff=0.7)  # Vertikálne usporiadanie s medzerou 0.5
        numbers4.next_to(column_4, RIGHT*0.4, buff=0.8)  # Medzi column_1 a column_2

        connections = VGroup()
        connection_rules = [
            [1],            # Prvý kruh v column_1 spojený s prvým kruhom v column_2
            [2],            # Druhý kruh v column_1 spojený s druhým kruhom v column_2
            [1, 3],         # Tretí kruh v column_1 spojený s prvým a tretím kruhom v column_2
            [2, 4],         # Štvrtý kruh v column_1 spojený s druhým a štvrtým kruhom v column_2
            [3, 5],         # Piaty kruh v column_1 spojený s tretím a piatym kruhom v column_2
            [4, 6],         # Šiesty kruh v column_1 spojený s štvrtým a šiestym kruhom v column_2
            [5, 7],         # Siedmy kruh v column_1 spojený s piatym a siedmym kruhom v column_2
            [6],             # Posledný kruh v column_1 spojený so šiestym kruhom v column_2
            [7]
        ]

        # Vytvorenie spojení podľa pravidiel
        for i, targets in enumerate(connection_rules):
            c1 = column_1[i][0]  # Kruh v column_1
            for target in targets:
                c2 = column_2[target - 1][0]  # Cieľový kruh v column_2 (indexované od 0)
                line = Line(c1.get_right(), c2.get_left(), color=YELLOW)
                connections.add(line)

        connections2 = VGroup()
        connection2_rules = [
            [1],            # Prvý kruh v column_1 spojený s prvým kruhom v column_2
            [2],            # Druhý kruh v column_1 spojený s druhým kruhom v column_2
            [1, 3],         # Tretí kruh v column_1 spojený s prvým a tretím kruhom v column_2
            [2, 4],         # Štvrtý kruh v column_1 spojený s druhým a štvrtým kruhom v column_2
            [3, 5],         # Piaty kruh v column_1 spojený s tretím a piatym kruhom v column_2
            [4],         # Šiesty kruh v column_1 spojený s štvrtým a šiestym kruhom v column_2
            [5]        # Siedmy kruh v column_1 spojený s piatym a siedmym kruhom v column_2
        ]

        # Vytvorenie spojení podľa pravidiel
        for i, targets in enumerate(connection2_rules):
            c1 = column_2[i][0]  # Kruh v column_1
            for target in targets:
                c2 = column_3[target - 1][0]  # Cieľový kruh v column_2 (indexované od 0)
                line = Line(c1.get_right(), c2.get_left(), color=YELLOW)
                connections.add(line)


        connections3 = VGroup()
        connection3_rules = [
            [1],            # Prvý kruh v column_1 spojený s prvým kruhom v column_2
            [2],            # Druhý kruh v column_1 spojený s druhým kruhom v column_2
            [1, 3],         # Tretí kruh v column_1 spojený s prvým a tretím kruhom v column_2
         # Piaty kruh v column_1 spojený s tretím a piatym kruhom v column_2
            [2],         # Šiesty kruh v column_1 spojený s štvrtým a šiestym kruhom v column_2
            [3]        # Siedmy kruh v column_1 spojený s piatym a siedmym kruhom v column_2
        ]

        # Vytvorenie spojení podľa pravidiel
        for i, targets in enumerate(connection3_rules):
            c1 = column_3[i][0]  # Kruh v column_1
            for target in targets:
                c2 = column_4[target - 1][0]  # Cieľový kruh v column_2 (indexované od 0)
                line = Line(c1.get_right(), c2.get_left(), color=YELLOW)
                connections.add(line)

        # Pridanie obrázka a kruhov do scény
        self.play(FadeIn(image), FadeIn(column_1), FadeIn(column_2), FadeIn(column_3), FadeIn(column_4),FadeIn(numbers),FadeIn(connections),FadeIn(numbers2),FadeIn(numbers3),FadeIn(numbers4),FadeIn(connections2),FadeIn(connections3))
        self.wait(2)  # Počkáme 2 sekundy na zobrazenie
