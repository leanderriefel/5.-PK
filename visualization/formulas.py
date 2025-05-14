from manim import *

# === User-configurable parameters (edit these) ===
text_color: str = "#000000"
bg_color: str = "#ffffff"
duration: float = 1
# ================================================

if bg_color.lower() in ("transparent", "none"):
    config.transparent = True
else:
    config.background_color = bg_color
    config.transparent = False


class FormulaTransition(Scene):
    def construct(self):
        f1 = MathTex(r"F = G \frac{m_1 m_2}{r^2}", color=text_color)
        f2 = MathTex(r"F = G \frac{m_1 m_2}{\|r\|^3} \hat{r}", color=text_color)

        margin = 0.5
        max_width = config.frame_width - margin
        f1.set_width(max_width)
        f2.set_width(max_width)
        f1.move_to(ORIGIN)
        f2.move_to(ORIGIN)

        self.play(Transform(f1, f2), run_time=duration)
