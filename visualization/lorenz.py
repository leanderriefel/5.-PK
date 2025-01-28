from manim import *
from scipy.integrate import solve_ivp


def lorenz_equation(t, state, sigma=10, rho=28, beta=8 / 3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])


def integrate_lorenz(function, state0, time, dt=0.01):
    solution = solve_ivp(
        function, t_span=(0, time), y0=state0, t_eval=np.arange(0, time, dt)
    )
    return solution.y.T


class Lorenz(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=(-30, 30, 5), y_range=(-30, 30, 5), z_range=(0, 50, 5)
        ).shift(IN * 3)
        self.add(axes)

        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.1)

        config.background_color = DARKER_GRAY

        evolution_time = 100
        speed = 0.5

        episilon = 1e-4
        n_states = 4

        states0 = (np.array([10.0, 10.0, 10.0 + i * episilon]) for i in range(n_states))
        colors = color_gradient(
            [
                ManimColor("#39FF14"),
                ManimColor("#00FEFC"),
                ManimColor("#FFFF00"),
                ManimColor("#FF13F0"),
            ],
            n_states,
        )

        curves = VGroup()
        blured_curves = VGroup()
        dots = []

        for state0, color in zip(states0, colors):
            points = integrate_lorenz(lorenz_equation, state0, evolution_time * speed)

            curve = VMobject()
            curve.set_points_smoothly(axes.coords_to_point(points))
            curve.set_stroke(color=color, width=1)
            curve.cap_style = CapStyleType.ROUND

            dot = Dot(radius=0.075, color=DARKER_GRAY, fill_opacity=1)
            dot.set_stroke(color=color, width=1)
            dot.add_updater(lambda d, c=curve: d.move_to(c.get_end()))
            dots.append(dot)

            curves.add(curve)

            for i in range(1, 4):
                blured_curve = curve.copy()
                blured_curve.set_stroke(
                    color=color, width=(1 + i * 10), opacity=(0.05 / i**2)
                )
                blured_curves.add(blured_curve)

        for dot in dots:
            dot.set_z_index(1)
            self.add_fixed_orientation_mobjects(dot)

        self.play(
            *[Create(curve) for curve in curves],
            *[Create(blured_curve) for blured_curve in blured_curves],
            run_time=evolution_time,
            rate_func=linear
        )
