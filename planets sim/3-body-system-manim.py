from manim import *
import numpy as np

# Physical constants and simulation parameters see https://arxiv.org/pdf/1705.00527 II.Ci.c. 246 (page 37 / 61)
G = 1  # Universal gravitational constant
m1 = 1  # Mass of Planet 1
m2 = 1  # Mass of Planet 2
m3 = 1  # Mass of Planet 3
v1 = 0.4127173212
v2 = 0.4811313628

# Initial conditions: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3]
state = np.array([
    -1, 0, v1, v2,
    1, 0, v1, v2,
    0, 0, -2*v1, -2*v2
])

def acceleration(state):
    # Unpack state for planets
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = state

    # Calculate relative positions
    r12 = np.array([x2 - x1, y2 - y1])
    r13 = np.array([x3 - x1, y3 - y1])
    r23 = np.array([x3 - x2, y3 - y2])

    # Calculate distances
    r12_mag = np.sqrt(r12[0] ** 2 + r12[1] ** 2)
    r13_mag = np.sqrt(r13[0] ** 2 + r13[1] ** 2)
    r23_mag = np.sqrt(r23[0] ** 2 + r23[1] ** 2)

    # Calculate gravitational accelerations
    a1 = G * (m2 * r12 / r12_mag**3 + m3 * r13 / r13_mag**3)
    a2 = G * (m1 * (-r12) / r12_mag**3 + m3 * r23 / r23_mag**3)
    a3 = G * (m1 * (-r13) / r13_mag**3 + m2 * (-r23) / r23_mag**3)

    return np.array([
        vx1, vy1, a1[0], a1[1],
        vx2, vy2, a2[0], a2[1],
        vx3, vy3, a3[0], a3[1]
    ])


def euler(state, dt=0.01):
    return state + acceleration(state) * dt


def rk4(state, dt=0.01):
    k1 = acceleration(state)
    k2 = acceleration(state + dt * k1 / 2)
    k3 = acceleration(state + dt * k2 / 2)
    k4 = acceleration(state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class ThreeBodySystem(Scene):
    def construct(self):
        global state

        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=8,
            y_length=8
        )
        self.add(axes)

        config.background_color = WHITE

        colors = [ManimColor("#00aaaa"), ManimColor("#aa00aa"), ManimColor("#aaaa00")]

        trails = VGroup()

        evolution_time = 10
        dt = 0.00001

        # Collect all points first
        points = [state]
        for _ in range(int(evolution_time / dt)):
            state = rk4(state, dt)
            points.append(state)

        # Create trails for each planet
        for i in range(3):
            # Extract coordinates for planet i from all points
            planet_points = [(p[4*i], p[4*i+1]) for p in points]
            # Transform points to scene coordinates
            transformed_points = [axes.c2p(x, y, 0) for x, y in planet_points]
            
            trail = VMobject()
            trail.set_points_smoothly(transformed_points)
            trail.set_color(colors[i])
            trail.set_opacity(0.3)
            trail.stroke_width = 1
            trails.add(trail)

            planet = Dot(color=colors[i], radius=0.04)
            planet.add_updater(lambda p, c=trail: p.move_to(c.get_end()))
            planet.set_z_index(1)
            self.add(planet)

        self.play(
            *[Create(trail) for trail in trails],
            run_time=evolution_time,
            rate_func=linear
        )
