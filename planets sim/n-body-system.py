import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numba as nb

plt.rcParams["toolbar"] = "None"

G = 1  # Gravitational constant

v1 = 0.4127173212
v2 = 0.4811313628

# [mass, x, y, vx, vy]
bodies = np.array(
    [
        [1.0, -1.0, 0.0, v1, v2],
        [1.0, 1.0, 0.0, v1, v2],
        [1.0, 0.0, 0.0, -2 * v1, -2 * v2],
    ],
    dtype=np.float64,
)

history = [[bodies[i, 1:3].copy()] for i in range(bodies.shape[0])]

# Plot
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.subplots_adjust(bottom=0.1)

fig.set_facecolor("white")
ax.set_facecolor("white")
for spine in ax.spines.values():
    spine.set_color("black")
ax.tick_params(axis="x", colors="black")
ax.tick_params(axis="y", colors="black")

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.grid(color="black", linewidth=0.2, alpha=0.4)
ax.set_box_aspect(1)

planets = [
    ax.plot(
        [], [], "o", color=f"C{i}", markersize=10, label=f"Planet {i+1}", alpha=0.8
    )[0]
    for i in range(3)
]
trails = [
    ax.plot([], [], "-", color=f"C{i}", alpha=0.3, linewidth=1)[0] for i in range(3)
]


@nb.njit()
def acceleration(bodies):
    n = bodies.shape[0]
    acc = np.zeros((n, 5))
    for i in range(n):
        acc[i, 0] = 0.0
        acc[i, 1] = bodies[i, 3]
        acc[i, 2] = bodies[i, 4]
        for j in range(n):
            if i == j:
                continue
            dx = bodies[j, 1] - bodies[i, 1]
            dy = bodies[j, 2] - bodies[i, 2]
            dist = np.sqrt(dx * dx + dy * dy)
            acc[i, 3] += G * bodies[j, 0] * dx / (dist**3)
            acc[i, 4] += G * bodies[j, 0] * dy / (dist**3)
    return acc


@nb.njit()
def rkdp45(bodies, dt=0.01):
    # c-values (time fractions)
    c2, c3, c4, c5, c6, c7 = 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0

    # Butcher tableau coefficients:
    a21 = 1 / 5
    a31, a32 = 3 / 40, 9 / 40
    a41, a42, a43 = 44 / 45, -56 / 15, 32 / 9
    a51, a52, a53, a54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
    a61, a62, a63, a64, a65 = (
        9017 / 3168,
        -355 / 33,
        46732 / 5247,
        49 / 176,
        -5103 / 18656,
    )
    a71, a72, a73, a74, a75, a76 = (
        35 / 384,
        0.0,
        500 / 1113,
        125 / 192,
        -2187 / 6784,
        11 / 84,
    )

    # Coefficients for 5th-order solution:
    b1, b2, b3, b4, b5, b6 = 35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
    # b7 = 0.0 implicitly for the 5th order

    # Coefficients for the 4th-order embedded solution:
    b1s, b2s, b3s, b4s, b5s, b6s, b7s = (
        5179 / 57600,
        0.0,
        7571 / 16695,
        393 / 640,
        -92097 / 339200,
        187 / 2100,
        1 / 40,
    )

    k1 = acceleration(bodies)
    k2 = acceleration(bodies + dt * (a21 * k1))
    k3 = acceleration(bodies + dt * (a31 * k1 + a32 * k2))
    k4 = acceleration(bodies + dt * (a41 * k1 + a42 * k2 + a43 * k3))
    k5 = acceleration(bodies + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
    k6 = acceleration(
        bodies + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
    )
    k7 = acceleration(
        bodies + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
    )

    y5 = bodies + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
    y4 = bodies + dt * (
        b1s * k1 + b2s * k2 + b3s * k3 + b4s * k4 + b5s * k5 + b6s * k6 + b7s * k7
    )

    error = np.linalg.norm(y5 - y4)
    print("dt:", dt, "error:", error)
    if error > 1e-10:
        dt = 0.9 * dt * (1e-10 / error) ** 0.2
        return rkdp45(bodies, dt)
    return y5


def update(frame):
    global bodies, history
    bodies = rkdp45(bodies)
    for i in range(bodies.shape[0]):
        history[i].append(bodies[i, 1:3].copy())
        planets[i].set_data(np.array([bodies[i, 1]]), np.array([bodies[i, 2]]))
        trail_x = [pos[0] for pos in history[i]]
        trail_y = [pos[1] for pos in history[i]]
        trails[i].set_data(np.array([trail_x]), np.array([trail_y]))


def reset(event):
    global history
    history = [[bodies[i, 1:3].copy()] for i in range(bodies.shape[0])]
    for p in planets:
        p.set_data([], [])
    for t in trails:
        t.set_data([], [])
    plt.draw()


reset_button = Button(plt.axes([0.8, 0.025, 0.1, 0.04]), "Reset")
reset_button.on_clicked(reset)

ani = FuncAnimation(fig, update, interval=10, cache_frame_data=False)
plt.show()
