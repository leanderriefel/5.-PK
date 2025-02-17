import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numba as nb
import json

G = 6.6743e-11  # Gravitational constant
AU_to_m = 1.496e11  # Astronomical unit to meters
AUday_to_ms = AU_to_m / 86400  # AU/day to m/s

with open("solar_system.json", "r") as f:
    data = json.load(f)

order = [
    "Sun",
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto",
]

rows = []
for name in order:
    if name in data:
        body = data[name]
        mass = body["mass"]
        pos = body["position"]
        vel = body["velocity"]
        row = [mass, pos["x"], pos["y"], pos["z"], vel["vx"], vel["vy"], vel["vz"]]
        rows.append(row)

# [mass, x, y, z, vx, vy, vz]
bodies = np.array(rows, dtype=np.float64)

bodies[:, 1:4] *= AU_to_m
bodies[:, 4:7] *= AUday_to_ms

history = [[bodies[i, 1:4].copy()] for i in range(bodies.shape[0])]

# Set up 3D plot
fig = plt.figure(figsize=(10, 10), dpi=100)
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.1)

ax.set_xlim(-2e12, 2e12)
ax.set_ylim(-2e12, 2e12)
ax.set_zlim(-2e12, 2e12)

ax.set_box_aspect((1, 1, 1))

planets = [
    ax.plot(
        [], [], [], "o", color=f"C{i}", markersize=10, label=f"Planet {i+1}", alpha=0.8
    )[0]
    for i in range(bodies.shape[0])
]
trails = [
    ax.plot([], [], [], "-", color=f"C{i}", alpha=0.3, linewidth=1)[0]
    for i in range(bodies.shape[0])
]


@nb.njit()
def acceleration(bodies):
    n = bodies.shape[0]
    # Each body now has 7 components: mass, x, y, z, vx, vy, vz.
    # The derivative array will also have 7 components.
    acc = np.zeros((n, 7))
    for i in range(n):
        acc[i, 0] = 0.0  # mass derivative is zero
        acc[i, 1] = bodies[i, 4]  # dx/dt = vx
        acc[i, 2] = bodies[i, 5]  # dy/dt = vy
        acc[i, 3] = bodies[i, 6]  # dz/dt = vz
        for j in range(n):
            if i == j:
                continue
            dx = bodies[j, 1] - bodies[i, 1]
            dy = bodies[j, 2] - bodies[i, 2]
            dz = bodies[j, 3] - bodies[i, 3]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            acc[i, 4] += G * bodies[j, 0] * dx / (dist**3)
            acc[i, 5] += G * bodies[j, 0] * dy / (dist**3)
            acc[i, 6] += G * bodies[j, 0] * dz / (dist**3)
    return acc


@nb.njit()
def rkdp45(bodies, dt=10000):
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

    # Coefficients for 4th-order embedded solution:
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
    if error > 1e-1:
      dt = 0.99 * dt * (1e-1 / error) ** 0.01
      return rkdp45(bodies, dt)
    return y5


@nb.njit()
def euler(bodies, dt=10000):
    acc = acceleration(bodies)
    return bodies + dt * acc


def update(frame):
    global bodies, history
    bodies = rkdp45(bodies)
    for i in range(bodies.shape[0]):
        history[i].append(bodies[i, 1:4].copy())
        # Update planet position: set x and y data, then z.
        planets[i].set_data(np.array([bodies[i, 1]]), np.array([bodies[i, 2]]))
        planets[i].set_3d_properties(np.array([bodies[i, 3]]))
        # Update trail data:
        trail_x = [pos[0] for pos in history[i]]
        trail_y = [pos[1] for pos in history[i]]
        trail_z = [pos[2] for pos in history[i]]
        trails[i].set_data(np.array(trail_x), np.array(trail_y))
        trails[i].set_3d_properties(np.array(trail_z))


def reset(event):
    global history
    history = [[bodies[i, 1:4].copy()] for i in range(bodies.shape[0])]
    for p in planets:
        p.set_data([], [])
        p.set_3d_properties([])
    for t in trails:
        t.set_data([], [])
        t.set_3d_properties([])
    plt.draw()


reset_button = Button(plt.axes([0.8, 0.025, 0.1, 0.04]), "Reset")
reset_button.on_clicked(reset)

ani = FuncAnimation(fig, update, interval=10, cache_frame_data=False)
plt.show()
