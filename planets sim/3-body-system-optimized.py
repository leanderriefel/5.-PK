import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numba as nb

plt.rcParams["toolbar"] = "None"

MAX_HISTORY = 10000  # Maximum number of points to keep in history

# Physical constants and simulation parameters see https://arxiv.org/pdf/1705.00527 II.Ci.c. 246 (page 37 / 61)
G = 1  # Universal gravitational constant
m1 = 1  # Mass of Planet 1
m2 = 1  # Mass of Planet 2
m3 = 1  # Mass of Planet 3
v1 = 0.4127173212
v2 = 0.4811313628

# Initial conditions: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3]
state = np.array([-1, 0, v1, v2, 1, 0, v1, v2, 0, 0, -2 * v1, -2 * v2])
history = [state.copy()]

# Set up the plotting window
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.subplots_adjust(bottom=0.1)

fig.set_facecolor("white")
ax.set_facecolor("white")
ax.spines[["bottom", "left", "top", "right"]].set_color("black")
ax.tick_params(axis="x", colors="black")
ax.tick_params(axis="y", colors="black")

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.grid(color="black", linewidth=0.2, alpha=0.4)
ax.set_box_aspect(1)

planet1 = ax.plot([], [], "o", color="#00aaaa", markersize=10, label="Planet 1", alpha=0.8)[0]
planet2 = ax.plot([], [], "o", color="#aa00aa", markersize=10, label="Planet 2", alpha=0.8)[0]
planet3 = ax.plot([], [], "o", color="#aaaa00", markersize=10, label="Planet 3", alpha=0.8)[0]

trail1 = ax.plot([], [], "-", color="#00aaaa", alpha=0.3, linewidth=1)[0]
trail2 = ax.plot([], [], "-", color="#aa00aa", alpha=0.3, linewidth=1)[0]
trail3 = ax.plot([], [], "-", color="#aaaa00", alpha=0.3, linewidth=1)[0]


@nb.njit()
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

    return np.array([vx1, vy1, a1[0], a1[1], vx2, vy2, a2[0], a2[1], vx3, vy3, a3[0], a3[1]])


@nb.njit()
def euler(state, dt=0.01):
    return state + acceleration(state) * dt


@nb.njit()
def rk4(state, dt=0.0001):
    k1 = acceleration(state)
    k2 = acceleration(state + dt * k1 / 2)
    k3 = acceleration(state + dt * k2 / 2)
    k4 = acceleration(state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@nb.njit()
def rkdp45(state, dt=0.01):
    # Dormand–Prince coefficients
    # c-values (time fraction):
    c2, c3, c4, c5, c6, c7 = 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0

    # Butcher tableau:
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

    # Coefficients for 5th-order solution
    b1, b2, b3, b4, b5, b6 = 35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
    # b7 = 0.0 implicit for the 5th order

    # Coefficients for 4th-order *embedded* solution
    # Used for error estimate
    b1s, b2s, b3s, b4s, b5s, b6s, b7s = (
        5179 / 57600,
        0.0,
        7571 / 16695,
        393 / 640,
        -92097 / 339200,
        187 / 2100,
        1 / 40,
    )

    k1 = acceleration(state)
    k2 = acceleration(state + dt * (a21 * k1))
    k3 = acceleration(state + dt * (a31 * k1 + a32 * k2))
    k4 = acceleration(state + dt * (a41 * k1 + a42 * k2 + a43 * k3))
    k5 = acceleration(state + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
    k6 = acceleration(state + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
    k7 = acceleration(state + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))

    # 5th-order solution
    y5 = state + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
    # 4th-order solution (embedded) might be used for error estimate later
    y4 = state + dt * (b1s * k1 + b2s * k2 + b3s * k3 + b4s * k4 + b5s * k5 + b6s * k6 + b7s * k7)

    error = np.linalg.norm(y5 - y4)
    print("dt:", dt, "error:", error)
    if error > 1e-10:
        dt = 0.9 * dt * (1e-10 / error) ** 0.2
        return rkdp45(state, dt)

    return y5


def update(frame):
    # Update the state of the system
    global history
    history.append(rkdp45(history[-1]))
    # if len(history) > MAX_HISTORY:
    #     history.pop(0)
    points = np.array(history)

    # Update the positions of the planets and their trails
    planet1.set_data(np.array([points[-1, 0]]), np.array([points[-1, 1]]))
    planet2.set_data(np.array([points[-1, 4]]), np.array([points[-1, 5]]))
    planet3.set_data(np.array([points[-1, 8]]), np.array([points[-1, 9]]))
    trail1.set_data(np.array([points[:, 0]]), np.array([points[:, 1]]))
    trail2.set_data(np.array([points[:, 4]]), np.array([points[:, 5]]))
    trail3.set_data(np.array([points[:, 8]]), np.array([points[:, 9]]))


def reset(event):
    # Reset the simulation to the initial conditions
    global history
    history = [state.copy()]

    planet1.set_data([], [])
    planet2.set_data([], [])
    planet3.set_data([], [])
    trail1.set_data([], [])
    trail2.set_data([], [])
    trail3.set_data([], [])
    plt.draw()


# Create a button to reset the simulation
reset_button = Button(plt.axes([0.8, 0.025, 0.1, 0.04]), "Reset")
reset_button.on_clicked(reset)

# Start animation
ani = FuncAnimation(fig, update, interval=10, cache_frame_data=False)
plt.show()
