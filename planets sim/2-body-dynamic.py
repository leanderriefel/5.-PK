import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# Configure matplotlib to display without toolbar
plt.rcParams["toolbar"] = "None"

# Physical constants and simulation parameters
G = 10  # Universal gravitational constant (simulation units)
m1 = 1  # Mass of Planet 1
m2 = 1000000  # Mass of Planet 2

# Calculate initial conditions
initial_distance = 10  # Starting distance between bodies

# Approximation for orbital speed:
e = 0.2
v = np.sqrt(G * (m1 + m2) * (1 + e) / initial_distance)

# Initial state vector: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
state = np.array([-initial_distance, 0, 0, v, 0, 0, 0, 0])
history = [state.copy()]

# Set up the plotting window
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.subplots_adjust(bottom=0.1)

fig.set_facecolor("#white")
ax.set_facecolor("#white")
ax.spines[["bottom", "left", "top", "right"]].set_color("black")
ax.tick_params(axis="x", colors="black")
ax.tick_params(axis="y", colors="black")

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.grid(color="black", linewidth=0.2, alpha=0.4)
ax.set_box_aspect(1)

planet1 = ax.plot(
    [], [], "o", color="#00aaaa", markersize=10, label="Planet 1", alpha=0.8
)[0]
planet2 = ax.plot(
    [], [], "o", color="#aa00aa", markersize=10, label="Planet 2", alpha=0.8
)[0]

trail1 = ax.plot([], [], "-", color="#00aaaa", alpha=0.3, linewidth=1)[0]
trail2 = ax.plot([], [], "-", color="#aa00aa", alpha=0.3, linewidth=1)[0]


def acceleration(state):
    # Unpack state for both planets
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state
    r = np.array([x2 - x1, y2 - y1])
    r_mag = np.linalg.norm(r)
    # Calculate gravitational acceleration based on Newton's law
    a1 = G * m2 * r / r_mag**3
    a2 = G * m1 * (-r) / r_mag**3
    return np.array([vx1, vy1, a1[0], a1[1], vx2, vy2, a2[0], a2[1]])


def euler(state, dt=0.01):
    return state + acceleration(state) * dt


def rk4(state, dt=0.001):
    k1 = acceleration(state)
    k2 = acceleration(state + dt * k1 / 2)
    k3 = acceleration(state + dt * k2 / 2)
    k4 = acceleration(state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def update(frame):
    global history
    history.append(rk4(history[-1]))
    points = np.array(history)

    planet1.set_data(np.array([points[-1, 0]]), np.array([points[-1, 1]]))
    planet2.set_data(np.array([points[-1, 4]]), np.array([points[-1, 5]]))
    trail1.set_data(np.array([points[:, 0]]), np.array([points[:, 1]]))
    trail2.set_data(np.array([points[:, 4]]), np.array([points[:, 5]]))


def reset(event):
    global history
    history = [state.copy()]

    planet1.set_data([], [])
    planet2.set_data([], [])
    trail1.set_data([], [])
    trail2.set_data([], [])
    plt.draw()


# Create and configure reset button
ax_reset = plt.axes([0.85, 0.05, 0.1, 0.05])
reset_button = Button(ax_reset, "Reset", color="white", hovercolor="#efefef")
reset_button.label.set_color("black")
reset_button.on_clicked(reset)

# Start animation
ani = FuncAnimation(fig, update, interval=20, cache_frame_data=False)
plt.show()
