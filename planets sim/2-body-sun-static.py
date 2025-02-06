import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# Configure matplotlib to display without toolbar
plt.rcParams["toolbar"] = "None"

# Physical constants and simulation parameters
G = 10  # Universal gravitational constant (simulation units)
m1 = 1000  # Mass of central body (Sun)
m2 = 10  # Mass of orbiting body (Planet)

# Calculate initial conditions
initial_distance = 10  # Starting distance between bodies
v_orbital = np.sqrt(G * m1 / initial_distance)  # Circular orbit velocity

# Initial state vector [x, y, vx, vy]
state = np.array([initial_distance, 0, 0, v_orbital])
history = [state.copy()]


# Set up the plotting window
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.subplots_adjust(bottom=0.1)

fig.set_facecolor("#000010")
ax.set_facecolor("#000010")
ax.spines[["bottom", "left"]].set_color("white")
ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.grid(color="white", linewidth=0.2, alpha=0.4)

sun = ax.plot(0, 0, "o", color="#ffff00", markersize=20, label="Sun", alpha=0.8)[0]
planet = ax.plot(
    [], [], "o", color="#00ffff", markersize=10, label="Planet", alpha=0.8
)[0]

trail = ax.plot([], [], "-", color="#00ffff", alpha=0.3, linewidth=1)[0]


def acceleration(state):
    x, y, vx, vy = state
    r = np.array([x, y])
    r_mag = np.linalg.norm(r)
    a = -G * m1 * r / r_mag**3
    return np.array([vx, vy, a[0], a[1]])


def euler(state, dt=0.01):
    return state + acceleration(state) * dt


def rk4(state, dt=0.01):
    k1 = acceleration(state)
    k2 = acceleration(state + dt * k1 / 2)
    k3 = acceleration(state + dt * k2 / 2)
    k4 = acceleration(state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def update(frame):
    global history
    history.append(rk4(history[-1]))

    points = np.array(history)
    planet.set_data(np.array([points[-1, 0]]), np.array([points[-1, 1]]))
    trail.set_data(np.array([points[:, 0]]), np.array([points[:, 1]]))


def reset(event):
    global history, state
    history = [np.array([initial_distance, 0, 0, v_orbital])]
    planet.set_data([], [])
    trail.set_data([], [])
    plt.draw()


# Create and configure reset button
ax_reset = plt.axes([0.85, 0.05, 0.1, 0.05])
reset_button = Button(ax_reset, "Reset", color="#000030", hovercolor="#000060")
reset_button.label.set_color("white")
reset_button.on_clicked(reset)

# Configure and position legend
plt.legend(
    loc="upper right",
    fontsize="medium",
    facecolor="#000030",
    edgecolor="none",
    labelcolor="white",
)

# Start animation
ani = FuncAnimation(fig, update, interval=20)
plt.show()
