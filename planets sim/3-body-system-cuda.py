import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numba as nb
from numba import cuda
from math import sqrt  # Add this import for CUDA-compatible sqrt

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

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.grid(color="black", linewidth=0.2, alpha=0.4)
ax.set_box_aspect(1)

planet1 = ax.plot([], [], "o", color="#00aaaa", markersize=10, label="Planet 1", alpha=0.8)[0]
planet2 = ax.plot([], [], "o", color="#aa00aa", markersize=10, label="Planet 2", alpha=0.8)[0]
planet3 = ax.plot([], [], "o", color="#aaaa00", markersize=10, label="Planet 3", alpha=0.8)[0]

trail1 = ax.plot([], [], "-", color="#00aaaa", alpha=0.3, linewidth=1)[0]
trail2 = ax.plot([], [], "-", color="#aa00aa", alpha=0.3, linewidth=1)[0]
trail3 = ax.plot([], [], "-", color="#aaaa00", alpha=0.3, linewidth=1)[0]


@cuda.jit
def acceleration_kernel(state, result):
    """CUDA kernel for calculating accelerations"""
    # Calculate accelerations for all bodies in parallel
    x1, y1 = state[0], state[1]
    x2, y2 = state[4], state[5]
    x3, y3 = state[8], state[9]

    # Calculate relative positions
    r12x, r12y = x2 - x1, y2 - y1
    r13x, r13y = x3 - x1, y3 - y1
    r23x, r23y = x3 - x2, y3 - y2

    # Calculate distances using CUDA-compatible sqrt
    r12_mag = sqrt(r12x**2 + r12y**2)
    r13_mag = sqrt(r13x**2 + r13y**2)
    r23_mag = sqrt(r23x**2 + r23y**2)

    # Calculate gravitational accelerations
    r12_cube = r12_mag**3
    r13_cube = r13_mag**3
    r23_cube = r23_mag**3

    # Planet 1 accelerations
    result[2] = 1.0 * (1.0 * r12x / r12_cube + 1.0 * r13x / r13_cube)  # Using constants directly
    result[3] = 1.0 * (1.0 * r12y / r12_cube + 1.0 * r13y / r13_cube)

    # Planet 2 accelerations
    result[6] = 1.0 * (1.0 * (-r12x) / r12_cube + 1.0 * r23x / r23_cube)
    result[7] = 1.0 * (1.0 * (-r12y) / r12_cube + 1.0 * r23y / r23_cube)

    # Planet 3 accelerations
    result[10] = 1.0 * (1.0 * (-r13x) / r13_cube + 1.0 * (-r23x) / r23_cube)
    result[11] = 1.0 * (1.0 * (-r13y) / r13_cube + 1.0 * (-r23y) / r23_cube)

    # Copy velocities
    result[0] = state[2]
    result[1] = state[3]
    result[4] = state[6]
    result[5] = state[7]
    result[8] = state[10]
    result[9] = state[11]


def acceleration(state):
    """Wrapper function to call CUDA kernel"""
    d_state = cuda.to_device(state)
    d_result = cuda.device_array_like(state)
    acceleration_kernel[1, 1](d_state, d_result)
    return d_result.copy_to_host()


def euler(state, dt=0.01):
    """Euler integration using CUDA-accelerated acceleration calculation"""
    return state + acceleration(state) * dt


def rk4(state, dt=0.0001):
    """RK4 integration using CUDA-accelerated acceleration calculation"""
    k1 = acceleration(state)
    k2 = acceleration(state + dt * k1 / 2)
    k3 = acceleration(state + dt * k2 / 2)
    k4 = acceleration(state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# Initialize CUDA device
cuda.select_device(0)


def update(frame):
    # Update the state of the system
    global history
    history.append(rk4(history[-1]))
    if len(history) > MAX_HISTORY:
        history.pop(0)
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
