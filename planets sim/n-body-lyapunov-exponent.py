import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numba as nb
import json

# Removed keyboard polling to avoid overhead

# Constants and unit conversions
G = 6.6743e-11
AU_to_m = 1.496e11  # 1 AU in metersb
AUday_to_ms = AU_to_m / 86400  # AU/day to m/s

# Load solar system data from JSON
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

# Create state array: [mass, x, y, z, vx, vy, vz]
bodies = np.array(rows, dtype=np.float64)
bodies[:, 1:4] *= AU_to_m
bodies[:, 4:7] *= AUday_to_ms

# Initial small perturbation for Lyapunov calculation
delta0 = 1e-10
bodies_delta = bodies.copy() + delta0

# Lists to store snapshots and dt history
history = [bodies[1:].copy()]  # Skip the Sun if desired
history_delta = [bodies_delta[1:].copy()]
dt_history = []


# Acceleration function: parallelized and optimized with Numba
@nb.njit(cache=True, fastmath=True, parallel=True)
def acceleration(bodies):
    n = bodies.shape[0]
    acc = np.zeros((n, 7))
    for i in nb.prange(n):
        # mass derivative remains 0; velocities copied over:
        acc[i, 0] = 0.0
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
            # Avoid division by zero:
            inv_dist3 = 1.0 / (dist * dist * dist + 1e-20)
            acc[i, 4] += G * bodies[j, 0] * dx * inv_dist3
            acc[i, 5] += G * bodies[j, 0] * dy * inv_dist3
            acc[i, 6] += G * bodies[j, 0] * dz * inv_dist3
    return acc


# Adaptive RKDP45 integrator (returns new state and dt used)
@nb.njit(cache=True, fastmath=True)
def rkdp45(bodies, dt=1000000.0):
    # Butcher tableau coefficients
    a21 = 1.0 / 5.0
    a31, a32 = 3.0 / 40.0, 9.0 / 40.0
    a41, a42, a43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
    a51, a52, a53, a54 = (
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
    )
    a61, a62, a63, a64, a65 = (
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    )
    a71, a72, a73, a74, a75, a76 = (
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    )

    # Coefficients for 5th-order solution:
    b1, b2, b3, b4, b5, b6 = (
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    )
    # Coefficients for 4th-order embedded solution:
    b1s, b2s, b3s, b4s, b5s, b6s, b7s = (
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    )

    k1 = acceleration(bodies)
    k2 = acceleration(bodies + dt * (a21 * k1))
    k3 = acceleration(bodies + dt * (a31 * k1 + a32 * k2))
    k4 = acceleration(bodies + dt * (a41 * k1 + a42 * k2 + a43 * k3))
    k5 = acceleration(bodies + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
    k6 = acceleration(bodies + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
    k7 = acceleration(bodies + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))

    y5 = bodies + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
    y4 = bodies + dt * (b1s * k1 + b2s * k2 + b3s * k3 + b4s * k4 + b5s * k5 + b6s * k6 + b7s * k7)
    error = np.linalg.norm(y5 - y4)
    if error > 1e-10:
        dt = 0.99 * dt * (1e-10 / error) ** 0.1
        return rkdp45(bodies, dt)
    return y5, dt


# Fixed-step RKDP45 for the perturbed trajectory (uses dt from reference)
@nb.njit(cache=True, fastmath=True)
def rkdp45_fixed(bodies, dt):
    a21 = 1.0 / 5.0
    a31, a32 = 3.0 / 40.0, 9.0 / 40.0
    a41, a42, a43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
    a51, a52, a53, a54 = (
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
    )
    a61, a62, a63, a64, a65 = (
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    )
    a71, a72, a73, a74, a75, a76 = (
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    )
    b1, b2, b3, b4, b5, b6 = (
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    )

    k1 = acceleration(bodies)
    k2 = acceleration(bodies + dt * (a21 * k1))
    k3 = acceleration(bodies + dt * (a31 * k1 + a32 * k2))
    k4 = acceleration(bodies + dt * (a41 * k1 + a42 * k2 + a43 * k3))
    k5 = acceleration(bodies + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
    k6 = acceleration(bodies + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
    y = bodies + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
    return y


# Simulation loop parameters
target_total_time = 1e30
iteration = 0

while sum(dt_history) < target_total_time:
    new_bodies, dt_used = rkdp45(bodies)
    bodies = new_bodies
    # To reduce overhead, store state snapshots every 100 iterations:
    if iteration % 100 == 0:
        history.append(bodies[1:].copy())
        dt_history.append(dt_used)
        new_bodies_delta = rkdp45_fixed(bodies_delta, dt_used)
        bodies_delta = new_bodies_delta
        history_delta.append(bodies_delta[1:].copy())
    iteration += 1

# Compute the Lyapunov exponent per unit time (seconds)
n = len(history)
total_time = sum(dt_history)
deltas = [np.linalg.norm(history[i] - history_delta[i]) for i in range(n)]
growths = [deltas[i] / delta0 for i in range(n)]
exponent = sum(np.log(g) for g in growths) / total_time

print(f"Lyapunov Exponent: {exponent} per second")
with open("lyapunov_exponent.txt", "w") as f:
    f.write(f"{exponent}")
