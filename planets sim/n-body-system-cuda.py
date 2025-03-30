import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import cuda
import math
import json

# Constants and unit conversions
G = 6.6743e-11  # gravitational constant in SI units
AU_to_m = 1.496e11
AUday_to_ms = AU_to_m / 86400

# Load solar system data from JSON
with open("solar_system.json", "r") as f:
    data = json.load(f)

order = [
    "Sun", "Mercury", "Venus", "Earth", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
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

# Create bodies array and convert positions/velocities to SI units.
# Each row: [mass, x, y, z, vx, vy, vz]
bodies = np.array(rows, dtype=np.float64)
bodies[:, 1:4] *= AU_to_m
bodies[:, 4:7] *= AUday_to_ms

# Prepare history for visualization trails
history = [[bodies[i, 1:4].copy()] for i in range(bodies.shape[0])]

# Set up 3D plot
fig = plt.figure(figsize=(10, 10), dpi=100)
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.1)
ax.set_xlim(-3e12, 3e12)
ax.set_ylim(-3e12, 3e12)
ax.set_zlim(-3e12, 3e12)
ax.set_box_aspect((1, 1, 1))
planets = [
    ax.plot([], [], [], "o", color=f"C{i}", markersize=10, label=f"Planet {i+1}", alpha=0.8)[0]
    for i in range(bodies.shape[0])
]
trails = [
    ax.plot([], [], [], "-", color=f"C{i}", alpha=0.3, linewidth=1)[0]
    for i in range(bodies.shape[0])
]

# --- CUDA RK4 Integration ---

# Device function to compute gravitational acceleration on body i
@cuda.jit(device=True)
def compute_acceleration(i, state, n, G):
    ax = 0.0
    ay = 0.0
    az = 0.0
    x = state[i, 1]
    y = state[i, 2]
    z = state[i, 3]
    for j in range(n):
        if i != j:
            dx = state[j, 1] - x
            dy = state[j, 2] - y
            dz = state[j, 3] - z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist > 0.0:
                factor = G * state[j, 0] / (dist*dist*dist)
                ax += dx * factor
                ay += dy * factor
                az += dz * factor
    return ax, ay, az

# RK4 kernel: We update the state for each body using four stages.
@cuda.jit
def rk4_kernel(bodies, temp, dt, G):
    i = cuda.grid(1)
    n = bodies.shape[0]
    if i < n:
        # Save original state locally
        mass = bodies[i, 0]
        x0 = bodies[i, 1]
        y0 = bodies[i, 2]
        z0 = bodies[i, 3]
        vx0 = bodies[i, 4]
        vy0 = bodies[i, 5]
        vz0 = bodies[i, 6]

        # --- k1 ---
        ax1, ay1, az1 = compute_acceleration(i, bodies, n, G)
        k1x = vx0
        k1y = vy0
        k1z = vz0
        k1vx = ax1
        k1vy = ay1
        k1vz = az1

        # --- Stage for k2: state + dt/2 * k1 ---
        x2 = x0 + dt/2 * k1x
        y2 = y0 + dt/2 * k1y
        z2 = z0 + dt/2 * k1z
        vx2 = vx0 + dt/2 * k1vx
        vy2 = vy0 + dt/2 * k1vy
        vz2 = vz0 + dt/2 * k1vz

        # Write k2 state to temp array for all bodies
        temp[i, 0] = mass
        temp[i, 1] = x2
        temp[i, 2] = y2
        temp[i, 3] = z2
        temp[i, 4] = vx2
        temp[i, 5] = vy2
        temp[i, 6] = vz2
    cuda.syncthreads()

    if i < n:
        # --- k2 ---
        ax2, ay2, az2 = compute_acceleration(i, temp, n, G)
        # k2 is evaluated at the intermediate state (already stored in temp)
        k2x = temp[i, 4]
        k2y = temp[i, 5]
        k2z = temp[i, 6]
        k2vx = ax2
        k2vy = ay2
        k2vz = az2

        # --- Stage for k3: state + dt/2 * k2 ---
        x3 = x0 + dt/2 * k2x
        y3 = y0 + dt/2 * k2y
        z3 = z0 + dt/2 * k2z
        vx3 = vx0 + dt/2 * k2vx
        vy3 = vy0 + dt/2 * k2vy
        vz3 = vz0 + dt/2 * k2vz

        temp[i, 1] = x3
        temp[i, 2] = y3
        temp[i, 3] = z3
        temp[i, 4] = vx3
        temp[i, 5] = vy3
        temp[i, 6] = vz3
    cuda.syncthreads()

    if i < n:
        # --- k3 ---
        ax3, ay3, az3 = compute_acceleration(i, temp, n, G)
        k3x = vx3
        k3y = vy3
        k3z = vz3
        k3vx = ax3
        k3vy = ay3
        k3vz = az3

        # --- Stage for k4: state + dt * k3 ---
        x4 = x0 + dt * k3x
        y4 = y0 + dt * k3y
        z4 = z0 + dt * k3z
        vx4 = vx0 + dt * k3vx
        vy4 = vy0 + dt * k3vy
        vz4 = vz0 + dt * k3vz

        temp[i, 1] = x4
        temp[i, 2] = y4
        temp[i, 3] = z4
        temp[i, 4] = vx4
        temp[i, 5] = vy4
        temp[i, 6] = vz4
    cuda.syncthreads()

    if i < n:
        # --- k4 ---
        ax4, ay4, az4 = compute_acceleration(i, temp, n, G)
        k4x = vx4
        k4y = vy4
        k4z = vz4
        k4vx = ax4
        k4vy = ay4
        k4vz = az4

        # --- Combine increments to update state ---
        bodies[i, 1] = x0 + dt/6.0 * (k1x + 2*k2x + 2*k3x + k4x)
        bodies[i, 2] = y0 + dt/6.0 * (k1y + 2*k2y + 2*k3y + k4y)
        bodies[i, 3] = z0 + dt/6.0 * (k1z + 2*k2z + 2*k3z + k4z)
        bodies[i, 4] = vx0 + dt/6.0 * (k1vx + 2*k2vx + 2*k3vx + k4vx)
        bodies[i, 5] = vy0 + dt/6.0 * (k1vy + 2*k2vy + 2*k3vy + k4vy)
        bodies[i, 6] = vz0 + dt/6.0 * (k1vz + 2*k2vz + 2*k3vz + k4vz)

# Transfer initial bodies array to the GPU
d_bodies = cuda.to_device(bodies)
d_temp = cuda.device_array_like(bodies)
dt = 10000.0  # time step in seconds

def update(frame):
    global d_bodies, d_temp, bodies, history
    threadsperblock = 32
    blockspergrid = (bodies.shape[0] + (threadsperblock - 1)) // threadsperblock
    rk4_kernel[blockspergrid, threadsperblock](d_bodies, d_temp, dt, G)
    d_bodies.copy_to_host(bodies)
    
    # Update history and set new positions for visualization
    for i in range(bodies.shape[0]):
        history[i].append(bodies[i, 1:4].copy())
        planets[i].set_data(np.array([bodies[i, 1]]), np.array([bodies[i, 2]]))
        planets[i].set_3d_properties(np.array([bodies[i, 3]]))
        trail_x = [pos[0] for pos in history[i]]
        trail_y = [pos[1] for pos in history[i]]
        trail_z = [pos[2] for pos in history[i]]
        trails[i].set_data(np.array(trail_x), np.array(trail_y))
        trails[i].set_3d_properties(np.array(trail_z))

ani = FuncAnimation(fig, update, interval=1, cache_frame_data=False)
plt.show()
