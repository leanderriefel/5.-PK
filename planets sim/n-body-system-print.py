import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numba as nb
import time
import json
import threading
import queue
from numba import prange

G = 1  # Gravitational constant

running = True

# -------------------------------------------------------------------------
G = 6.6743e-11
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
DT = 86400.0  # time step in seconds for Verlet
# -------------------------------------------------------------------------


@nb.jit(fastmath=True, parallel=True)
def solve_kepler(M, e, tol=1e-10, max_iter=5):
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E


@nb.njit(fastmath=True, parallel=True)
def verlet_batch(bodies, dt, nsteps):
    n = bodies.shape[0]
    for _ in range(nsteps):
        # First kick
        for i in prange(n):
            ax = 0.0
            ay = 0.0
            az = 0.0
            for j in range(n):
                if i == j:
                    continue
                dx = bodies[j, 1] - bodies[i, 1]
                dy = bodies[j, 2] - bodies[i, 2]
                dz = bodies[j, 3] - bodies[i, 3]
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                inv = G * bodies[j, 0] / (dist * dist * dist)
                ax += dx * inv
                ay += dy * inv
                az += dz * inv
            bodies[i, 4] += 0.5 * dt * ax
            bodies[i, 5] += 0.5 * dt * ay
            bodies[i, 6] += 0.5 * dt * az
        # Drift
        for i in prange(n):
            bodies[i, 1] += dt * bodies[i, 4]
            bodies[i, 2] += dt * bodies[i, 5]
            bodies[i, 3] += dt * bodies[i, 6]
        # Second kick
        for i in prange(n):
            ax = 0.0
            ay = 0.0
            az = 0.0
            for j in range(n):
                if i == j:
                    continue
                dx = bodies[j, 1] - bodies[i, 1]
                dy = bodies[j, 2] - bodies[i, 2]
                dz = bodies[j, 3] - bodies[i, 3]
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                inv = G * bodies[j, 0] / (dist * dist * dist)
                ax += dx * inv
                ay += dy * inv
                az += dz * inv
            bodies[i, 4] += 0.5 * dt * ax
            bodies[i, 5] += 0.5 * dt * ay
            bodies[i, 6] += 0.5 * dt * az
    return bodies


STEP_INTERVAL = 100_000  # number of steps per log
log_queue = queue.Queue()
logfile_name = f"data-{time.time()}.jsonl"


def log_writer():
    with open(logfile_name, "a") as logfile:
        while True:
            entry = log_queue.get()
            if entry is None:
                break
            logfile.write(json.dumps(entry) + "\n")
            logfile.flush()


log_thread = threading.Thread(target=log_writer, daemon=True)
log_thread.start()

print("Press CTRL+C to quit.")
try:
    steps = 0  # in millions
    elapsed_time = 0.0  # in years
    while True:
        bodies = verlet_batch(bodies, DT, STEP_INTERVAL)
        steps += STEP_INTERVAL / 1_000_000
        elapsed_time += (DT * STEP_INTERVAL) / 31536000
        log_entry = {
            "step": steps,
            "time": elapsed_time,
            "state": bodies.tolist(),
        }
        log_queue.put(log_entry)
        print(f"Step {round(steps, 2)} Million, Time {round(elapsed_time, 2)} Years")
except KeyboardInterrupt:
    print("Quitting...")
    log_queue.put(None)
    log_thread.join()
