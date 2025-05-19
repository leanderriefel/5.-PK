import time
import numpy as np
import keyboard
import threading, queue
import json
import pyopencl as cl

platforms = cl.get_platforms()
gpu_devices = platforms[0].get_devices(cl.device_type.GPU)
ctx = cl.Context(devices=gpu_devices)
cl_queue = cl.CommandQueue(ctx)

# --- Define conversion factors to rescale to AU, days, solar masses ---
# 1 AU = 1.496e11 m, 1 day = 86400 s, 1 solar mass = 1.988e30 kg.
# mass_conv = 1 / 1.988e30  # convert kg to solar masses
# pos_conv = 1 / 1.496e11  # convert meters to AU
# vel_conv = 86400 / 1.496e11  # convert m/s to AU/day

# In these units the gravitational constant becomes:
# G_new = G * (86400^2) / (1.496e11^3) * 1.988e30
G_SI = 6.6743e-11
G_new = G_SI * (86400**2) / ((1.496e11) ** 3) * 1.988e30  # ~2.96e-4
G_new = np.float32(G_new)

# --- Define conversion factors ---
# Data is assumed to be in AU (positions) and AU/day (velocities)
# Only masses are converted from kg to solar masses.
mass_conv = 1 / 1.988e30  # kg -> solar masses

# Gravitational constant in AU^3/(solar mass * day^2)
G_SI = 6.6743e-11
G_new = G_SI * (86400**2) / ((1.496e11) ** 3) * 1.988e30  # ~2.96e-4
G_new = np.float32(G_new)

# --- Load Solar-System Data ---
with open("solar_system.json", "r") as f:
    data = json.load(f)

labels = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
rows = []
for name in labels:
    if name in data:
        body = data[name]
        mass = body["mass"]
        pos = body["position"]
        vel = body["velocity"]
        row = [mass, pos["x"], pos["y"], pos["z"], vel["vx"], vel["vy"], vel["vz"]]
        rows.append(row)
# Create an array (n,7) in float64 (SI units)
bodies_orig = np.array(rows, dtype=np.float64)
# Convert masses (positions and velocities remain unchanged)
bodies_orig[:, 0] *= mass_conv
bodies_orig = bodies_orig.astype(np.float32)

print("Initial values (mass, pos, vel):")
for i, lab in enumerate(labels):
    print(f"{lab}: {bodies_orig[i]}")

# --- Create structured array matching std430 layout (single precision) ---
# For single precision std430:
#   - float: 4 bytes; a vec4 must be 16-byte aligned.
# Our Body struct:
#   offset 0: float mass (4 bytes)
#   offset 4: 12 bytes padding
#   offset 16: vec4 pos (16 bytes) – use only xyz
#   offset 32: vec4 vel (16 bytes) – use only xyz
# Total = 48 bytes.
body_dtype = np.dtype({"names": ["mass", "pad", "pos", "vel"], "formats": [np.float32, "V12", (np.float32, 4), (np.float32, 4)], "offsets": [0, 4, 16, 32], "itemsize": 48})
n_bodies = bodies_orig.shape[0]
bodies_struct = np.zeros(n_bodies, dtype=body_dtype)
for i in range(n_bodies):
    bodies_struct["mass"][i] = bodies_orig[i, 0]
    bodies_struct["pos"][i, :3] = bodies_orig[i, 1:4]
    bodies_struct["pos"][i, 3] = 0.0
    bodies_struct["vel"][i, :3] = bodies_orig[i, 4:7]
    bodies_struct["vel"][i, 3] = 0.0

opencl_kernel = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void yoshida4(
    __global double4* restrict pos,
    __global double4* restrict vel,
    __global double* restrict mass,
    const double dt,
    const double G,
    const int n
) {
    int i = get_global_id(0);
    if (i >= n) return;

    const double a[4] = {
        0.5153528374311229364,
       -0.085782019412973646,
        0.4415830236164665242,
        0.1288461583653841854
    };

    const double b[4] = {
        0.1344961992774310892,
       -0.2248198030794208058,
        0.7563200005156682911,
        0.3340036032863214255
    };

    double4 a_vec;
    double4 p = pos[i];
    double4 v = vel[i];

    __local double4 shared_pos[64];
    __local double shared_mass[64];

    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    for (int step = 0; step < 4; step++) {
        a_vec = (double4)(0.0);
        for (int offset = 0; offset < n; offset += group_size) {
            int j = offset + lid;
            if (j < n) {
                shared_pos[lid] = pos[j];
                shared_mass[lid] = mass[j];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll 11
            for (int k = 0; k < group_size && (offset + k) < n; k++) {
                if (i == offset + k) continue;
                double4 r = shared_pos[k] - p;
                double distSqr = dot(r.xyz, r.xyz) + 1e-9;
                double invDist3 = 1.0 / (sqrt(distSqr) * distSqr);
                a_vec.xyz += G * shared_mass[k] * r.xyz * invDist3;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        v += b[step] * dt * a_vec;
        p += a[step] * dt * v;
    }

    pos[i] = p;
    vel[i] = v;
}
"""


def compute_energy(bodies_struct):
    total_kinetic = 0.0
    total_potential = 0.0
    n = bodies_struct.shape[0]

    for i in range(n):
        m_i = bodies_struct["mass"][i]
        v_i = np.linalg.norm(bodies_struct["vel"][i, :3])
        total_kinetic += 0.5 * m_i * v_i**2

        for j in range(i + 1, n):
            m_j = bodies_struct["mass"][j]
            r_ij = np.linalg.norm(bodies_struct["pos"][i, :3] - bodies_struct["pos"][j, :3])
            total_potential -= G_new * m_i * m_j / r_ij

    return total_kinetic + total_potential


def gpu_yoshida4_opencl(bodies_struct, dt):
    num = bodies_struct.shape[0]
    masses = bodies_struct["mass"].astype(np.float64)
    pos = bodies_struct["pos"][:, :3].astype(np.float64)
    vel = bodies_struct["vel"][:, :3].astype(np.float64)

    # pad to double4 (w=0)
    pos = np.hstack((pos, np.zeros((num, 1), dtype=np.float64)))
    vel = np.hstack((vel, np.zeros((num, 1), dtype=np.float64)))

    mf = cl.mem_flags
    pos_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pos)
    vel_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vel)
    mass_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=masses)

    program = cl.Program(ctx, opencl_kernel).build()

    group_size = 64
    global_size = ((num + group_size - 1) // group_size) * group_size
    program.yoshida4(cl_queue, (global_size,), (group_size,), pos_buf, vel_buf, mass_buf, np.float64(dt), np.float64(G_new), np.int32(num))

    cl.enqueue_barrier(cl_queue)
    cl.enqueue_copy(cl_queue, pos, pos_buf)
    cl.enqueue_copy(cl_queue, vel, vel_buf)

    result = np.empty(num, dtype=body_dtype)
    result["mass"] = masses.astype(np.float32)
    result["pos"][:, :3] = pos[:, :3].astype(np.float32)
    result["pos"][:, 3] = 0.0
    result["vel"][:, :3] = vel[:, :3].astype(np.float32)
    result["vel"][:, 3] = 0.0

    return result


# --- Main Simulation Loop (Headless, JSONL output) ---
dt = np.float32(1e0)
sim_steps = 0
sim_time = 0.0
output_filename = "simulation_output.jsonl"
output_file = open(output_filename, "w")
paused = False

# Create a queue for asynchronous file writes (adjust maxsize as needed)
write_queue = queue.Queue(maxsize=1000)
stop_event = threading.Event()


def file_writer(filename, write_queue, stop_event):
    with open(filename, "w") as f:
        while not stop_event.is_set() or not write_queue.empty():
            try:
                line = write_queue.get(timeout=0.1)
                f.write(line + "\n")
                write_queue.task_done()
            except queue.Empty:
                continue


# Start the background file writer thread
writer_thread = threading.Thread(target=file_writer, args=(output_filename, write_queue, stop_event))
writer_thread.start()

print("Starting simulation. Press 't' to toggle pause/unpause, and 'q' to quit.")

try:
    while True:
        # Check for quit request:
        if keyboard.is_pressed("q"):
            print("Quitting simulation ('q' pressed).")
            break

        # Toggle pause/unpause when 't' is pressed.
        if keyboard.is_pressed("t"):
            paused = not paused
            print("Paused." if paused else "Unpaused.")
            # Wait a bit to debounce
            time.sleep(0.3)

        # If paused, simply wait
        if paused:
            time.sleep(0.1)
            continue

        # Run one simulation step.
        bodies_struct = gpu_yoshida4_opencl(bodies_struct, dt)
        sim_steps += 1
        sim_time += dt

        # Prepare state dict for output.
        state = {"step": sim_steps, "sim_time_days": float(sim_time), "bodies": []}
        for i in range(n_bodies):
            pos = bodies_struct["pos"][i, :3].tolist()
            vel = bodies_struct["vel"][i, :3].tolist()
            state["bodies"].append({"name": labels[i], "pos": pos, "vel": vel})
        write_queue.put(json.dumps(state))

        if sim_steps % 100 == 0:
            print(f"Step {sim_steps}, Time: {sim_time:.2f} days")

except KeyboardInterrupt:
    print("Simulation terminated via KeyboardInterrupt.")
finally:
    output_file.close()
    stop_event.set()
    writer_thread.join()
    print("Simulation data written to simulation_output_opencl.jsonl")
