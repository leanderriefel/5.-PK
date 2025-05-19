import numpy as np
import json
import time
import threading
import queue
import pyopencl as cl
import keyboard

# --- OpenCL Setup ---
platforms = cl.get_platforms()
gpu_devices = platforms[0].get_devices(cl.device_type.GPU)
ctx = cl.Context(devices=gpu_devices)
cl_queue = cl.CommandQueue(ctx)

# --- Constants ---
G_SI = 6.6743e-11
G_new = G_SI * (86400**2) / ((1.496e11) ** 3) * 1.988e30  # AU^3 / (solar mass * day^2)
G_new = np.float64(G_new)
mass_conv = 1 / 1.988e30

# --- Load Data ---
with open("solar_system.json", "r") as f:
    data = json.load(f)

labels = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
rows = []
for name in labels:
    if name in data:
        body = data[name]
        row = [body["mass"] * mass_conv, body["position"]["x"], body["position"]["y"], body["position"]["z"], 
               body["velocity"]["vx"], body["velocity"]["vy"], body["velocity"]["vz"]]
        rows.append(row)

bodies_orig = np.array(rows, dtype=np.float64)
n_bodies = bodies_orig.shape[0]

# pad to double4 (w = 0)
pos = np.hstack((bodies_orig[:, 1:4], np.zeros((n_bodies, 1), dtype=np.float64)))
vel = np.hstack((bodies_orig[:, 4:7], np.zeros((n_bodies, 1), dtype=np.float64)))
mass = bodies_orig[:, 0].astype(np.float64)

# --- Allocate OpenCL Buffers ---
mf = cl.mem_flags
pos_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pos)
vel_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vel)
mass_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mass)

# --- OpenCL Kernel ---
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

    const double w0 = -0.117767998417887E+1;
    const double w1 = 0.235573213359357E+0;
    const double d0 = 0.0;
    const double d1 = -0.132127026654696E+1;

    __local double4 shared_pos[64];
    __local double shared_mass[64];
    
    int lid = get_local_id(0);
    int group_size = get_local_size(0);
    int num_groups = (n + group_size - 1) / group_size;

    double4 p = pos[i];
    double4 v = vel[i];
    double4 a = (double4)(0.0);

    // Yoshida coefficients for 4th order
    const double c[4] = {
        0.675603595979828817023843900325,
        -0.175603595979828817023843900325,
        -0.175603595979828817023843900325,
        0.675603595979828817023843900325
    };
    
    const double d[4] = {
        0.1756035959798288170238439003252,
        0.5178904303638644295446602361904,
        -0.1756035959798288170238439003252,
        0.4821095696361355704553397638096
    };

    // Main integration loop
    for (int step = 0; step < 4; step++) {
        // Reset acceleration
        a = (double4)(0.0);
        
        // Compute gravitational forces using tiled approach
        for (int tile = 0; tile < num_groups; tile++) {
            // Load chunk of positions into local memory
            int offset = tile * group_size;
            if (offset + lid < n) {
                shared_pos[lid] = pos[offset + lid];
                shared_mass[lid] = mass[offset + lid];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // Compute interactions with all bodies in this tile
            #pragma unroll 4
            for (int j = 0; j < group_size && offset + j < n; j++) {
                if (offset + j != i) {
                    double4 r = shared_pos[j] - p;
                    double distSqr = dot(r.xyz, r.xyz) + 1e-12;
                    double invDist = rsqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    a.xyz += G * shared_mass[j] * r.xyz * invDist3;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Update velocity and position using Yoshida coefficients
        v += d[step] * dt * a;
        p += c[step] * dt * v;
    }

    // Store final position and velocity
    pos[i] = p;
    vel[i] = v;
}
"""

program = cl.Program(ctx, opencl_kernel).build()

# --- Compute Step Function ---
def gpu_yoshida_step(dt):
    global pos, vel
    global_size = ((n_bodies + 63) // 64) * 64
    program.yoshida4(cl_queue, (global_size,), (64,), pos_buf, vel_buf, mass_buf, 
                    np.float64(dt), G_new, np.int32(n_bodies))
    cl.enqueue_barrier(cl_queue)
    cl.enqueue_copy(cl_queue, pos, pos_buf)
    cl.enqueue_copy(cl_queue, vel, vel_buf)

# --- Async Writer ---
output_filename = "simulation_output_opencl.jsonl"
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

writer_thread = threading.Thread(target=file_writer, args=(output_filename, write_queue, stop_event))
writer_thread.start()

# --- Main Loop ---
dt = 1.0
sim_steps = 0
sim_time = 0.0
paused = False

print("Simulation running. Press 't' to toggle pause, 'q' to quit.")
try:
    while True:
        if keyboard.is_pressed("q"):
            print("Quitting...")
            break
        if keyboard.is_pressed("t"):
            paused = not paused
            print("Paused." if paused else "Resumed.")
            time.sleep(0.3)

        if paused:
            time.sleep(0.1)
            continue

        gpu_yoshida_step(dt)
        sim_steps += 1
        sim_time += dt

        state = {
            "step": sim_steps,
            "sim_time_days": sim_time,
            "bodies": [{
                "name": labels[i],
                "pos": pos[i, :3].tolist(),
                "vel": vel[i, :3].tolist()
            } for i in range(n_bodies)]
        }
        write_queue.put(json.dumps(state))

        if sim_steps % 1000 == 0:
            print(f"Step {sim_steps}, Time {sim_time:.2f} days")

except KeyboardInterrupt:
    print("Interrupted.")
finally:
    stop_event.set()
    writer_thread.join()
    print(f"Output written to {output_filename}")
