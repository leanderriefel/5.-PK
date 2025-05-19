import numpy as np
import json
import time
import threading
import queue
import keyboard
import cupy as cp

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
pos = cp.asarray(np.hstack((bodies_orig[:, 1:4], np.zeros((n_bodies, 1), dtype=np.float64))))
vel = cp.asarray(np.hstack((bodies_orig[:, 4:7], np.zeros((n_bodies, 1), dtype=np.float64))))
mass = cp.asarray(bodies_orig[:, 0].astype(np.float64))

# --- CUDA Kernel ---
cuda_kernel = cp.RawKernel(r'''
extern "C" __global__
void yoshida4(double4* pos, double4* vel, double* mass, double dt, double G, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    __shared__ double4 shared_pos[64];
    __shared__ double shared_mass[64];
    
    int tid = threadIdx.x;
    int num_blocks = (n + blockDim.x - 1) / blockDim.x;

    double4 p = pos[i];
    double4 v = vel[i];
    double4 a;

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
        a.x = a.y = a.z = a.w = 0.0;
        
        // Compute gravitational forces using tiled approach
        for (int tile = 0; tile < num_blocks; tile++) {
            // Load chunk of positions into shared memory
            int offset = tile * blockDim.x;
            if (offset + tid < n) {
                shared_pos[tid] = pos[offset + tid];
                shared_mass[tid] = mass[offset + tid];
            }
            __syncthreads();

            // Compute interactions with all bodies in this tile
            #pragma unroll 4
            for (int j = 0; j < blockDim.x && offset + j < n; j++) {
                if (offset + j != i) {
                    double4 r;                    r.x = shared_pos[j].x - p.x;
                    r.y = shared_pos[j].y - p.y;
                    r.z = shared_pos[j].z - p.z;
                    double distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 1e-8;// Increased softening parameter
                    double dist = sqrt(distSqr);
                    double invDist3 = 1.0 / (dist * distSqr);
                    double factor = G * shared_mass[j] * invDist3;
                    a.x += factor * r.x;
                    a.y += factor * r.y;
                    a.z += factor * r.z;
                }
            }
            __syncthreads();
        }

        // Update velocity and position using Yoshida coefficients
        v.x += d[step] * dt * a.x;
        v.y += d[step] * dt * a.y;
        v.z += d[step] * dt * a.z;
        
        p.x += c[step] * dt * v.x;
        p.y += c[step] * dt * v.y;
        p.z += c[step] * dt * v.z;
    }

    // Store final position and velocity
    pos[i] = p;
    vel[i] = v;
}
''', 'yoshida4')

# --- Compute Step Function ---
def gpu_yoshida_step(dt):
    global pos, vel
    block_size = 64
    grid_size = (n_bodies + block_size - 1) // block_size
    cuda_kernel((grid_size,), (block_size,),
               (pos, vel, mass, cp.float64(dt), cp.float64(G_new), cp.int32(n_bodies)))

# --- Async Writer ---
output_filename = "simulation_output_cuda.jsonl"
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

        # Get data from GPU for output
        pos_cpu = cp.asnumpy(pos)
        vel_cpu = cp.asnumpy(vel)
        
        state = {
            "step": sim_steps,
            "sim_time_days": sim_time,
            "bodies": [{
                "name": labels[i],
                "pos": pos_cpu[i, :3].tolist(),
                "vel": vel_cpu[i, :3].tolist()
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
