import numpy as np
import json
import time
import threading
import queue
import pyopencl as cl

# ---------------------- Configuration ----------------------
DT = 1.0  # days per integrator sub -step
INNER_STEPS = 1_000_000  # sub -steps executed inside one GPU kernel launch
OUTPUT_BATCHES = 100  # write one snapshot every N launches
OUTPUT_FILE = "simulation_output_opencl.jsonl"

# ---------------------- OpenCL setup -----------------------
platforms = cl.get_platforms()
if not platforms:
    raise RuntimeError("No OpenCL platforms found.")
ctx = cl.Context(devices=platforms[0].get_devices(device_type=cl.device_type.GPU))
cl_queue = cl.CommandQueue(ctx)

# ---------------------- Constants --------------------------
G_SI = 6.6743e-11  # m^3 kg^-1 s^-2
G_new = G_SI * (86400**2) / ((1.496e11) ** 3) * 1.988e30  # AU^3 / (M☉ · day^2)
G_new = np.float64(G_new)
mass_conv = 1 / 1.988e30  # kg → M☉

# ---------------------- Load data --------------------------
with open("solar_system.json", "r") as f:
    data = json.load(f)

labels = [
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
    "36 Atalante",
]
rows = []
for name in labels:
    if name in data:
        body = data[name]
        rows.append(
            [
                body["mass"] * mass_conv,
                body["position"]["x"],
                body["position"]["y"],
                body["position"]["z"],
                body["velocity"]["vx"],
                body["velocity"]["vy"],
                body["velocity"]["vz"],
            ]
        )

bodies_orig = np.array(rows, dtype=np.float64)
if bodies_orig.size == 0:
    raise ValueError("No bodies loaded from solar_system.json")

n_bodies = bodies_orig.shape[0]

# Pad to double4 (w component = 0)
pos = np.hstack((bodies_orig[:, 1:4], np.zeros((n_bodies, 1), dtype=np.float64)))
vel = np.hstack((bodies_orig[:, 4:7], np.zeros((n_bodies, 1), dtype=np.float64)))
mass = bodies_orig[:, 0].astype(np.float64)

# ---------------------- OpenCL buffers ---------------------
mf = cl.mem_flags
pos_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pos)
vel_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vel)
# Mass is small and read -only — put it in constant memory via a separate arg
mass_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR | mf.HOST_NO_ACCESS, hostbuf=mass)

# ---------------------- OpenCL kernel ----------------------
opencl_kernel = rf"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// max 64 bodies (fits in one work -group & 4 KiB local memory)
#define MAX_BODIES 64

__kernel void yoshida4_batch_local(
    __global double4 *restrict pos_g,
    __global double4 *restrict vel_g,
    __constant double *restrict mass_c,
    const double dt,
    const double G,
    const int n,
    const int inner_steps)
{{
    const int i = get_local_id(0);   // body index within work -group (one body per work -item)
    if (i >= n) return;

    __local double4 pos_l[MAX_BODIES];

    // Yoshida 4 coefficients
    const double a[4] = {{ 0.5153528374311229364, -0.085782019412973646, 0.4415830236164665242, 0.1288461583653841854 }};
    const double b[4] = {{ 0.1344961992774310892, -0.2248198030794208058, 0.7563200005156682911, 0.3340036032863214255 }};

    // Load initial positions & velocities into registers / local memory
    double4 p = pos_g[i];
    double4 v = vel_g[i];
    pos_l[i] = p;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = 0; step < inner_steps; ++step) {{
        for (int stage = 0; stage < 4; ++stage) {{
            double4 a_vec = (double4)(0.0);
            for (int j = 0; j < n; ++j) {{
                double4 r = pos_l[j] - p;
                double distSqr = dot(r.xyz, r.xyz) + 1e-12;
                double invDist = rsqrt(distSqr);
                double invDist3 = invDist / distSqr;  // 1/r^3
                a_vec.xyz += G * mass_c[j] * r.xyz * invDist3;
            }}
            v += b[stage] * dt * a_vec;
            p += a[stage] * dt * v;
        }}
        pos_l[i] = p;  // update local copy for next sub -step
        barrier(CLK_LOCAL_MEM_FENCE);
    }}

    // write back final state to global memory
    pos_g[i] = p;
    vel_g[i] = v;
}}
"""

# Build with optional fast -math
program = cl.Program(ctx, opencl_kernel).build(options=["-cl-fast-relaxed-math"])

# ---------------- GPU helper functions ---------------------
global_size = (max(64, 1 << (n_bodies - 1).bit_length()),)  # at least one warp -aligned group
local_size = (global_size[0],)


def gpu_advance(inner_steps: int):
    program.yoshida4_batch_local(
        cl_queue,
        global_size,
        local_size,
        pos_buf,
        vel_buf,
        mass_buf,
        np.float64(DT),
        G_new,
        np.int32(n_bodies),
        np.int32(inner_steps),
        wait_for=None,
    )


# ---------------- Asynchronous read -back pipeline ----------
# We overlap GPU compute with host JSON serialization.
class Snapshot:
    __slots__ = ("event", "pos", "vel", "sim_days", "launch")

    def __init__(self, event, pos_view, vel_view, sim_days: float, launch: int):
        self.event = event
        self.pos = pos_view  # memoryview of pinned host buf
        self.vel = vel_view
        self.sim_days = sim_days
        self.launch = launch


write_queue: "queue.Queue[Snapshot]" = queue.Queue(maxsize=32)
stop_event = threading.Event()


def file_writer(filename: str, wq: "queue.Queue[Snapshot]", stop_evt: threading.Event):
    with open(filename, "w") as f:
        while not stop_evt.is_set() or not wq.empty():
            try:
                snap = wq.get(timeout=0.1)
                snap.event.wait()  # ensure copy completed
                state = {
                    "launch": snap.launch,
                    "sim_time_days": snap.sim_days,
                    "bodies": [
                        {
                            "name": labels[i],
                            "pos": snap.pos[i, :3].tolist(),
                            "vel": snap.vel[i, :3].tolist(),
                        }
                        for i in range(n_bodies)
                    ],
                }
                f.write(json.dumps(state) + "\n")
                wq.task_done()
            except queue.Empty:
                continue


writer_thread = threading.Thread(target=file_writer, args=(OUTPUT_FILE, write_queue, stop_event), daemon=True)
writer_thread.start()

# ------------------- Main loop -----------------------------
launches = 0
sim_days = 0.0
TOTAL_STEPS_PER_LAUNCH = INNER_STEPS

# Host -side pinned buffers for async copy
pos_host = np.empty_like(pos)
vel_host = np.empty_like(vel)

print("Running… press Ctrl+C to stop.")


def enqueue_snapshot(sim_days: float, launches: int):
    # Asynchronous copy (non -blocking)
    evt1 = cl.enqueue_copy(cl_queue, pos_host, pos_buf, is_blocking=False)
    evt2 = cl.enqueue_copy(cl_queue, vel_host, vel_buf, is_blocking=False, wait_for=[evt1])
    snap = Snapshot(evt2, pos_host.copy(), vel_host.copy(), sim_days, launches)  # copy views to isolate from next overwrite
    write_queue.put(snap)


try:
    while True:
        t0 = time.perf_counter()
        gpu_advance(TOTAL_STEPS_PER_LAUNCH)
        launches += 1
        sim_days += DT * TOTAL_STEPS_PER_LAUNCH

        if launches % OUTPUT_BATCHES == 0:
            enqueue_snapshot(sim_days, launches)
            yrs = sim_days / 365.25
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"{yrs:,.2f} yrs simulated — launch {launches} — {dt_ms:.1f} ms")
except KeyboardInterrupt:
    print("\nInterrupted — writing final snapshot…")
finally:
    enqueue_snapshot(sim_days, launches)
    stop_event.set()
    writer_thread.join()
    print(f"Output written to {OUTPUT_FILE}")
