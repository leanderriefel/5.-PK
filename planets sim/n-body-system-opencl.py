import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import json
import pyopencl as cl

platforms = cl.get_platforms()
if not platforms:
    raise RuntimeError("No OpenCL platforms found.")
ctx = cl.Context(devices=platforms[0].get_devices(device_type=cl.device_type.GPU))
cl_queue = cl.CommandQueue(ctx)

# --- Define conversion factors to rescale to AU, days, solar masses ---
# 1 AU = 1.496e11 m, 1 day = 86400 s, 1 solar mass = 1.988e30 kg.
mass_conv = 1 / 1.988e30  # convert kg to solar masses
pos_conv = 1 / 1.496e11  # convert meters to AU
vel_conv = 86400 / 1.496e11  # convert m/s to AU/day

# In these units the gravitational constant becomes:
# G_new = G * (86400^2) / (1.496e11^3) * 1.988e30
G_SI = 6.6743e-11
G_new = G_SI * (86400**2) / ((1.496e11) ** 3) * 1.988e30  # ~2.96e-4
G_new = np.float64(G_new)

# --- Load Solar-System Data ---
with open("solar_system.json", "r") as f:
    data = json.load(f)

labels = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "36 Atalante"]
rows = []
for name in labels:
    if name in data:
        body = data[name]
        mass = body["mass"]
        pos = body["position"]
        vel = body["velocity"]
        row = [mass, pos["x"], pos["y"], pos["z"], vel["vx"], vel["vy"], vel["vz"]]
        rows.append(row)
# Original array in shape (n,7), float64 (SI units)
bodies_orig = np.array(rows, dtype=np.float64)
# Rescale: mass to solar masses, pos to AU, vel to AU/day.
bodies_orig[:, 0] *= mass_conv
bodies_orig[:, 1:4] *= 1.0
bodies_orig[:, 4:7] *= 1.0

# --- Create a structured array matching std430 layout for single precision ---
# In std430 for single precision:
#   float: 4 bytes, vec4: 16 bytes.
# We want our Body to have:
#   mass at offset 0 (4 bytes),
#   a pad of 12 bytes (so that the next vec4 starts at offset 16),
#   pos as vec4 (offset 16), vel as vec4 (offset 32).
# Total itemsize = 48 bytes.
body_dtype = np.dtype({"names": ["mass", "pad", "pos", "vel"], "formats": [np.float32, "V12", (np.float32, 4), (np.float32, 4)], "offsets": [0, 4, 16, 32], "itemsize": 48})
n_bodies = bodies_orig.shape[0]
bodies_struct = np.zeros(n_bodies, dtype=body_dtype)
for i in range(n_bodies):
    bodies_struct["mass"][i] = bodies_orig[i, 0]
    bodies_struct["pos"][i, :3] = bodies_orig[i, 1:4]
    bodies_struct["pos"][i, 3] = 0.0  # padding
    bodies_struct["vel"][i, :3] = bodies_orig[i, 4:7]
    bodies_struct["vel"][i, 3] = 0.0  # padding


# --- Simulation Globals ---
dt = np.float64(1.0)  # time step in days (adjust for stability)
axis_limit = 40.0 # in AU (set appropriately for your system)
sim_steps = 0
sim_time = {0: 0.0}  # time in days
paused = False
display_index = 0
slider_active = False
max_trail_length = 1e8

# History for trails (store only first 3 components of pos)
history = [[bodies_struct["pos"][i, :3].copy()] for i in range(n_bodies)]

# --- Plot Setup ---
fig = plt.figure(figsize=(25, 25), dpi=100)
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
ax.set_xlim(-axis_limit, axis_limit)
ax.set_ylim(-axis_limit, axis_limit)
ax.set_zlim(-axis_limit, axis_limit)
ax.set_box_aspect((1, 1, 1))
planets = [ax.plot([], [], [], "o", color=f"C{i}", markersize=10, alpha=0.8, label=labels[i])[0] for i in range(n_bodies)]
trails = [ax.plot([], [], [], "-", color=f"C{i}", alpha=0.3, linewidth=1)[0] for i in range(n_bodies)]
time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
ax.legend(loc="upper right")


pos = np.zeros((n_bodies, 4), dtype=np.float64)
vel = np.zeros((n_bodies, 4), dtype=np.float64)
pos[:, :3] = bodies_orig[:, 1:4]
vel[:, :3] = bodies_orig[:, 4:7]
mass = np.ascontiguousarray(bodies_orig[:, 0], dtype=np.float64)

mf = cl.mem_flags
pos_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pos)
vel_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vel)
mass_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR | mf.HOST_NO_ACCESS, hostbuf=mass)


opencl_kernel = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void verlet_batch(
    __global double4 *pos,      // (x,y,z,0)
    __global double4 *vel,      // (vx,vy,vz,0)
    __constant double *mass,    // masses in M_sun
    const double dt,
    const double G,
    const int    n)
{
    int i = get_global_id(0);
    if(i>=n) return;

    // calculate acceleration
    double3 ri = pos[i].xyz;
    double3 ai = (double3)(0.0);
    for(int j=0; j<n; ++j){
        if(j==i) continue;
        double3 rj = pos[j].xyz - ri;
        double d2 = dot(rj,rj) + 1e-12;
        double inv = rsqrt(d2);
        ai += G * mass[j] * rj * inv*inv*inv;
    }

    // first kick
    double3 vi = vel[i].xyz + 0.5*dt*ai;

    // drift
    ri += dt*vi;

    // recompute acceleration
    double3 ai2 = (double3)(0.0);
    for(int j=0; j<n; ++j){
        if(j==i) continue;
        double3 rj = pos[j].xyz - ri;
        double d2 = dot(rj,rj) + 1e-12;
        double inv = rsqrt(d2);
        ai2 += G * mass[j] * rj * inv*inv*inv;
    }

    // second kick
    vi += 0.5*dt*ai2;

    // write back
    pos[i] = (double4)(ri,0.0);
    vel[i] = (double4)(vi,0.0);
}
"""

program = cl.Program(ctx, opencl_kernel).build(options=["-cl-fast-relaxed-math"])

global_size = (n_bodies,)
local_size = (n_bodies,)

def gpu_verlet(bodies_struct, dt):
    program.verlet_batch(
        cl_queue,
        global_size,
        local_size,
        pos_buf,
        vel_buf,
        mass_buf,
        np.float64(dt),
        np.float64(G_new),
        np.int32(n_bodies),
    )


def pause_start(event):
    global paused
    paused = not paused
    if not paused:
        update(0)


def jump_forward(event):
    global display_index
    new_index = min(display_index + 1, sim_steps)
    display_index = new_index
    slider.set_val(display_index)


def slider_update(val):
    global display_index
    display_index = int(slider.val)
    update_display(display_index)
    fig.canvas.draw_idle()


def update_display(steps):
    steps = min(int(steps), sim_steps)
    for i in range(n_bodies):
        pos = history[i][steps]
        planets[i].set_data([pos[0]], [pos[1]])
        planets[i].set_3d_properties([pos[2]])
        start_idx = max(0, steps - int(max_trail_length))
        trail_data = history[i][start_idx : steps + 1]
        if len(trail_data) > 0:
            trail_x = [p[0] for p in trail_data]
            trail_y = [p[1] for p in trail_data]
            trail_z = [p[2] for p in trail_data]
            trails[i].set_data(trail_x, trail_y)
            trails[i].set_3d_properties(trail_z)
    time_days = sim_time.get(steps, 0)  # time is in days now.
    time_text.set_text(f"Time: {time_days:.2f} days")


ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, "History", 0, 0, valinit=0, valstep=1)
slider.on_changed(slider_update)

ax_pause = plt.axes([0.15, 0.02, 0.15, 0.04])
btn_pause = Button(ax_pause, "Pause/Start")
btn_pause.on_clicked(pause_start)

ax_jump = plt.axes([0.75, 0.02, 0.15, 0.04])
btn_jump = Button(ax_jump, "Jump Forward")
btn_jump.on_clicked(jump_forward)

pos_host = np.empty_like(pos)
vel_host = np.empty_like(vel)

def update(frame):
    global bodies_struct, sim_steps, display_index
    gpu_verlet(bodies_struct, dt)

    cl.enqueue_copy(cl_queue, pos_host, pos_buf, is_blocking=True)
    cl.enqueue_copy(cl_queue, vel_host, vel_buf, is_blocking=True)

    new_pos = pos_host.copy()
    new_vel = vel_host.copy()

    new_struct = bodies_struct.copy()
    for i in range(n_bodies):
        new_struct["pos"][i, :3] = new_pos[i][:3]
        new_struct["vel"][i, :3] = new_vel[i][:3]

    bodies_struct = new_struct

    sim_steps += 1
    sim_time[sim_steps] = sim_time.get(sim_steps - 1, 0) + dt
    for i in range(n_bodies):
        history[i].append(bodies_struct["pos"][i, :3].copy())

    slider.valmax = sim_steps
    slider.ax.set_xlim(slider.valmin, slider.valmax)
    if not paused:
        display_index = sim_steps
        slider.set_val(display_index)
        update_display(sim_steps)

    return planets + trails + [time_text]


ani = FuncAnimation(fig, update, interval=0, cache_frame_data=False)
plt.show()