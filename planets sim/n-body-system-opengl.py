import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import json

# --- OpenGL & Context Setup ---
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glfw

if not glfw.init():
    raise Exception("GLFW could not be initialized!")
context_window = glfw.create_window(100, 100, "Hidden Context", None, None)
if not context_window:
    glfw.terminate()
    raise Exception("GLFW window could not be created!")
glfw.make_context_current(context_window)

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
bodies_orig[:, 0] *= 1 / 1.988e30  # kg -> solar masses
bodies_orig[:, 1:4] *= 1
bodies_orig[:, 4:7] *= 1
bodies_orig = bodies_orig.astype(np.float32)

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
dt = np.float32(1e0)  # time step in days (adjust for stability)
axis_limit = 16.0  # in AU (set appropriately for your system)
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

# --- Compute Shader (Single Precision version) ---
# We use float and vec4 instead of double/dvec4.
# We also set a local_size_x of 64.
compute_shader_source = r"""
#version 430
layout(local_size_x = 64) in;

struct Body {
    float mass;
    vec4 pos; // only xyz used
    vec4 vel;
};

layout(std430, binding = 0) buffer BodiesIn {
    Body bodies[];
};

layout(std430, binding = 1) buffer BodiesOut {
    Body new_bodies[];
};

layout(std430, binding = 2) buffer ParamBlock {
    float dt;
    float G;
    float numBodies_f;
};

const float a21 = 1.0/5.0;
const float a31 = 3.0/40.0;
const float a32 = 9.0/40.0;
const float a41 = 44.0/45.0;
const float a42 = -56.0/15.0;
const float a43 = 32.0/9.0;
const float a51 = 19372.0/6561.0;
const float a52 = -25360.0/2187.0;
const float a53 = 64448.0/6561.0;
const float a54 = -212.0/729.0;
const float a61 = 9017.0/3168.0;
const float a62 = -355.0/33.0;
const float a63 = 46732.0/5247.0;
const float a64 = 49.0/176.0;
const float a65 = -5103.0/18656.0;
const float a71 = 35.0/384.0;
const float a72 = 0.0;
const float a73 = 500.0/1113.0;
const float a74 = 125.0/192.0;
const float a75 = -2187.0/6784.0;
const float a76 = 11.0/84.0;
const float b1  = 35.0/384.0;
const float b2  = 0.0;
const float b3  = 500.0/1113.0;
const float b4  = 125.0/192.0;
const float b5  = -2187.0/6784.0;
const float b6  = 11.0/84.0;

Body computeAcceleration(Body b, int numBodies, float G) {
    Body acc;
    acc.mass = 0.0;
    acc.pos = b.vel;
    acc.vel = vec4(0.0);
    for (int j = 0; j < numBodies; j++) {
        if(j == int(gl_GlobalInvocationID.x)) continue;
        vec3 r = bodies[j].pos.xyz - b.pos.xyz;
        float dist = length(r);
        float invDist3 = 1.0 / (dist * dist * dist + 1e-6);
        acc.vel.xyz += G * bodies[j].mass * r * invDist3;
    }
    return acc;
}

void main(){
    uint i = gl_GlobalInvocationID.x;
    int numBodies = int(numBodies_f);
    if(i >= uint(numBodies)) return;
    
    float mydt = dt;
    float myG  = G;
    
    Body b = bodies[i];
    Body k1 = computeAcceleration(b, numBodies, myG);
    
    Body temp;
    temp.mass = b.mass;
    temp.pos = b.pos + mydt * a21 * k1.pos;
    temp.vel = b.vel + mydt * a21 * k1.vel;
    Body k2 = computeAcceleration(temp, numBodies, myG);
    
    temp.pos = b.pos + mydt * (a31 * k1.pos + a32 * k2.pos);
    temp.vel = b.vel + mydt * (a31 * k1.vel + a32 * k2.vel);
    Body k3 = computeAcceleration(temp, numBodies, myG);
    
    temp.pos = b.pos + mydt * (a41 * k1.pos + a42 * k2.pos + a43 * k3.pos);
    temp.vel = b.vel + mydt * (a41 * k1.vel + a42 * k2.vel + a43 * k3.vel);
    Body k4 = computeAcceleration(temp, numBodies, myG);
    
    temp.pos = b.pos + mydt * (a51 * k1.pos + a52 * k2.pos + a53 * k3.pos + a54 * k4.pos);
    temp.vel = b.vel + mydt * (a51 * k1.vel + a52 * k2.vel + a53 * k3.vel + a54 * k4.vel);
    Body k5 = computeAcceleration(temp, numBodies, myG);
    
    temp.pos = b.pos + mydt * (a61 * k1.pos + a62 * k2.pos + a63 * k3.pos + a64 * k4.pos + a65 * k5.pos);
    temp.vel = b.vel + mydt * (a61 * k1.vel + a62 * k2.vel + a63 * k3.vel + a64 * k4.vel + a65 * k5.vel);
    Body k6 = computeAcceleration(temp, numBodies, myG);
    
    temp.pos = b.pos + mydt * (a71 * k1.pos + a72 * k2.pos + a73 * k3.pos + a74 * k4.pos + a75 * k5.pos + a76 * k6.pos);
    temp.vel = b.vel + mydt * (a71 * k1.vel + a72 * k2.vel + a73 * k3.vel + a74 * k4.vel + a75 * k5.vel + a76 * k6.vel);
    Body k7 = computeAcceleration(temp, numBodies, myG);
    
    Body y5;
    y5.mass = b.mass;
    y5.pos = b.pos + mydt * (b1 * k1.pos + b2 * k2.pos + b3 * k3.pos +
                              b4 * k4.pos + b5 * k5.pos + b6 * k6.pos);
    y5.vel = b.vel + mydt * (b1 * k1.vel + b2 * k2.vel + b3 * k3.vel +
                              b4 * k4.vel + b5 * k5.vel + b6 * k6.vel);
    
    new_bodies[i] = y5;
}
""".strip()

shader_program = compileProgram(compileShader(compute_shader_source, GL_COMPUTE_SHADER))


def gpu_rkdp45_single(bodies_struct, dt):
    glfw.make_context_current(context_window)
    num_bodies = bodies_struct.shape[0]
    body_bytes = bodies_struct.tobytes()

    ssbo_in = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_in)
    glBufferData(GL_SHADER_STORAGE_BUFFER, len(body_bytes), body_bytes, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_in)

    ssbo_out = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_out)
    glBufferData(GL_SHADER_STORAGE_BUFFER, len(body_bytes), None, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_out)

    param_data = np.array([dt, G_new, float(num_bodies)], dtype=np.float32)
    param_bytes = param_data.tobytes()
    ssbo_param = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_param)
    glBufferData(GL_SHADER_STORAGE_BUFFER, len(param_bytes), param_bytes, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo_param)

    glUseProgram(shader_program)
    glDispatchCompute(num_bodies, 1, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_out)
    new_body_bytes = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, len(body_bytes))
    # Our struct is 48 bytes per body = 12 floats.
    new_bodies = np.frombuffer(new_body_bytes, dtype=np.float32).reshape(num_bodies, 12)

    glDeleteBuffers(1, [ssbo_in])
    glDeleteBuffers(1, [ssbo_out])
    glDeleteBuffers(1, [ssbo_param])

    new_bodies_struct = np.empty(num_bodies, dtype=body_dtype)
    new_bodies_struct["mass"] = new_bodies[:, 0]
    # new_bodies columns:
    #   0: mass, 1-3: pad, 4-7: pos, 8-11: vel.
    new_bodies_struct["pos"][:, :3] = new_bodies[:, 4:7]
    new_bodies_struct["pos"][:, 3] = 0.0
    new_bodies_struct["vel"][:, :3] = new_bodies[:, 8:11]
    new_bodies_struct["vel"][:, 3] = 0.0
    return new_bodies_struct


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


def update(frame):
    global bodies_struct, sim_steps, display_index
    new_struct = gpu_rkdp45_single(bodies_struct, dt)
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
    update_display(display_index)
    return planets + trails + [time_text]


ani = FuncAnimation(fig, update, interval=0, cache_frame_data=False)
plt.show()

glfw.terminate()
