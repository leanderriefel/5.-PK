import numpy as np
import moderngl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import json

G = 6.6743e-11
dt = 10000
AU_to_m = 1.496e11
AUday_to_ms = AU_to_m / 86400

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
        rows.append(
            [
                body["mass"],
                body["position"]["x"],
                body["position"]["y"],
                body["position"]["z"],
                body["velocity"]["vx"],
                body["velocity"]["vy"],
                body["velocity"]["vz"],
            ]
        )

bodies = np.array(rows, dtype=np.float32)
bodies[:, 1:4] *= AU_to_m
bodies[:, 4:7] *= AUday_to_ms

history = [[bodies[i, 1:4].copy()] for i in range(bodies.shape[0])]

fig = plt.figure(figsize=(10, 10), dpi=100)
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.1)
ax.set_xlim(-3e12, 3e12)
ax.set_ylim(-3e12, 3e12)
ax.set_zlim(-3e12, 3e12)
ax.set_box_aspect((1, 1, 1))
planets = [
    ax.plot(
        [], [], [], "o", color=f"C{i}", markersize=10, label=f"Planet {i+1}", alpha=0.8
    )[0]
    for i in range(bodies.shape[0])
]
trails = [
    ax.plot([], [], [], "-", color=f"C{i}", alpha=0.3, linewidth=1)[0]
    for i in range(bodies.shape[0])
]

ctx = moderngl.create_standalone_context()

compute_shader_source = """
#version 430

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) buffer BodyBuffer { float data[]; };

uniform float dt;
uniform int n_bodies;
uniform float G;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(n_bodies)) return;

    int index = int(i) * 7;

    float mass = data[index];

    vec3 pos = vec3(data[index+1], data[index+2], data[index+3]);
    vec3 vel = vec3(data[index+4], data[index+5], data[index+6]);
    vec3 acc = vec3(0.0);

    for (int j = 0; j < n_bodies; j++){
        if (j == int(i)) continue;

        int j_index = j * 7;
        float mass_j = data[j_index];
        vec3 pos_j = vec3(data[j_index+1], data[j_index+2], data[j_index+3]);
        vec3 diff = pos_j - pos;
        float dist = length(diff);

        acc += G * mass_j * diff / (dist * dist * dist);
    }

    vel += acc * dt;
    pos += vel * dt;

    data[index+1] = pos.x;
    data[index+2] = pos.y;
    data[index+3] = pos.z;
    data[index+4] = vel.x;
    data[index+5] = vel.y;
    data[index+6] = vel.z;
}
"""


prog = ctx.compute_shader(compute_shader_source)
prog["dt"].value = dt
prog["n_bodies"].value = bodies.shape[0]
prog["G"].value = G


def update(frame):
    global bodies, history
    ssbo = ctx.buffer(bodies.tobytes())
    ssbo.bind_to_storage_buffer(binding=0)
    prog.run(group_x=bodies.shape[0])
    bodies = np.frombuffer(ssbo.read(), dtype="f4").reshape(-1, 7)
    for i in range(bodies.shape[0]):
        history[i].append(bodies[i, 1:4].copy())
        planets[i].set_data([bodies[i, 1]], [bodies[i, 2]])
        planets[i].set_3d_properties([bodies[i, 3]])
        trail_x = [pos[0] for pos in history[i]]
        trail_y = [pos[1] for pos in history[i]]
        trail_z = [pos[2] for pos in history[i]]
        trails[i].set_data(trail_x, trail_y)
        trails[i].set_3d_properties(trail_z)


def reset(event):
    global history
    history = [[bodies[i, 1:4].copy()] for i in range(bodies.shape[0])]
    for p in planets:
        p.set_data([], [])
        p.set_3d_properties([])
    for t in trails:
        t.set_data([], [])
        t.set_3d_properties([])
    plt.draw()


reset_button = Button(plt.axes([0.8, 0.025, 0.1, 0.04]), "Reset")
reset_button.on_clicked(reset)
ani = FuncAnimation(fig, update, interval=10, cache_frame_data=False)
plt.show()
