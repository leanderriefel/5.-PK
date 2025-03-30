import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import numba as nb
import json

# Konstanten und Umrechnungen
G = 6.6743e-11
AU_to_m = 1.496e11  # Astronomische Einheit in Meter
AUday_to_ms = AU_to_m / 86400  # AU/Tag in m/s

# Solar-System-Daten laden
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

# History zur Speicherung der Zustände (für Slider und Trails)
history = [[bodies[i, 1:4].copy()] for i in range(bodies.shape[0])]

# Plot Setup (angepasster unterer Rand für Slider/Buttons)
fig = plt.figure(figsize=(25, 25), dpi=100)
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

ax.set_xlim(-3e12, 3e12)
ax.set_ylim(-3e12, 3e12)
ax.set_zlim(-3e12, 3e12)
ax.set_box_aspect((1, 1, 1))

# Planeten (Punkte) und Trails (Linien) zeichnen
planets = [ax.plot([], [], [], "o", color=f"C{i}", markersize=10, alpha=0.8, label=order[i])[0] for i in range(bodies.shape[0])]
trails = [ax.plot([], [], [], "-", color=f"C{i}", alpha=0.3, linewidth=1)[0] for i in range(bodies.shape[0])]

# Zeitanzeige (als Text im Plot)
time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

# Legende
ax.legend(loc="upper right")


# --- Simulationsfunktionen ---
@nb.njit()
def acceleration(bodies):
    n = bodies.shape[0]
    acc = np.zeros((n, 7))
    for i in range(n):
        acc[i, 0] = 0.0  # Massen-Ableitung = 0
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
            acc[i, 4] += G * bodies[j, 0] * dx / (dist**3)
            acc[i, 5] += G * bodies[j, 0] * dy / (dist**3)
            acc[i, 6] += G * bodies[j, 0] * dz / (dist**3)
    return acc


@nb.njit()
def rkdp45(bodies, dt):
    # c-Werte (Zeitanteile)
    c2, c3, c4, c5, c6, c7 = 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0

    # Butcher-Tabellen-Koeffizienten:
    a21 = 1 / 5
    a31, a32 = 3 / 40, 9 / 40
    a41, a42, a43 = 44 / 45, -56 / 15, 32 / 9
    a51, a52, a53, a54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
    a61, a62, a63, a64, a65 = 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656
    a71, a72, a73, a74, a75, a76 = 35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84

    # Koeffizienten für 5. Ordnung (5th-Order Lösung):
    b1, b2, b3, b4, b5, b6 = 35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
    # b7 = 0.0 implizit

    # Koeffizienten für die 4. Ordnung (eingebettete Lösung):
    b1s, b2s, b3s, b4s, b5s, b6s, b7s = 5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40

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
    # Ausgabe zur Kontrolle (kann kommentiert werden)
    # print("dt:", dt, "error:", error)
    if error > 1e-1:
        dt = 0.99 * dt * (1e-1 / error) ** 0.01
        return rkdp45(bodies, dt)
    return y5, dt


# Globale Variablen für Simulation
dt = 1e4  # Zeitschritt (in Sekunden)
sim_steps = 0  # Anzahl der bisher berechneten Schritte
sim_time = {0: 0}  # Zeit in Sekunden
paused = False  # Pause-Status
display_index = 0  # Angezeigter History-Eintrag
slider_active = False  # Slider-Aktivität
max_trail_length = 1e4  # Maximale Länge der Trails


# --- Callback-Funktionen für Buttons und Slider ---
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
    steps = min(steps, sim_steps)
    for i in range(bodies.shape[0]):
        pos = history[i][steps]
        planets[i].set_data([pos[0]], [pos[1]])
        planets[i].set_3d_properties([pos[2]])

        start_idx = max(0, steps - max_trail_length)
        trail_data = history[i][start_idx : steps + 1]
        if len(trail_data) > 0:
            trail_x = [p[0] for p in trail_data]
            trail_y = [p[1] for p in trail_data]
            trail_z = [p[2] for p in trail_data]
            trails[i].set_data(trail_x, trail_y)
            trails[i].set_3d_properties(trail_z)

    sim_time_days = sim_time.get(steps, 0) / 86400
    time_text.set_text(f"Time: {sim_time_days:.2f} days")


# --- Widgets: Slider und Buttons ---
ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, "History", 0, 0, valinit=0, valstep=1)
slider.on_changed(slider_update)

ax_pause = plt.axes([0.15, 0.02, 0.15, 0.04])
btn_pause = Button(ax_pause, "Pause/Start")
btn_pause.on_clicked(pause_start)

ax_jump = plt.axes([0.75, 0.02, 0.15, 0.04])
btn_jump = Button(ax_jump, "Jump Forward")
btn_jump.on_clicked(jump_forward)


# --- Update-Funktion der Animation ---
def update(frame):
    global bodies, sim_steps, display_index
    bodies_new, corrected_dt = rkdp45(bodies, dt)
    bodies[:] = bodies_new
    sim_steps += 1
    sim_time[sim_steps] = sim_time.get(sim_steps - 1, 0) + corrected_dt

    for i in range(bodies.shape[0]):
        history[i].append(bodies[i, 1:4].copy())

    # Update Slider-Bereich dynamisch
    slider.valmax = sim_steps
    slider.ax.set_xlim(slider.valmin, slider.valmax)

    # Automatische Slider-Bewegung nur wenn nicht pausiert
    if not paused and not slider_active:
        display_index = sim_steps
        slider.set_val(display_index)

    update_display(display_index)
    return planets + trails + [time_text]


ani = FuncAnimation(fig, update, interval=10, cache_frame_data=False)
plt.show()
