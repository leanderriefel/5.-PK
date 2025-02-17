import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numba as nb

plt.rcParams["toolbar"] = "None"

G = 9.81  # Gravitational constant

class Body:
  def __init__(self, mass, position, velocity):
    self.mass = mass
    self.position = position
    self.velocity = velocity

v1 = 0.4127173212
v2 = 0.4811313628

bodies = [
  Body(1, np.array([-1, 0]), np.array([v1, v2])),
  Body(1, np.array([1, 0]), np.array([v1, v2])),
  Body(1, np.array([0, 0]), np.array([-2*v1, -2*v2])),
]

history = [[body.position.copy()] for body in bodies]

# Plot
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.subplots_adjust(bottom=0.1)

fig.set_facecolor("white")
ax.set_facecolor("white")
ax.spines[["bottom", "left", "top", "right"]].set_color("black")
ax.tick_params(axis="x", colors="black")
ax.tick_params(axis="y", colors="black")

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.grid(color="black", linewidth=0.2, alpha=0.4)
ax.set_box_aspect(1)

planets = [ax.plot([], [], "o", color=f"C{i}", markersize=10, label=f"Planet {i+1}", alpha=0.8)[0] for i in range(3)]
trails = [ax.plot([], [], "-", color=f"C{i}", alpha=0.3, linewidth=1)[0] for i in range(3)]

@nb.njit()
def acceleration(bodies):
  accelerations = np.zeros((len(bodies), 2))
  for i, body in enumerate(bodies):
    for j, other in enumerate(bodies):
      if i == j:
        continue
      r = other.position - body.position
      dist = np.sqrt(r[0]**2 + r[1]**2)
      accelerations[i] += G * other.mass * r / dist**3
  return accelerations