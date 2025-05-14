import numpy as np
import json
import time
import threading
import queue
import pyopencl as cl

# ---------------------- Configuration ----------------------
DT = 4.0  # days per WH step  (≤ 6 d keeps Mercury resolved)
INNER_STEPS = 200_000  # WH steps per GPU launch
OUTPUT_BATCHES = 10  # write one snapshot every N launches
OUTPUT_FILE = "simulation_output_whfast.jsonl"

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

# pad to double4 (w component=0)
pos = np.hstack((bodies_orig[:, 1:4], np.zeros((n_bodies, 1))))
vel = np.hstack((bodies_orig[:, 4:7], np.zeros((n_bodies, 1))))
mass = np.ascontiguousarray(bodies_orig[:, 0], dtype=np.float64)

# ---------------------- OpenCL buffers ---------------------
mf = cl.mem_flags
pos_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pos)
vel_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vel)
mass_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR | mf.HOST_NO_ACCESS, hostbuf=mass)

# ---------------------- OpenCL kernel ----------------------
opencl_kernel = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define MAX_BODIES 64

// Universelle Keplergleichung
double kepler_U(double dt, double r0, double vr0, double alpha, double mu)
{
    const double tol = 1e-14;
    double x = sqrt(mu) * fabs(alpha) * dt;

    // Newton-Raphson-Verfahren (5 Iterationen)
    for(int it=0; it<5; ++it) {
        double x2 = x*x;
        double z  = alpha * x2; // z = alpha * x^2

        // Stumpff-Funktionen
        double C, S;
        if (fabs(z) < 1e-7) { // wenn |z| sehr klein -> C = ca. 1/2 und S = ca. 1/6
            C = 0.5 - z/24.0 + z*z/720.0;
            S = (1.0/6.0) - z/120.0 + z*z/5040.0;
        } else { // sonst allgemeine Lösung
            if (z > 0) { // Elliptical or circular
                double sqrt_z = sqrt(z);
                C = (1.0 - cos(sqrt_z))/z;
                S = (sqrt_z - sin(sqrt_z))/pow(sqrt_z, 3.0);
            } else { // z < 0, Hyperbolic
                double sqrt_mz = sqrt(-z); // -z is positive
                C = (cosh(sqrt_mz) - 1.0)/(-z);
                S = (sinh(sqrt_mz) - sqrt_mz)/pow(sqrt_mz, 3.0);
            }
        }


        // https://orbital-mechanics.space/time-since-periapsis-and-keplers-equation/universal-variables.html#equation-time-since-periapsis-and-keplers-equation-universal-variables-8
        double F = r0*vr0/sqrt(mu)*x2*C + (1.0 - alpha*r0)*x2*x*S + r0*x - sqrt(mu)*dt;

        // https://orbital-mechanics.space/time-since-periapsis-and-keplers-equation/universal-variables.html#equation-time-since-periapsis-and-keplers-equation-universal-variables-11
        double dFdx = r0*vr0/sqrt(mu)*x*(1 - alpha*x2*S) + (1.0 - alpha*r0)*x2*C + r0;

        // Newton-Schritt
        double dx = -F/dFdx;
        x += dx;
        if (fabs(dx) < tol) break; // Abbruch wenn |dx| < Toleranz
    }

    return x;
}

__kernel void whfast_batch(
    __global double4 *restrict pos_g, // Positionen
    __global double4 *restrict vel_g, // Geschwindigkeiten
    __constant double *restrict mass_c, // Massen
    const double dt, // Zeitinkrement
    const double G, // Gravitationskonstante
    const int n, // Anzahl der Körper
    const int inner_steps // Anzahl der WH-Schritte
)
{
    const int i = get_local_id(0); // aktueller Körper (Thread-ID)
    if(i >= n) return; // wenn außerhalb der Anzahl der Körper

    // lokale Speicherung für Positionen, Geschwindigkeiten und Massen
    __local double4 pos_l[MAX_BODIES];
    __local double4 vel_l[MAX_BODIES];
    __local double   mass_l[MAX_BODIES];

    pos_l[i]  = pos_g[i];
    vel_l[i]  = vel_g[i];
    mass_l[i] = mass_c[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    const double mu = G * mass_l[0]; // mu = G * M_0 (Gravitationskonstante * Masse der Sonne)

    // WHFast-Schritte
    for(int step=0; step<inner_steps; ++step) {
        double3 acc = (double3)(0.0); // Beschleunigung
        double4 p_i = pos_l[i]; // aktuelle Position
        if(i>0) { // wenn nicht die Sonne (i=0)
            for(int j=1;j<n;++j) { // alle anderen Körper
                if(j==i) continue; // überspringen wenn selbst
                double3 r = pos_l[j].xyz - p_i.xyz; // Vektor Körper j -> Körper i
                double d2 = dot(r,r) + 1e-12; // Quadrat der Entfernung (1e-12 verhindert Teilen durch 0)
                double invd  = rsqrt(d2); // Inverse der Entfernung
                double invd3 = invd / d2; // Inverse der Entfernung hoch 3
                acc += G * mass_l[j] * r * invd3; // resultierende Beschleunigung
            }
        }
        vel_l[i].xyz += 0.5 * dt * acc; // halbe Geschwindigkeit für ersten Kick
        barrier(CLK_LOCAL_MEM_FENCE);

        // Drift um die Sonne (Sonnenposition bleibt fix)
        if(i>0) {
            double3 r0 = pos_l[i].xyz; // aktuelle Position
            double3 v0 = vel_l[i].xyz; // aktuelle Geschwindigkeit
            double r0mag = length(r0); // Abstand zur Sonne
            
            // Corrected alpha calculation
            // alpha = 1/a = 2/r - v^2/mu
            double mu_sun = G * mass_l[0]; // mu = G * M_sun
            double v0_sq = dot(v0, v0);
            double alpha = 2.0 / r0mag - v0_sq / mu_sun;

            double vr0 = dot(r0,v0);
            double x = kepler_U(dt, r0mag, vr0, alpha, mu_sun); // Universelle Variable
            double z = alpha * x*x;

            // Stumpff-Funktionen
            double C, S;
            if(fabs(z) < 1e-7) { // Taylor series for z approx 0
                C = 0.5 - z/24.0 + z*z/720.0;
                S = (1.0/6.0) - z/120.0 + z*z/5040.0;
            } else {
                if (z > 0) { // Elliptical or circular
                    double sqrt_z = sqrt(z);
                    C = (1.0 - cos(sqrt_z))/z;
                    S = (sqrt_z - sin(sqrt_z))/pow(sqrt_z, 3.0);
                } else { // z < 0, Hyperbolic
                    double sqrt_mz = sqrt(-z); // -z is positive
                    C = (cosh(sqrt_mz) - 1.0)/(-z);
                    S = (sinh(sqrt_mz) - sqrt_mz)/pow(sqrt_mz, 3.0);
                }
            }

            // Lagrange-Koeffizienten
            double f = 1 - (x*x)*C/r0mag;
            double g = dt - (x*x*x)*S/sqrt(mu_sun);

            double3 r = f*r0 + g*v0; // neue Position
            double rmag = length(r); // Abstand zur Sonne
            double fdot = -sqrt(mu_sun)*x*S/(r0mag*rmag); // f-Ableitung
            double gdot = 1 - (x*x)*C/rmag; // g-Ableitung
            double3 v = fdot*r0 + gdot*v0; // neue Geschwindigkeit

            pos_l[i].xyz = r;
            vel_l[i].xyz = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ---- second half-kick ----
        acc = (double3)(0.0);
        p_i = pos_l[i];
        if(i>0) {
            for(int j=1;j<n;++j) {
                if(j==i) continue;
                double3 r = pos_l[j].xyz - p_i.xyz;
                double d2 = dot(r,r) + 1e-12;
                double invd  = rsqrt(d2);
                double invd3 = invd / d2;
                acc += G * mass_l[j] * r * invd3;
            }
        }
        vel_l[i].xyz += 0.5 * dt * acc;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    pos_g[i] = pos_l[i];
    vel_g[i] = vel_l[i];
}
"""

program = cl.Program(ctx, opencl_kernel).build(options=["-cl-fast-relaxed-math"])

# ---------------- GPU helper -------------------------------
global_size = (max(64, 1 << (n_bodies - 1).bit_length()),)
local_size = (global_size[0],)


def gpu_advance(inner_steps: int):
    program.whfast_batch(
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
    )


# ------------- async read‑back identical to prior ----------
class Snapshot:
    __slots__ = ("event", "pos", "vel", "sim_days", "launch")

    def __init__(self, ev, pos_v, vel_v, sd, l):
        self.event = ev
        self.pos = pos_v
        self.vel = vel_v
        self.sim_days = sd
        self.launch = l


write_queue: "queue.Queue[Snapshot]" = queue.Queue(maxsize=32)
stop_event = threading.Event()


def file_writer(fname, wq, stop_evt):
    with open(fname, "w") as f:
        while not stop_evt.is_set() or not wq.empty():
            try:
                s = wq.get(timeout=0.1)
                s.event.wait()
                state = {"launch": s.launch, "sim_time_days": s.sim_days, "bodies": [{"name": labels[i], "pos": s.pos[i, :3].tolist(), "vel": s.vel[i, :3].tolist()} for i in range(n_bodies)]}
                f.write(json.dumps(state) + "\n")
                wq.task_done()
            except queue.Empty:
                continue


threading.Thread(target=file_writer, args=(OUTPUT_FILE, write_queue, stop_event), daemon=True).start()

launches = 0
sim_days = 0.0
TOTAL_STEPS_PER_LAUNCH = INNER_STEPS
pos_host = np.empty_like(pos)
vel_host = np.empty_like(vel)
print("Running WHFast …  press Ctrl+C to stop.")


def enqueue_snapshot(sd, l):
    ev1 = cl.enqueue_copy(cl_queue, pos_host, pos_buf, is_blocking=False)
    ev2 = cl.enqueue_copy(cl_queue, vel_host, vel_buf, is_blocking=False, wait_for=[ev1])
    write_queue.put(Snapshot(ev2, pos_host.copy(), vel_host.copy(), sd, l))


try:
    while True:
        t0 = time.perf_counter()
        gpu_advance(TOTAL_STEPS_PER_LAUNCH)
        launches += 1
        sim_days += DT * TOTAL_STEPS_PER_LAUNCH
        if launches % OUTPUT_BATCHES == 0:
            enqueue_snapshot(sim_days, launches)
            yrs = sim_days / 365.25
            print(f"{yrs:,.2f} yrs — launch {launches}")
except KeyboardInterrupt:
    print("\nInterrupted - final snapshot …")
finally:
    enqueue_snapshot(sim_days, launches)
    stop_event.set()
    print(f"Wrote snapshots to {OUTPUT_FILE}")
