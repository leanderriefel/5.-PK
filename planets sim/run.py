# This script sets up and integrates vanilla integrations of the Solar System.
# The simulation will run for approximately 1  month.

# Both REBOUND and REBOUNDx are required. 
# They can be installed with 'pip install rebound reboundx'
import rebound
import reboundx
from reboundx import constants


import numpy as np
twopi = 2.*np.pi
from sys import argv
planetnames = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']


try:
    # Original simulations have ids from -1150 to 1150
    n = int(argv[1]) # Simulation id.
except:
    print("Need the simulation id as a command line argument.")
    exit()

filename = 'solarsystem_'+("m" if n<0 else "p")+str(abs(n))+".bin"


# Attempt to restart the simulation
try:
    sim = rebound.Simulation(filename)
    sim.automateSimulationArchive(filename, step=int(1e4*twopi/sim.dt), deletefile=False)
    print("Continuing from {0:8.3f} Myrs".format(sim.t/twopi/1e6))
except:
    # Attempt to read in initial conditions from file
    try:
        sim = rebound.Simulation('ss.bin')
    except:
        # Fall back: get initial conditions from NASA Horizons
        sim = rebound.Simulation()
        sim.add(planetnames, date='2000-01-01 12:00')
        sim.save('ss.bin') # Store to file for reuse.

    # Perturb initial simulation by moving Mercury's x coordinate a tiny bit.
    au = 149597870700000 # Number of milimeters in 1 AU.
    dx = (0.38 * n)/au
    sim.particles[1].x += dx

    # We move to the centre of mass frame.
    sim.move_to_com()
    # We choose the WHFast implementation of the Wisdom-Holman integrator
    # with a modified kernel an 17th order symplectic correctors. 
    sim.integrator = 'whckl'
    # We set the timestep of almost exactly 8 days. The sqrt(65) ensures we have a transcendental number.
    sim.dt = np.sqrt(65)*twopi/365.25       
    # The following settings are important. If you are new to REBOUND, read through the
    # Advanced WHFast tutorial to understand what's going on. 
    sim.ri_whfast.safe_mode = 0                 # combines symplectic correctors and drift steps at beginning and end of timesteps.
    sim.ri_whfast.keep_unsynchronized = True    # allows for bit-by-bit reproducibility
    # In case there is a close encounter or an ejection from the system, the simulation will stop.
    sim.exit_min_distance = 3e-3 # Distance between the Earth and Moon.
    sim.exit_max_distance = 1000. # Unlikely that the planet is bound.
    # Setup the output frequency for the SimulationArchive.
    # Note that we set up the output interval as a fixed number of timesteps (an integer) and
    # not a fixed time (a floating point number) to avoid rounding issues and unequal sampling
    sim.automateSimulationArchive(filename, step=int(1e4*twopi/sim.dt), deletefile=True)

# Regardless of whether we started the simulation from scratch or continue running from a SimulationArchive, 
# we need to setup the non-Newtonian potential using REBOUNDx.
rebx = reboundx.Extras(sim)
gr = rebx.load_force('gr_potential')
rebx.add_force(gr)
gr.params['c'] = constants.C

# Finally, we run the integration. 
# The simulation will exit if an escape or encounter occurs 
try:
    sim.integrate(5e9*twopi, exact_finish_time=False)
except rebound.Escape as esc:
    print(esc)
except rebound.Encounter as enc:
    print(enc)
