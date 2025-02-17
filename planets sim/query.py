import os
import re
import json
from astroquery.jplhorizons import Horizons
from astropy.time import Time

# Define a valid time span (start must be earlier than stop)
epochs = {"start": "2000-01-01T12:00:00", "stop": "2000-01-01T12:01:00", "step": "1m"}

# Dictionary of major body Horizons IDs
body_ids = {
    "Sun": "10",
    "Mercury": "199",
    "Venus": "299",
    "Earth": "399",
    "Mars": "499",
    "Jupiter": "599",
    "Saturn": "699",
    "Uranus": "799",
    "Neptune": "899",
    "Pluto": "999",
}

# Fallback GM values (in km^3/s^2) for bodies that might not return GM
fallback_GM = {
    "Sun": 1.32712440041939e11,  # Sun's GM
    "Earth": 3.986004418e05,  # Earth's GM
    "Pluto": 975.5,  # Pluto's GM (from published values)
}

# Gravitational constant in km^3/(kg s^2)
G = 6.6743e-11

# Dictionary to store all results
results = {}

# Loop over each body
for name, body_id in body_ids.items():
    try:
        # Create a Horizons object using a dictionary for epochs
        obj = Horizons(id=body_id, location="@0", epochs=epochs)
        # Query state vectors (returns an Astropy table)
        vec_table = obj.vectors()
        # Use the first row (since we only need one snapshot)
        state = vec_table[0]
        # Get the raw text header using the asynchronous method
        response = obj.vectors_async()
        raw_text = response.text
    except Exception as e:
        print(f"Error querying {name}: {e}")
        continue

    # Try to extract GM from the raw text header using a regex
    match = re.search(r"GM\s*\(km\^3/s\^2\)\s*=\s*([\d\.E+-]+)", raw_text)
    if match:
        gm_value = float(match.group(1))
    else:
        # Use fallback value if available
        gm_value = fallback_GM.get(name, None)

    if gm_value is not None:
        mass = gm_value / G  # mass in kg
    else:
        mass = None

    # Store the relevant data in our results dictionary
    results[name] = {
        "GM": gm_value,
        "mass": mass,
        "position": {
            "x": float(state["x"]),
            "y": float(state["y"]),
            "z": float(state["z"]),
        },
        "velocity": {
            "vx": float(state["vx"]),
            "vy": float(state["vy"]),
            "vz": float(state["vz"]),
        },
    }

# Write the complete dictionary to a JSON file
with open("solar_system.json", "w") as f:
    json.dump(results, f, indent=4)

print("Saved solar system data to solar_system.json")
