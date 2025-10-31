# generate_traffic.py (REAL-WORLD CITIES VERSION)
# !/usr/bin/env python
"""
Script to generate SUMO scenarios from real-world OpenStreetMap (OSM) data.
It converts OSM files into SUMO networks, generates random traffic for them,
and creates all necessary configuration files.
"""
import os
import re
import subprocess
import sys

# --- CITY DEFINITIONS ---
# List of city names that correspond to the .osm files in the `osm_data` folder.
# The script will generate a full scenario for each city in this list.
CITIES_TO_PROCESS = [
    'delhi',
    'mumbai',
    'guwahati',
    'bangaluru',
    'paris',
    'london',
    'nyc',
    'tokyo'
]
# Directory where your raw .osm files are stored.
OSM_DATA_DIR = "osm_data"
# Directory where the final SUMO scenarios will be saved.
SCENARIO_OUTPUT_DIR = "sumo_scenario"

try:
    from config import INNER_STEPS
except ImportError:
    print("Warning: Could not import INNER_STEPS from config.py. Using default of 100.")
    INNER_STEPS = 100


def create_uav_definitions_file(net_file_path, definitions_file_path):
    """Creates an additional-file for SUMO defining the UAV type and a dummy route."""
    try:
        with open(net_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        match = re.search(r'<edge id="([^:]\w*)"', content)
        if not match:
            print(f"✗ ERROR: Could not find any valid edges in {net_file_path}.")
            return False
        valid_edge_id = match.group(1)
        definitions_xml = f"""<additional>
    <vType id="UAV_TYPE" guiShape="aircraft" color="1,0,0" length="3" collisionModel="simple" impatience="1"/>
    <route id="dummy_route" edges="{valid_edge_id}"/>
</additional>"""
        with open(definitions_file_path, 'w') as f:
            f.write(definitions_xml)
        return True
    except Exception as e:
        print(f"✗ ERROR: Failed to create UAV definitions file: {e}")
        return False


def generate_real_world_scenarios():
    """Converts OSM files and generates complete SUMO scenarios."""
    # --- 1. Check for SUMO_HOME ---
    if 'SUMO_HOME' not in os.environ:
        print("ERROR: SUMO_HOME environment variable is not set.")
        return False

    sumo_tools_path = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(sumo_tools_path)
    netconvert_path = os.path.join(os.environ['SUMO_HOME'], 'bin', 'netconvert')

    os.makedirs(SCENARIO_OUTPUT_DIR, exist_ok=True)
    typemap_path = os.path.abspath("typemap.xml")
    if not os.path.exists(typemap_path):
        print(f"✗ CRITICAL ERROR: The typemap file was not found at {typemap_path}. Make sure 'typemap.xml' is in your main project directory.")
        return False
    # --- 2. Loop Through Each City ---
    for city in CITIES_TO_PROCESS:
        print(f"\n--- Processing Scenario: {city} ---")
        osm_file = os.path.join(OSM_DATA_DIR, f"{city}.osm")
        if not os.path.exists(osm_file):
            print(f"  ✗ WARNING: OSM file not found at '{osm_file}'. Skipping.")
            continue

        # --- 3. Define file paths for this city ---
        net_file = os.path.join(SCENARIO_OUTPUT_DIR, f"{city}.net.xml")
        route_file = os.path.join(SCENARIO_OUTPUT_DIR, f"{city}.rou.xml")
        config_file = os.path.join(SCENARIO_OUTPUT_DIR, f"{city}.sumocfg")
        definitions_file = os.path.join(SCENARIO_OUTPUT_DIR, f"{city}.add.xml")

        # --- 4. Convert OSM to SUMO Network using netconvert ---
        print(f"  -> Converting OSM to network file: {net_file}")
        # Note: Real-world data is complex. This command is simplified.
        # For production use, you might need a type-map to filter for car-only roads.
        try:
            subprocess.run([
                netconvert_path,
                f'--osm-files={osm_file}',
                f'--type-files={typemap_path}',
                '-o', net_file,
                '--geometry.remove',
                '--ramps.guess',
                '--junctions.join',
                '--tls.guess-signals',
                '--tls.discard-simple',
                '--no-internal-links',
                '--remove-edges.by-vclass', 'pedestrian,bicycle,rail,tram',  # Filter out non-vehicle edges
                '--keep-edges.by-vclass', 'passenger,delivery',  # Keep only vehicle roads
                '--ignore-errors.edge-type',  # Suppress edge type warnings
            ], check=True, capture_output=True, text=True, encoding='utf-8', timeout=300)

            # Check if the network file was actually created
            if not os.path.exists(net_file):
                print(f"  ✗ Network conversion failed: output file not created")
                continue

        except subprocess.TimeoutExpired as e:
            print(f"  ✗ Network conversion timed out for {city}")
            continue
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Network conversion failed for {city}: {e.stderr}")
            continue

        # --- 5. Generate Random Vehicle Trips ---
        print(f"  -> Generating route file: {route_file}")
        try:
            subprocess.run([
                sys.executable, os.path.join(sumo_tools_path, 'randomTrips.py'),
                '-n', net_file,
                '-r', route_file,
                '-e', str(INNER_STEPS),
                '-p', '1.0',  # Standard traffic density
                '--vehicle-class', 'passenger',
                '--min-distance', '1000'
            ], check=True, capture_output=True, text=True, encoding='utf-8', timeout=300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  ✗ Route generation failed for {city}: {e.stderr}")
            continue

        # --- 6. Create UAV definitions and final SUMO config ---
        if not create_uav_definitions_file(net_file, definitions_file):
            continue

        config_xml = f"""<configuration>
    <input>
        <net-file value="{city}.net.xml"/>
        <route-files value="{city}.rou.xml"/>
        <additional-files value="{city}.add.xml"/>
    </input>
    <time><begin value="0"/></time>
    <report><no-step-log value="true"/></report>
</configuration>"""
        with open(config_file, 'w') as f:
            f.write(config_xml)

        print(f"  ✓ Successfully generated all files for {city}.")

    print("\n✓ Real-world scenario generation process complete.")
    return True


if __name__ == "__main__":
    generate_real_world_scenarios()
