# generate_traffic.py
"""
Script to generate SUMO scenarios from real-world OpenStreetMap (OSM) data.
It converts OSM files into SUMO networks, generates random traffic for them,
and creates all necessary configuration files (including UAV definitions).
"""
import os
import re
import subprocess
import sys

# Adjust path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import INNER_STEPS
except ImportError:
    print("Warning: Could not import INNER_STEPS from config.py. Using default of 100.")
    INNER_STEPS = 100

# --- CITY DEFINITIONS ---
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

OSM_DATA_DIR = "osm_data"
SCENARIO_OUTPUT_DIR = "sumo_scenario"


def create_uav_definitions_file(net_file_path, definitions_file_path):
    """
    Creates an additional-file for SUMO defining the UAV type and a dummy route.
    It scans the .net.xml file to find a valid edge ID to ensure the route is valid.
    """
    try:
        with open(net_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Regex to find the first valid edge ID that isn't an internal edge (starting with :)
        match = re.search(r'<edge id="([^:]\w*)"', content)
        if not match:
            print(f"✗ ERROR: Could not find any valid edges in {net_file_path}.")
            return False

        valid_edge_id = match.group(1)

        definitions_xml = f"""<additional>
    <vType id="UAV_TYPE" guiShape="aircraft" color="0,255,0" length="3" minGap="0" maxSpeed="50" speedMode="0" collisionModel="simple" impatience="1"/>
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
        print("Please set it to your SUMO installation directory (e.g., C:\\Program Files (x86)\\Eclipse\\Sumo)")
        return False

    sumo_tools_path = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(sumo_tools_path)

    # Path to netconvert binary
    netconvert_bin = "netconvert"
    # On Windows it might be netconvert.exe, but normally valid in path if SUMO installed

    os.makedirs(SCENARIO_OUTPUT_DIR, exist_ok=True)

    # Typemap is needed to map OSM road types to SUMO properties
    typemap_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "sumo_scenario", "typemap.xml"))
    if not os.path.exists(typemap_path):
        print(f"✗ CRITICAL ERROR: 'typemap.xml' not found at {typemap_path}.")
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

        # --- 4. Convert OSM to SUMO Network ---
        print(f"  -> Converting OSM to network file...")
        try:
            # We use subprocess to call netconvert
            cmd = [
                netconvert_bin,
                f'--osm-files={osm_file}',
                f'--type-files={typemap_path}',
                '-o', net_file,
                '--geometry.remove',
                '--ramps.guess',
                '--junctions.join',
                '--tls.guess-signals',
                '--tls.discard-simple',
                '--no-internal-links',
                '--remove-edges.by-vclass', 'pedestrian,bicycle,rail,tram,subway,ship',
                '--keep-edges.by-vclass', 'passenger,delivery,bus,taxi',
                '--ignore-errors.edge-type',
                '--no-warnings'
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            if not os.path.exists(net_file):
                print(f"  ✗ Network conversion failed: Output file missing.")
                continue

        except subprocess.CalledProcessError as e:
            print(f"  ✗ Network conversion failed: {e.stderr.decode()}")
            continue
        except FileNotFoundError:
            print("  ✗ ERROR: 'netconvert' tool not found. Check your SUMO installation/PATH.")
            return False

        # --- 5. Generate Random Vehicle Trips ---
        print(f"  -> Generating random traffic routes...")
        try:
            subprocess.run([
                sys.executable, os.path.join(sumo_tools_path, 'randomTrips.py'),
                '-n', net_file,
                '-r', route_file,
                '-e', str(INNER_STEPS),  # End time matches episode length
                '-p', '0.5',  # Period: spawn a car every 0.5 seconds roughly
                '--vehicle-class', 'passenger',
                '--min-distance', '500',  # Min trip distance in meters
                '--validate'
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Route generation failed: {e.stderr.decode()}")
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

        print(f"  ✓ Successfully generated scenario for {city}.")

    print("\n✓ Real-world scenario generation process complete.")
    return True


if __name__ == "__main__":
    generate_real_world_scenarios()
