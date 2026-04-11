import os
import re
import glob
import math

def calculate_sites_from_obj(obj_path):
    """
    Parses the OBJ file to find the bounding dimensions 
    and calculate top, bottom, and horizontal radius sites.
    Assumes the object is centered at the global origin (0, 0, 0).
    """
    if not os.path.exists(obj_path):
        print(f"Warning: {obj_path} not found. Using default site values.")
        return -0.01, 0.01, 0.08
    
    min_z = float('inf')
    max_z = float('-inf')
    max_r = 0.0
    
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    min_z = min(min_z, z)
                    max_z = max(max_z, z)
                    
                    # Horizontal radius is the max euclidean distance in the XY plane
                    r = math.sqrt(x**2 + y**2)
                    max_r = max(max_r, r)
                    
    # Fallback if the obj file was empty or corrupted
    if min_z == float('inf'):
        return -0.01, 0.01, 0.08
        
    return min_z, max_z, max_r

def clean_and_generate_model():
    # Target directory
    thing = "mouse_taipan" # Set to the target object name
    folder_path = f"/Users/cindy/experiments/LIBERO/libero/libero/assets/industry_objects/{thing}"
    xml_output_path = os.path.join(folder_path, f"{thing}.xml")
    model_obj_path = os.path.join(folder_path, "model.obj")
    
    # 1. Find and sort all remaining collision files
    search_pattern = os.path.join(folder_path, "model_collision_*.obj")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No collision files found in {folder_path}.")
        return

    # Extract the number for accurate sorting (e.g., 2 before 10)
    def extract_num(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'model_collision_(\d+)\.obj', filename)
        return int(match.group(1)) if match else -1

    files.sort(key=extract_num)

    # 2. Rename files sequentially
    print(f"Found {len(files)} collision meshes. Renaming sequentially...")
    
    # First, rename to a temporary name to avoid accidental overwrites 
    temp_files = []
    for i, file_path in enumerate(files):
        temp_path = os.path.join(folder_path, f"temp_col_{i}.obj")
        os.rename(file_path, temp_path)
        temp_files.append(temp_path)
        
    # Second, rename to the final target name
    for i, temp_path in enumerate(temp_files):
        final_path = os.path.join(folder_path, f"model_collision_{i}.obj")
        os.rename(temp_path, final_path)
        
    num_collisions = len(temp_files)
    print(f"Successfully renumbered meshes from model_collision_0.obj to model_collision_{num_collisions-1}.obj")

    # 3. Calculate Sites dynamically using the primary model.obj
    min_z, max_z, max_r = calculate_sites_from_obj(model_obj_path)
    print(f"Calculated Sites -> Bottom Z: {min_z:.4f}, Top Z: {max_z:.4f}, Radius: {max_r:.4f}")

    # 4. Generate the XML strings
    asset_meshes = ""
    collision_geoms = ""
    
    for i in range(num_collisions):
        asset_meshes += f'    <mesh name="model_collision_{i}" file="model_collision_{i}.obj"/>\n'
        # Assign the 'collision' class as defined in defaults
        collision_geoms += f'          <geom mesh="model_collision_{i}" class="collision"/>\n'

    # Robosuite / LIBERO Environment XML Template (Matching pump bottle format)
    xml_content = f"""<mujoco model="{thing}">
  <default>
    <default class="visual">
      <geom group="1" type="mesh" contype="0" conaffinity="0"/>
    </default>
    <default class="collision">
      <geom group="0" type="mesh"/>
    </default>
  </default>
  <asset>
    <texture type="2d" name="texture" file="texture.png"/>
    <material name="material_0" texture="texture" specular="0.5" shininess="0.5"/>
    <mesh name="model" file="model.obj"/>
{asset_meshes.rstrip()}
  </asset>
  
  <worldbody>
    <body>
      <body name="object">
        <body name="{thing}" pos="0 0 0" quat="1 0 0 0">
          <inertial pos="0 0 0" mass="0.2" diaginertia="1e-3 1e-3 1e-3" />
          <geom material="material_0" mesh="model" class="visual"/>
{collision_geoms.rstrip()}
        </body>
      </body>
      <site rgba="0 0 0 0" size="0.01" pos="0 0 {min_z:.4f}" quat="1 0 0 0" name="bottom_site"/>        
      <site rgba="0 0 0 0" size="0.01" pos="0 0 {max_z:.4f}" quat="1 0 0 0" name="top_site"/>   
      <site rgba="0 0 0 0" size="0.01" pos="{max_r:.4f} 0 0" quat="1 0 0 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""

    # 5. Write the structured XML
    with open(xml_output_path, 'w') as f:
        f.write(xml_content)
    
    print(f"Generated properly structured XML at: {xml_output_path}")

if __name__ == "__main__":
    clean_and_generate_model()