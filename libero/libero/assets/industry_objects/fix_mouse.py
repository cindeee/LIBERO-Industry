
import os
import re
import glob

def clean_and_generate_model():
    # Target directory
    thing = "keyboard"
    folder_path = f"/Users/cindy/experiments/LIBERO/libero/libero/assets/industry_objects/{thing}"
    xml_output_path = os.path.join(folder_path, "model.xml")
    
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
    # (e.g., renaming 3 to 2 when 2 already exists)
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

    # 3. Generate the XML strings
    asset_meshes = ""
    collision_geoms = ""
    
    for i in range(num_collisions):
        asset_meshes += f'    <mesh name="model_collision_{i}" file="model_collision_{i}.obj"/>\n'
        collision_geoms += f'                <geom mesh="model_collision_{i}" type="mesh" group="3"/>\n'

    # Robosuite / LIBERO Environment XML Template
    xml_content = f"""<mujoco model="{thing}">
  <asset>
    <texture type="2d" name="texture" file="texture.png"/>
    <material name="material_0" texture="texture" specular="0.5" shininess="0.5"/>
    <mesh name="model" file="model.obj"/>
{asset_meshes.rstrip()}
  </asset>
  
  <worldbody>
    <body>
        <body name="object" pos="0 0 0"> 
            <body name="{thing}" pos="0 0 0" quat="1 0 0 0"> 
                <inertial pos="0 0 0" mass="0.2" diaginertia="1e-3 1e-3 1e-3" />
                
                <geom material="material_0" mesh="model" type="mesh" contype="0" conaffinity="0" group="2"/>
                
                {collision_geoms.rstrip()}
            </body>
        </body>
      
      <site rgba="0 0 0 0" size="0.01" pos="0 0 -0.01" quat="1 0 0 0" name="bottom_site"/>        
      <site rgba="0 0 0 0" size="0.01" pos="0 0 0.01" quat="1 0 0 0" name="top_site"/>   
      <site rgba="0 0 0 0" size="0.01" pos="0.08 0 0" quat="1 0 0 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""

    # 4. Write the structured XML
    with open(xml_output_path, 'w') as f:
        f.write(xml_content)
    
    print(f"Generated properly structured XML at: {xml_output_path}")

if __name__ == "__main__":
    clean_and_generate_model()