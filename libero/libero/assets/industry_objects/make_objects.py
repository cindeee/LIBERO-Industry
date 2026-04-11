import os

BASE_DIR = "/Users/cindy/experiments/LIBERO/libero/libero/assets/industry_objects"

# Dimensions mapping based on your measurements (X, Y, Z)
objects_data = {
    'cream_bottle': {'x': 0.042, 'y': 0.0244, 'z': 0.091, 'type': 'box'},
    'cream_jar':    {'x': 0.0668, 'y': 0.0668, 'z': 0.0515, 'type': 'cylinder'},
    'pump_bottle':  {'x': 0.0327, 'y': 0.0327, 'z': 0.122, 'type': 'cylinder'},
    'serum_bottle': {'x': 0.0273, 'y': 0.0273, 'z': 0.0846, 'type': 'cylinder'}
}

def generate_xml_for_object(obj_name, data):
    folder_path = os.path.join(BASE_DIR, obj_name)
    xml_output_path = os.path.join(folder_path, f"{obj_name}.xml")
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return

    # Find all .obj files in the specific object's folder
    obj_files = [f for f in os.listdir(folder_path) if f.endswith('.obj')]
    
    if not obj_files:
        print(f"⚠️ No .obj files found in {folder_path}. Skipping.")
        return

    # 1. Build Asset tags
    assets_str = ""
    for f in obj_files:
        mesh_name = f.replace('.obj', '')
        assets_str += f'      <mesh name="{mesh_name}" file="{f}" scale="1 1 1" />\n'

    # 2. Build Geom tags
    geoms_str = ""
    for f in obj_files:
        mesh_name = f.replace('.obj', '')
        # CHANGED: quat="0.7071068 0.7071068 0 0" (+90 degrees on X-axis) to stand Blender exports upright
        geoms_str += f'        <geom type="mesh" mesh="{mesh_name}" rgba="1 1 1 1" friction="0.6 0.005 0.001" quat="1 0 0 0" group="1" contype="1" conaffinity="1" pos="0 0 0"/>\n'

    # 3. Math Calculations
    half_z = data['z'] / 2.0
    
    if data['type'] == 'cylinder':
        radius = data['x'] / 2.0
        site_size = f"{radius:.5f} {half_z:.5f}"
        site_type = "cylinder"
    else: # box for cream_bottle
        half_x = data['x'] / 2.0
        half_y = data['y'] / 2.0
        radius = half_x # Use half_x for the horizontal radius site
        site_size = f"{half_x:.5f} {half_y:.5f} {half_z:.5f}"
        site_type = "box"

    # 4. Construct the XML string
    xml_content = f"""<mujoco model="{obj_name}">
    <size njmax="500" nconmax="100" />
    <asset>
{assets_str.rstrip()}
    </asset>

  <worldbody>
    <body>
      <body name="object" pos="0 0 0" quat="1 0 0 0">
        <inertial pos="0 0 0" mass="0.5" diaginertia="1e-3 1e-3 1e-3" />       
         
{geoms_str.rstrip()}
        
        <site name="bottle_site" type="{site_type}" size="{site_size}" pos="0 0 0" quat="1 0 0 0" rgba="1 0 0 0"/>
        <site name="contact_site" type="{site_type}" size="{site_size}" pos="0 0 0" quat="1 0 0 0" rgba="0 0 1 0" group="0"/>
      </body>
      <site rgba="0 0 1 1" size="0.01" pos="0 0 -{half_z:.5f}" quat="1 0 0 0" name="bottom_site"/>  
      <site rgba="0 1 0 1" size="0.01" pos="0 0 {half_z:.5f}" quat="1 0 0 0" name="top_site"/>   
      <site rgba="1 0 0 1" size="0.01" pos="{radius:.5f} 0 0" quat="1 0 0 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""

    # 5. Save the file
    with open(xml_output_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    print(f"✅ Successfully generated {xml_output_path}")

# Run the generation for all 4 objects
for obj_name, data in objects_data.items():
    generate_xml_for_object(obj_name, data)

print("\n🎉 All XML files generated with updated geometry orientations!")