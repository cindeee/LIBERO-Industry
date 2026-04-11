import os
import glob

# Base directory
base_dir = "/Users/cindy/experiments/LIBERO/libero/libero/assets/industry_objects"
shoes = ['shoe_left', 'shoe_right']

for shoe in shoes:
    shoe_dir = os.path.join(base_dir, shoe)
    
    if not os.path.exists(shoe_dir):
        print(f"Directory not found: {shoe_dir}")
        continue

    print(f"--- Processing {shoe} ---")

    # 1. Rename any _001 files (like model_collision_30_001.mtl/obj)
    for filename in os.listdir(shoe_dir):
        if "_001" in filename:
            old_path = os.path.join(shoe_dir, filename)
            new_path = os.path.join(shoe_dir, filename.replace("_001", ""))
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {os.path.basename(new_path)}")

    # 2. Remove all .mtl files
    mtl_files = glob.glob(os.path.join(shoe_dir, "*.mtl"))
    for mtl_file in mtl_files:
        os.remove(mtl_file)
        print(f"Removed: {os.path.basename(mtl_file)}")

    # 3. Format the XML file
    xml_path = os.path.join(shoe_dir, "model.xml")
    if os.path.exists(xml_path):
        with open(xml_path, 'r') as f:
            xml_data = f.read()

        # Fix any internal XML mesh references to _001
        xml_data = xml_data.replace("_001", "")
        
        # Update mujoco model tag name
        xml_data = xml_data.replace('<mujoco model="model">', f'<mujoco model="{shoe}">')
        
        # Wrap the original body inside the standard object wrapper
        old_body_start = '<body name="model">'
        new_body_structure = f'''<body name="object" pos="0 0 0" quat="1 0 0 0">
        <body name="{shoe}" pos="0 0 0" quat="1 0 0 0">'''
        xml_data = xml_data.replace(old_body_start, new_body_structure)
        
        # Inject the closing tags and the standard LIBERO size sites
        # Dimensions: 0.107 0.285 0.12 (Half-extents: Z=0.06, X=0.0535, Y=0.1425)
        old_body_end = '''    </body>
  </worldbody>'''
        
        new_body_end = '''      </body>
      </body>
      <site name="bottom_site" pos="0 0 -0.06" size="0.01" rgba="0 0 0 0" quat="1 0 0 0" />
      <site name="top_site" pos="0 0 0.06" size="0.01" rgba="0 0 0 0" quat="1 0 0 0" />
      <site name="horizontal_radius_site" pos="0.0535 0.1425 0" size="0.01" rgba="0 0 0 0" quat="1 0 0 0" />
    </body>
  </worldbody>'''
        
        xml_data = xml_data.replace(old_body_end, new_body_end)

        # Save the properly formatted XML back out
        with open(xml_path, 'w') as f:
            f.write(xml_data)
        print(f"Successfully formatted model.xml for {shoe}")
    else:
        print(f"Could not find model.xml in {shoe_dir}")

print("--- Script complete! ---")