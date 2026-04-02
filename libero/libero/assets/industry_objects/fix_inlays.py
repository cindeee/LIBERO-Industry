import os
import xml.etree.ElementTree as ET

# Base directory for the LIBERO assets
BASE_DIR = "/Users/cindy/experiments/LIBERO/libero/libero/assets/industry_objects"

# The variant letters we want to fix
variants = ['b', 'c', 'd']

for variant in variants:
    folder_name = f"pump_bottle_inlay_{variant}"
    xml_file = f"{folder_name}.xml"
    file_path = os.path.join(BASE_DIR, folder_name, xml_file)
    
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        continue
        
    print(f"Processing {xml_file}...")
    
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # 1. Fix Visibility & Collision Groups
    for default_geom in root.findall('.//default/geom'):
        if default_geom.get('group') == '2':
            default_geom.set('group', '1')  # Visuals
        elif default_geom.get('group') == '3':
            default_geom.set('group', '0')  # Collisions
            
    # 2. Fix Missing Mesh Names in Assets
    for mesh in root.findall('.//asset/mesh'):
        file_attr = mesh.get('file')
        if file_attr and not mesh.get('name'):
            # Extract name by removing the '.obj' extension
            name = file_attr.replace('.obj', '')
            mesh.set('name', name)
            
    # 3. Fix the Hierarchy Nesting
    worldbody = root.find('worldbody')
    if worldbody is not None:
        main_body = worldbody.find('body')
        if main_body is not None:
            obj_body = main_body.find('body[@name="object"]')
            
            if obj_body is not None:
                inner_body_name = f"pump_bottle_inlay_{variant}"
                inner_body = obj_body.find(f'body[@name="{inner_body_name}"]')
                
                # Check if it's already properly nested to avoid double-wrapping
                if inner_body is None:
                    # Create the nested body matching Inlay A's spatial properties
                    inner_body = ET.Element('body', {
                        'name': inner_body_name, 
                        'pos': '0 0 0', 
                        'quat': '1 0 0 0'
                    })
                    
                    # Migrate all existing geoms and sites (like hole_goals) into the inner_body
                    for child in list(obj_body):
                        inner_body.append(child)
                        obj_body.remove(child)
                        
                    # Inject rgba into the main visual mesh if missing
                    visual_geom = inner_body.find(f'geom[@mesh="{inner_body_name}"]')
                    if visual_geom is not None and not visual_geom.get('rgba'):
                        visual_geom.set('rgba', '0.8 0.8 0.8 1')
                        
                    # Nest the body
                    obj_body.append(inner_body)

                # --- 3b. Add the place_region site inside the inner body ---
                if inner_body.find('site[@name="place_region"]') is None:
                    ET.SubElement(inner_body, 'site', {
                        'name': 'place_region',
                        'type': 'box',
                        'size': '0.06 0.06 0.001',
                        'pos': '0 0 0',
                        'rgba': '0 0 1 0.2',
                        'quat': '1 0 0 0',
                        'group': '0'
                    })
                    
            # 4. Inject standard interaction sites (if they don't already exist)
            standard_sites = [
                {'rgba': '0 0 0 0', 'size': '0.05', 'pos': '0 0 -0.01', 'name': 'bottom_site'},
                {'rgba': '0 0 0 0', 'size': '0.05', 'pos': '0 0 0.01', 'name': 'top_site'},
                {'rgba': '0 0 0 0', 'size': '0.05', 'pos': '0 0 0.05', 'name': 'horizontal_radius_site'}
            ]
            
            for site_info in standard_sites:
                if main_body.find(f'site[@name="{site_info["name"]}"]') is None:
                    ET.SubElement(main_body, 'site', site_info)

    # 5. Prettify and save the changes
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ", level=0)
    
    tree.write(file_path, encoding='utf-8', xml_declaration=False)
    print(f"✅ Successfully updated {xml_file}!")

print("All targeted files now match the precision hierarchy and site layout of Inlay A.")