import os
import xml.etree.ElementTree as ET

BASE_DIR = "/Users/cindy/experiments/LIBERO/libero/libero/assets/industry_objects"
variants = ['a', 'b', 'c', 'd']

# Base X, Y coordinates for the center of each hole
base_xy = {
    'cream_bottle': '0.026151 0.026151',
    'jar': '0.025076 -0.045993',
    'pump_bottle': '-0.065 0',
    'serum_bottle': '0.027063 0.007534'
}

# The full dimensions from Blender (Dim 1, Dim 2, Dim 3)
dimensions = {
    'a': {
        'cream_bottle': (0.0504, 0.02928, 0.1092),
        'jar': (0.08016, 0.08016, 0.0618),
        'pump_bottle': (0.03924, 0.03924, 0.1464),
        'serum_bottle': (0.03276, 0.03276, 0.10152)
    },
    'b': {
        'cream_bottle': (0.0462, 0.02684, 0.1001),
        'jar': (0.07348, 0.07348, 0.05665),
        'pump_bottle': (0.03597, 0.03597, 0.1342),
        'serum_bottle': (0.03003, 0.03003, 0.09306)
    },
    'c': {
        'cream_bottle': (0.0441, 0.02562, 0.09555),
        'jar': (0.07014, 0.07014, 0.054075),
        'pump_bottle': (0.034335, 0.034335, 0.1281),
        'serum_bottle': (0.028665, 0.028665, 0.08883)
    },
    'd': {
        'cream_bottle': (0.04284, 0.024888, 0.09282),
        'jar': (0.068136, 0.068136, 0.05253),
        'pump_bottle': (0.033354, 0.033354, 0.12444),
        'serum_bottle': (0.027846, 0.027846, 0.086292)
    }
}

for variant in variants:
    folder_name = f"cosmetics_inlay_{variant}"
    xml_file = f"{folder_name}.xml"
    file_path = os.path.join(BASE_DIR, folder_name, xml_file)
    
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        continue
        
    print(f"Rebuilding {xml_file}...")
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # --- 1. Find all meshes from the <asset> tag ---
    mesh_names = []
    for mesh in root.findall('.//asset/mesh'):
        name = mesh.get('name')
        if name:
            mesh_names.append(name)
            
    # --- 2. Remove the broken worldbody ---
    old_worldbody = root.find('worldbody')
    if old_worldbody is not None:
        root.remove(old_worldbody)
        
    # --- 3. Build the pristine, Robosuite-compliant worldbody ---
    new_worldbody = ET.SubElement(root, 'worldbody')
    
    # Layer 1: Unnamed Root Body
    root_body = ET.SubElement(new_worldbody, 'body')
    
    # Layer 2: "object" Body
    object_body = ET.SubElement(root_body, 'body', {'name': 'object', 'pos': '0 0 0', 'quat': '1 0 0 0'})
    
    # Layer 3: "cosmetics_inlay_X" Body
    inlay_body = ET.SubElement(object_body, 'body', {'name': folder_name, 'pos': '0 0 0', 'quat': '1 0 0 0'})
    
    # Add Inertial property
    ET.SubElement(inlay_body, 'inertial', {'pos': '0 0 0', 'mass': '0.5', 'diaginertia': '1e-3 1e-3 1e-3'})
    
    # --- 4. Reconstruct Geoms from Meshes ---
    for mesh_name in mesh_names:
        if 'collision' in mesh_name:
            ET.SubElement(inlay_body, 'geom', {
                'type': 'mesh', 'mesh': mesh_name, 'group': '0', 'quat': '1 0 0 0'
            })
        else:
            ET.SubElement(inlay_body, 'geom', {
                'type': 'mesh', 'mesh': mesh_name, 'group': '1', 'quat': '1 0 0 0',
                'rgba': '0.8 0.8 0.8 1', 'contype': '0', 'conaffinity': '0'
            })
            
    # --- 5. Inject Mathematical Goal Points & Areas ---
    for bottle_type, xy in base_xy.items():
        dims = dimensions[variant][bottle_type]
        
        if bottle_type == 'cream_bottle':
            radius = dims[1] / 2.0
        else:
            radius = dims[0] / 2.0

        max_planar = max(dims[0], dims[2])
        box_xy_half_size = max_planar / 2.0
        
        z_contact_point = 0.025 - radius
        z_center_area = 0.025 - (radius / 2.0)
        z_half_size_area = radius / 2.0
        
        # Point Site
        ET.SubElement(inlay_body, 'site', {
            'name': f"goal_point_{bottle_type}",
            'type': 'sphere',
            'size': '0.003',
            'pos': f"{xy} {z_contact_point:.6f}",
            'quat': '1 0 0 0',
            'rgba': '1 0 0 0.8',
            'group': '0'
        })

        # Area Site
        ET.SubElement(inlay_body, 'site', {
            'name': f"goal_area_{bottle_type}",
            'type': 'box',
            'size': f"{box_xy_half_size:.6f} {box_xy_half_size:.6f} {z_half_size_area:.6f}",
            'pos': f"{xy} {z_center_area:.6f}",
            'quat': '1 0 0 0',
            'rgba': '0 1 0 0.2',
            'group': '0'
        })
        
    # --- 6. Inject Standard Boundary Sites (Outside the nested bodies) ---
    ET.SubElement(root_body, 'site', {'name': 'bottom_site', 'pos': '0 0 -0.0245', 'size': '0.01', 'rgba': '0 0 0 0', 'quat': '1 0 0 0'})
    ET.SubElement(root_body, 'site', {'name': 'top_site', 'pos': '0 0 0.0245', 'size': '0.01', 'rgba': '0 0 0 0', 'quat': '1 0 0 0'})
    ET.SubElement(root_body, 'site', {'name': 'horizontal_radius_site', 'pos': '0.1 0 0', 'size': '0.01', 'rgba': '0 0 0 0', 'quat': '1 0 0 0'})

    # --- 7. Save ---
    if hasattr(ET, "indent"): ET.indent(tree, space="  ")
    tree.write(file_path, encoding='utf-8')
    print(f"✅ Successfully rescued and completely rebuilt {xml_file}!")