import mujoco
import numpy as np

# Load your model
model = mujoco.MjModel.from_xml_path('/Users/cindy/experiments/LIBERO/libero/libero/assets/industry_objects/pump_bottle/pump_bottle.xml')
data = mujoco.MjData(model)

def get_mesh_vertices(model, mesh_name):
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
    if mesh_id == -1:
        raise ValueError(f"Mesh '{mesh_name}' not found.")
    vert_adr = model.mesh_vertadr[mesh_id]
    vert_num = model.mesh_vertnum[mesh_id]
    return model.mesh_vert[vert_adr : vert_adr + vert_num]

# Get vertices for both exterior meshes
vertices_base = get_mesh_vertices(model, 'mesh_base')
vertices_cap = get_mesh_vertices(model, 'mesh_cap')

# Combine them to find the global bounding box
all_vertices = np.vstack((vertices_base, vertices_cap))

# Calculate bounding box
min_bound = np.min(all_vertices, axis=0)
max_bound = np.max(all_vertices, axis=0)

diameter_x = max_bound[0] - min_bound[0]
diameter_y = max_bound[1] - min_bound[1]
height = max_bound[2] - min_bound[2]
center_z = (max_bound[2] + min_bound[2]) / 2

print(f"Diameter (X): {diameter_x}")
print(f"Diameter (Y): {diameter_y}")
print(f"Height (Z): {height}")
print(f"Center (Z): {center_z}")
print(f"Min Z: {min_bound[2]} | Max Z: {max_bound[2]}")