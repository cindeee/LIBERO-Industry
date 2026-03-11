import os
import sys
sys.path.insert(0, '/Users/cindy/experiments/LIBERO')
import numpy as np
from libero.libero.envs import OffScreenRenderEnv

# Use the generated BDDL file
bddl_file = "./industry_pddl/CONVEYOR_TEST_SCENE_place_box_from_conveyor_belt_to_goal.bddl"

env_args = {
    "bddl_file_name": bddl_file,
    "camera_heights": 256,
    "camera_widths": 256,
    "has_renderer": False,
    "has_offscreen_renderer": True,
    "use_camera_obs": True,
    "control_freq": 20,
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
obs = env.reset()
inner = env.env  # actual BDDLBaseDomain

print("=" * 80)
print("DEBUGGING CONVEYOR BELT PHYSICS")
print("=" * 80)

# Step 1: Check objects_dict
print("\n1. Objects in objects_dict:")
for obj_name in inner.objects_dict.keys():
    print(f"   - {obj_name}")

# Step 2: Check body IDs
print("\n2. Object body IDs:")
for obj_name in inner.objects_dict.keys():
    body_id = inner.obj_body_id.get(obj_name)
    print(f"   - {obj_name}: body_id = {body_id}")

# Step 3: Check joint addresses
print("\n3. Joint addresses (body_jntadr):")
for obj_name in inner.objects_dict.keys():
    body_id = inner.obj_body_id.get(obj_name)
    if body_id is not None:
        jnt_adr = inner.sim.model.body_jntadr[body_id]
        print(f"   - {obj_name}: jnt_adr = {jnt_adr}")

# Step 4: Check initial positions
print("\n4. Initial positions:")
for obj_name in inner.objects_dict.keys():
    body_id = inner.obj_body_id.get(obj_name)
    if body_id is not None:
        pos = inner.sim.data.body_xpos[body_id]
        print(f"   - {obj_name}: pos = {pos}")

# Step 5: Check conveyor site position
print("\n5. Conveyor belt site:")
try:
    site_id = inner.sim.model.site_name2id("conveyor_belt_1_contact_region")
    site_pos = inner.sim.data.site_xpos[site_id]
    print(f"   - conveyor_belt_1_contact_region: pos = {site_pos}")
except:
    print("   - conveyor_belt_1_contact_region: NOT FOUND")

# Step 6: Run simulation and check which objects get velocity applied
print("\n6. Testing velocity application (10 steps):")
dummy_action = [0.] * 7

for step in range(10):
    # Before step
    print(f"\n   Step {step}:")
    
    # Check which objects are detected on belt
    site_positions = []
    for site_name in inner.conveyor_site_names:
        try:
            site_id = inner.sim.model.site_name2id(site_name)
            site_pos = inner.sim.data.site_xpos[site_id]
            site_positions.append(site_pos)
        except:
            continue
    
    for obj_name in inner.objects_dict.keys():
        body_id = inner.obj_body_id[obj_name]
        obj_pos = inner.sim.data.body_xpos[body_id]
        
        # Check distance to belt
        for site_pos in site_positions:
            horizontal_dist = np.linalg.norm(obj_pos[:2] - site_pos[:2])
            vertical_dist = abs(obj_pos[2] - site_pos[2])
            
            on_belt = horizontal_dist < 0.1 and vertical_dist < 0.04
            
            if on_belt:
                print(f"      {obj_name}: ON BELT (h_dist={horizontal_dist:.4f}, v_dist={vertical_dist:.4f})")
            else:
                print(f"      {obj_name}: OFF BELT (h_dist={horizontal_dist:.4f}, v_dist={vertical_dist:.4f})")
    
    obs, reward, done, info = env.step(dummy_action)

# Step 7: Check final velocities
print("\n7. Final velocities after 10 steps:")
for obj_name in inner.objects_dict.keys():
    body_id = inner.obj_body_id[obj_name]
    jnt_adr = inner.sim.model.body_jntadr[body_id]
    vel = inner.sim.data.qvel[jnt_adr:jnt_adr+3]
    print(f"   - {obj_name}: vel = {vel}")

# Step 8: Check final positions
print("\n8. Final positions after 10 steps:")
for obj_name in inner.objects_dict.keys():
    body_id = inner.obj_body_id[obj_name]
    pos = inner.sim.data.body_xpos[body_id]
    print(f"   - {obj_name}: pos = {pos}")

env.close()
print("\n" + "=" * 80)
