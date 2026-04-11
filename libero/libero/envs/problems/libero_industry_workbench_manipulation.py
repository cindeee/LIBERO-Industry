from robosuite.utils.mjcf_utils import new_site
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from libero.libero.envs.regions.base_region_sampler import MultiRegionRandomSampler
import numpy as np
import xml.etree.ElementTree as ET

from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.robots import *
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.utils import rectangle2xyrange
from libero.libero.assets.industry_objects.conveyor_physics import ConveyorBeltMixin
from libero.libero.assets.industry_objects.conveyor_slot_physics import ConveyorSlotMixin

class ConveyorPhysicsSampler:
    """Samples physics parameters for the conveyor belt to enable Domain Randomization."""
    def __init__(self, speed_range=(0.02, 0.1), modes=["steady", "pulse"]):
        self.speed_range = speed_range
        self.modes = modes

    def sample(self):
        return {
            "speed": np.random.uniform(*self.speed_range),
            "mode": np.random.choice(self.modes)
        }

@register_problem
class Libero_Industry_Workbench_Manipulation( ConveyorBeltMixin, ConveyorSlotMixin, BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.completed_objects = set()
        self.workspace_name = "industry_workbench"
        self.visualization_sites_list = []
        if "industry_workbench_full_size" in kwargs:
            self.industry_workbench_full_size = workbench_full_size
        else:
            self.industry_workbench_full_size = (1.0, 1.2, 0.05)
        self.industry_workbench_offset = (0.0, 0, 0.80)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.industry_workbench_full_size[2]

        self.current_conveyor_params = {"speed": 0.05, "mode": "steady"}
        self.physics_sampler = ConveyorPhysicsSampler()

        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.industry_workbench_offset})
        kwargs.update({"arena_type": "industry_workbench"})
        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update(
                {"scene_xml": "scenes/libero_industry_workbench_base_style.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "gray-ceramic",
                        "wall_style": "yellow-linen",
                }
            }
        )

        super().__init__(bddl_file_name, *args, **kwargs)
        
    # reset conveyor upon parsing BDDL
    def _reset_internal(self):
        self._track_initialized = False 
        super()._reset_internal()
        if hasattr(self, 'dynamic_groove_names'):
            self.sim.forward()
            self._initialize_track()
            self.sim.forward()

        # Look through the parsed BDDL initial state for our 'running' predicate
        found_running_state = False
        for state in self.parsed_problem["initial_state"]:
            if state[0].lower() == "atspeed":
                # state format from BDDL: ['AtSpeed', 'conveyor_belt_1', '0.05', 'steady']
                self.current_conveyor_params["speed"] = float(state[2])
                self.current_conveyor_params["mode"] = state[3]
                found_running_state = True
                break
                
        # If the BDDL didn't specify it, randomly sample it for generalization!
        if not found_running_state:
            self.current_conveyor_params = self.physics_sampler.sample()

        # Apply the parameters to the Mixin
        self.setup_conveyor_belt(
            velocity=self.current_conveyor_params["speed"], 
            local_axis=(0, 1, 0)
        )

    def step(self, action):
        if self.sim.data.time == 0.0: # Only print on the very first frame
            print("\n[DEBUG MUJOCO] Checking simulation memory...")
            if hasattr(self, "dynamic_groove_names") and len(self.dynamic_groove_names) > 0:
                test_name = self.dynamic_groove_names[0]
                if test_name in self.obj_body_id:
                    body_id = self.obj_body_id[test_name]
                    world_pos = self.sim.data.body_xpos[body_id]
                    print(f"[DEBUG MUJOCO] SUCCESS: {test_name} is in MuJoCo!")
                    print(f"[DEBUG MUJOCO] Position: {world_pos}")
                else:
                    print(f"[DEBUG MUJOCO] FAILED: {test_name} is NOT in obj_body_id. It was lost during XML compilation.")
            else:
                print("[DEBUG MUJOCO] FAILED: dynamic_groove_names is empty.")
        # ==========================================================
            bottle_name = "pump_bottle_flat_1"
            if bottle_name in self.obj_body_id:
                bottle_id = self.obj_body_id[bottle_name]
                bottle_pos = self.sim.data.body_xpos[bottle_id]
                
                # Print the position every ~0.5 seconds (assuming 20 control_freq)
                # to monitor if it's drifting or sinking.
                step_count = int(self.sim.data.time * 20)
                if step_count % 10 == 0:
                    print(f"[PHYSICS DEBUG | t={self.sim.data.time:.2f}] {bottle_name} POS: x={bottle_pos[0]:.3f}, y={bottle_pos[1]:.3f}, z={bottle_pos[2]:.3f}")
                    
                # WARNING TRIGGER: Check if it falls through the table or flies up
                if bottle_pos[2] < 0.78: # Assuming table surface is around 0.8
                    print(f"🚨 [PHYSICS ERROR] {bottle_name} fell through the belt! Z={bottle_pos[2]:.3f}")
                elif bottle_pos[2] > 0.90:
                    print(f"🚨 [PHYSICS ERROR] {bottle_name} bounced/flew into the air! Z={bottle_pos[2]:.3f}")
        # ==========================================================
        # Handle "pulse" mode (rhythmic start/stop)
        if self.current_conveyor_params["mode"] == "pulse":
            # Example: 1 second on, 1 second off based on simulation time
            sim_time = self.sim.data.time
            is_moving = 1.0 if (sim_time % 2.0) < 1.0 else 0.0
            # Dynamically update the velocity property used by ConveyorBeltMixin
            self.conveyor_velocity = self.current_conveyor_params["speed"] * is_moving
            
        # Run the standard step logic
        obs, reward, done, info = super().step(action)
        
        # RECORD THE EXACT SPEED IN THE INFO DICTIONARY
        info["conveyor_speed"] = self.conveyor_velocity
        info["conveyor_mode"] = self.current_conveyor_params["mode"]
        
        return obs, reward, done, info
            

    def _load_fixtures_in_arena(self, mujoco_arena):
        """Nothing extra to load in this simple problem."""
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == "industry_workbench":
                continue
            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
                    name=fixture_instance,
                    joints=None,
                )

    def _load_objects_in_arena(self, mujoco_arena):
        from libero.libero.envs.objects import get_object_fn
        self.dynamic_groove_names = [] 
        
        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(name=object_name)

        for state in self.parsed_problem["initial_state"]:
            if state[0].lower() == "loadslot":
                
                # 1. check arguments
                expected_format = "('LoadSlot', 'conveyor_name', 'groove_type', 'orientation (x/y)', 'mode', 'speed')"
                assert len(state) == 6, f"\n[BDDL ERROR] Incorrect arguments for LoadSlot.\nExpected: {expected_format}\nGot: {state}\n"
                
                conveyor_name = state[1]
                groove_type = state[2]
                orientation = state[3].lower()
                mode = state[4].lower()
                speed = float(state[5])
                
                groove_class = get_object_fn(groove_type)
                dummy = groove_class(name="dummy")
                
                dim_x = dummy.parsed_size[0] * 2.0
                dim_y = dummy.parsed_size[1] * 2.0
                for site in dummy.worldbody.findall(".//site"):
                    if "horizontal_radius_site" in site.get("name", ""):
                        pos_str = site.get("pos")
                        if pos_str:
                            # Convert "0.1 0.0935 0" into [0.1, 0.0935, 0.0]
                            pos_vals = [abs(float(x)) for x in pos_str.split(" ")]
                            
                            # If the creator put the site on the diagonal, use X and Y independently
                            if pos_vals[0] > 0 and pos_vals[1] > 0:
                                dim_x = pos_vals[0] * 2.0
                                dim_y = pos_vals[1] * 2.0
                            else:
                                # Standard circular fallback (just use the largest value as radius)
                                radius = max(pos_vals)
                                dim_x = radius * 2.0
                                dim_y = radius * 2.0
                                
                            print(f"[DEBUG PARSING] Standardized size for {groove_type}: X={dim_x}, Y={dim_y}")
                        break
                margin = 0.01 


                groove_bottom_offset = dummy.bottom_offset[2]
                
                if orientation == "x":
                    groove_step = dim_x + margin 
                    target_quat_arr = [0.7071068, 0, 0, 0.7071068] # w, x, y, z format
                else: 
                    groove_step = dim_y + margin
                    target_quat_arr = [1, 0, 0, 0]
                
                # Dynamically estimate how many grooves to spawn based on the ghost conveyor's parsed size
                ghost_obj = self.objects_dict.get(conveyor_name) or self.fixtures_dict.get(conveyor_name)
                try:
                    site_xml = None
                    # Flexible search that ignores Robosuite's prefix renaming
                    for site in ghost_obj.worldbody.findall(".//site"):
                        if "ghost_active_region" in site.get("name", ""):
                            site_xml = site
                            break
                            
                    if site_xml is not None:
                        size_str = site_xml.get("size")
                        half_length = float(size_str.split(" ")[1])
                        print(f"[DEBUG PARSING] Successfully read active region half-length: {half_length}")
                    else:
                        print("[DEBUG PARSING] Could not find any site containing 'ghost_active_region'")
                        half_length = 0.4 # Fallback
                except Exception as e:
                    print(f"[DEBUG PARSING] XML parsing failed unexpectedly: {e}")
                    half_length = 0.4 # Fallback
                
                conveyor_length = half_length * 2
                num_grooves = int(conveyor_length / groove_step)
                
                print(f"[DEBUG PARSING] Found LoadSlot!")
                print(f"[DEBUG PARSING] Conveyor length: {conveyor_length}, Groove step: {groove_step}")
                print(f"[DEBUG PARSING] Generating {num_grooves} grooves of type '{groove_type}'")
                # ----------------------

                # ========================================================
                # NEW: DYNAMIC CONVEYOR VISUAL GENERATION
                # ========================================================
                import xml.etree.ElementTree as ET
                
                perfect_length = num_grooves * groove_step
                belt_half_width = (dim_y if orientation == "x" else dim_x) / 2.0
                
                target_body = None
                for body in ghost_obj.worldbody.findall(".//body"):
                    if "object" in body.get("name", ""):
                        target_body = body
                        break
                if target_body is None:
                    target_body = ghost_obj.worldbody.find("body")
                    
                # 1. Bulletproof duplicate check
                if target_body is not None and target_body.find(f".//geom[@name='{conveyor_name}_dyn_belt']") is None:
                    
                    # 2. PRECISE MATH: Use the fixture's own z_offset to find the table surface
                    table_surface_z = -ghost_obj.z_offset  # Translates to -0.005 locally
                    
                    # Belt: 2mm thick sheet resting explicitly ON the table
                    belt_z = table_surface_z + 0.001
                    ET.SubElement(target_body, "geom", {
                        "name": f"{conveyor_name}_dyn_belt", "type": "box",
                        "size": f"{belt_half_width} {perfect_length/2} 0.001",
                        "pos": f"0 0 {belt_z}", "quat": "1 0 0 0",
                        "rgba": "0.15 0.15 0.15 1", "group": "1", "contype": "0", "conaffinity": "0"
                    })
                    
                    # Side Frames: 4cm tall, resting ON the table
                    frame_z = table_surface_z + 0.02
                    ET.SubElement(target_body, "geom", {
                        "name": f"{conveyor_name}_dyn_frame_l", "type": "box",
                        "size": f"0.01 {perfect_length/2 + 0.02} 0.02",
                        "pos": f"{-belt_half_width - 0.01} 0 {frame_z}", "quat": "1 0 0 0",
                        "rgba": "0.3 0.35 0.4 1", "group": "1", "contype": "0", "conaffinity": "0"
                    })
                    ET.SubElement(target_body, "geom", {
                        "name": f"{conveyor_name}_dyn_frame_r", "type": "box",
                        "size": f"0.01 {perfect_length/2 + 0.02} 0.02",
                        "pos": f"{belt_half_width + 0.01} 0 {frame_z}", "quat": "1 0 0 0",
                        "rgba": "0.3 0.35 0.4 1", "group": "1", "contype": "0", "conaffinity": "0"
                    })

                    # Rollers: At the ends, sitting ON the table
                    roller_z = table_surface_z + 0.01
                    roller_quat = "0.7071068 0 0.7071068 0"
                    ET.SubElement(target_body, "geom", {
                        "name": f"{conveyor_name}_dyn_roller_s", "type": "cylinder",
                        "size": f"0.01 {belt_half_width}", "pos": f"0 {-perfect_length/2} {roller_z}", 
                        "quat": roller_quat, "rgba": "0.2 0.2 0.25 1", "group": "1", "contype": "0", "conaffinity": "0"
                    })
                    ET.SubElement(target_body, "geom", {
                        "name": f"{conveyor_name}_dyn_roller_e", "type": "cylinder",
                        "size": f"0.01 {belt_half_width}", "pos": f"0 {perfect_length/2} {roller_z}", 
                        "quat": roller_quat, "rgba": "0.2 0.2 0.25 1", "group": "1", "contype": "0", "conaffinity": "0"
                    })

                    # Hoods: 2cm thick, sitting on TOP of the side frames
                    hood_z = table_surface_z + 0.04 + 0.01
                    ET.SubElement(target_body, "geom", {
                        "name": f"{conveyor_name}_dyn_hood_s", "type": "box",
                        "size": f"{belt_half_width + 0.02} 0.04 0.01",
                        "pos": f"0 {-perfect_length/2} {hood_z}", "quat": "1 0 0 0",
                        "rgba": "0.3 0.35 0.4 1", "group": "1", "contype": "0", "conaffinity": "0"
                    })
                    ET.SubElement(target_body, "geom", {
                        "name": f"{conveyor_name}_dyn_hood_e", "type": "box",
                        "size": f"{belt_half_width + 0.02} 0.04 0.01",
                        "pos": f"0 {perfect_length/2} {hood_z}", "quat": "1 0 0 0",
                        "rgba": "0.3 0.35 0.4 1", "group": "1", "contype": "0", "conaffinity": "0"
                    })
                    
                    print("\n[DEBUG VISUALS] --- DYNAMIC CONVEYOR GENERATION ---")
                    print(f"[DEBUG VISUALS] Target Conveyor: {conveyor_name}")
                    print(f"[DEBUG VISUALS] Table Surface Local Z: {table_surface_z}")
                    print("[DEBUG VISUALS] ---------------------------------------\n")
                # ========================================================
                for i in range(num_grooves):
                    g_name = f"dyn_{groove_type}_{i}"
                    g_obj = groove_class(name=g_name)
                    self.objects_dict[g_name] = g_obj
                    self.dynamic_groove_names.append(g_name)
                
                # Pass the TARGET CONVEYOR NAME to the physics engine, not hardcoded coordinates.
                self.setup_physical_track(
                    conveyor_name=conveyor_name,
                    speed=speed,
                    mode=mode,
                    groove_quat=target_quat_arr,
                    groove_bottom_offset=groove_bottom_offset, 
                    groove_step=groove_step  # Explicitly pass the tight spacing down! 
                )

    def _load_sites_in_arena(self, mujoco_arena):
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        
        for object_region_name in list(region_dict.keys()):

            if "industry_workbench" in object_region_name:
                ranges = region_dict[object_region_name]["ranges"][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2 + self.workspace_offset[0],
                    (ranges[3] + ranges[1]) / 2 + self.workspace_offset[1],
                )
                target_zone = TargetZone(
                    name=object_region_name,
                    rgba=region_dict[object_region_name]["rgba"],
                    zone_size=zone_size,
                    z_offset=self.workspace_offset[2],
                    zone_centroid_xy=zone_centroid_xy,
                )
                object_sites_dict[object_region_name] = target_zone
                mujoco_arena.table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos + np.array([0.0, 0.0, -0.90]),
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type="box",
                    )
                )
                continue
            # Handle other fixture/object regions
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for (name, body) in query_dict.items():
                    try:
                        if "worldbody" not in list(body.__dict__.keys()):
                            # This is a special case for CompositeObject, we skip this as this is very rare in our benchmark
                            continue
                    except:
                        continue
                    for part in body.worldbody.find("body").findall(".//body"):
                        sites = part.findall(".//site")
                        joints = part.findall("./joint")
                        if sites == []:
                            break
                        for site in sites:
                            site_name = site.get("name")
                            if site_name == object_region_name:
                                object_sites_dict[object_region_name] = SiteObject(
                                    name=site_name,
                                    parent_name=body.name,
                                    joints=[joint.get("name") for joint in joints],
                                    size=site.get("size"),
                                    rgba=site.get("rgba"),
                                    site_type=site.get("type"),
                                    site_pos=site.get("pos"),
                                    site_quat=site.get("quat"),
                                    object_properties=body.object_properties,
                                )
        self.object_sites_dict = object_sites_dict

        # Keep track of visualization objects
        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _add_placement_initializer(self):
        super()._add_placement_initializer()
        
        # We must patch ALL 3 samplers LIBERO uses, not just the base one
        initializers = [
            self.placement_initializer,
            self.conditional_placement_initializer,
            self.conditional_placement_on_objects_initializer
        ]
        
        for init in initializers:
            if hasattr(init, 'samplers'):
                for sampler_name, sampler in init.samplers.items():
                    obj_name = sampler_name.replace("_sampler", "")
                    
                    if self.is_fixture(obj_name):
                        correct_z_offset = self.fixtures_dict[obj_name].z_offset
                        sampler.z_offset = correct_z_offset
                        print(f"[DEBUG HOTFIX] Patched fixture {obj_name} with z_offset: {correct_z_offset}")
                    else:
                        # Prevent movable objects from dropping from the sky
                        sampler.z_offset = 0.002
                        print(f"[DEBUG HOTFIX] Patched movable {obj_name} to 2mm drop height")

    def _check_success(self):
        """
        Check if the goal is achieved. Consider conjunction goals at the moment
        """
        goal_state = self.parsed_problem["goal_state"]
        result = True
        for state in goal_state:
            result = self._eval_predicate(state) and result
        return result

    def _eval_predicate(self, state):
        """Custom success checking based on physical distance."""
        if state[0].lower() == "routed":
            bottle_name = state[1]
            
            bottle_id = self.obj_body_id[bottle_name]
            bottle_pos = self.sim.data.body_xpos[bottle_id]
            
            # Check if the bottle is fully seated inside ANY of the physical grooves
            for g_name in self.dynamic_groove_names:
                site_name = f"{g_name}_place_region"
                
                if site_name in self.sim.model.site_names:
                    site_id = self.sim.model.site_name2id(site_name)
                    site_pos = self.sim.data.site_xpos[site_id]
                    
                    dist = np.linalg.norm(bottle_pos[:2] - site_pos[:2])
                    z_dist = abs(bottle_pos[2] - site_pos[2])
                    
                    if dist < 0.02 and z_dist < 0.05:
                        return True 
            return False

        if len(state) == 3:
            # Checking binary logical predicates
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        elif len(state) == 2:
            # Checking unary logical predicates
            predicate_fn_name = state[0]
            object_name = state[1]
            return eval_predicate_fn(
                predicate_fn_name, self.object_states_dict[object_name]
            )

    def _setup_references(self):
        super()._setup_references()

    def _post_process(self):
        super()._post_process()

        self.set_visualization()

    def set_visualization(self):

        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name).object_properties["vis_site_names"].items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if ((self.sim.model.site_rgba[vis_g_id][3] <= 0) and site_visible) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0) and not site_visible
                ):
                    # We toggle the alpha value
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    )

    def _setup_camera(self, mujoco_arena):
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.6586131746834771, 0.0, 1.6103500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

        # For visualization purpose
        mujoco_arena.set_camera(
            camera_name="frontview", pos=[1.0, 0.0, 1.48], quat=[0.56, 0.43, 0.43, 0.56]
        )
        mujoco_arena.set_camera(
            camera_name="galleryview",
            pos=[2.844547668904445, 2.1279684793440667, 3.128616846013882],
            quat=[
                0.42261379957199097,
                0.23374411463737488,
                0.41646939516067505,
                0.7702690958976746,
            ],
        )
        mujoco_arena.set_camera(
            camera_name="paperview",
            pos=[2.1, 0.535, 2.075],
            quat=[0.513, 0.353, 0.443, 0.645],
        )
