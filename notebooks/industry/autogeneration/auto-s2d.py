import os
import glob
import traceback 
from collections import namedtuple

import numpy as np
import pandas as pd
import openpyxl
import imageio

# LIBERO Imports
from libero.libero.utils.mu_utils import MU_DICT, SCENE_DICT, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
import libero.libero.utils.task_generation_utils as task_utils
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

# ==============================================================================
# CONFIGURATION VARIABLES 
# ==============================================================================

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
except NameError:
    SCRIPT_DIR = os.getcwd()

OUTPUT_FOLDER = "/Users/cindy/experiments/LIBERO/libero/libero/bddl_files/libero_industry/S2D_cosmetics"
VIDEO_OUT_DIR = os.path.join(SCRIPT_DIR, "output_videos_s2d")
EXCEL_PATH = os.path.join(SCRIPT_DIR, "S2D_generation_tracker.xlsx")

# Video Settings
MAX_FRAMES = 150  
VIDEO_FPS = 30

# Mappings
INLAY_MAPPING = {
    1: ("cosmetics_inlay_a", "Novice"),
    2: ("cosmetics_inlay_b", "10%"),
    3: ("cosmetics_inlay_c", "5%"),
    4: ("cosmetics_inlay_d", "2%"),
}

SPEED_RANGES = {
    1: (0.02, 0.04),
    2: (0.04, 0.06),
    3: (0.06, 0.08)
}
SPEED_LABELS = {1: "0.02-0.04", 2: "0.04-0.06", 3: "0.06-0.08"}

MODES = {
    "S": "Steady",
    "P": "Pulse"
}

OBJECTS = ["pump_bottle_upright", "serum_bottle_upright", "cream_bottle_upright", "cream_jar_upright"]

LANGUAGE = "place the cosmetic products to the moving inlay with objects front facing up"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def init_excel_tracker():
    columns = ["Task ID", "Suite", "Precision (p)", "Speed Range (v)", "Actual Speed", 
               "Belt Yaw (rad)", "Bottle Yaw (rad)", "Pattern", "Axis", 
               "takt time", "kit", "Special Flag", "Status", "Video_File", "Error_Msg"]
    if os.path.exists(EXCEL_PATH):
        return pd.read_excel(EXCEL_PATH)
    else:
        return pd.DataFrame(columns=columns)

def simulate_and_save_video(bddl_path, video_save_path, task_seed):
    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": 256,
        "camera_widths": 256,
        "has_renderer": False,           
        "has_offscreen_renderer": True,  
        "use_camera_obs": True,
        "control_freq": 20,
        "renderer": "mujoco"      
    }
    try:
        env = OffScreenRenderEnv(**env_args)
        env.seed(task_seed)
        obs = env.reset()
        frames = []
        for _ in range(MAX_FRAMES):
            obs, _, _, _ = env.step([0.] * 7)
            frames.append(obs["agentview_image"][::-1]) 
        env.close()
        imageio.mimsave(video_save_path, frames, fps=VIDEO_FPS)
        return True, ""
    except Exception as e:
        return False, str(e)

def create_and_register_scene(scene_name, inlay_name, sampled_speed, mode_name, belt_yaw, bottle_yaw, axis_choice):
    class DynamicConveyorScene(InitialSceneTemplates):
        def __init__(self):
            fixture_num_info = {"industry_workbench": 1, "conveyor_ghost": 1}
            object_num_info = {obj: 1 for obj in OBJECTS}
            super().__init__(workspace_name="industry_workbench", fixture_num_info=fixture_num_info, object_num_info=object_num_info)

        def define_regions(self):
            print(f"[DEBUG] define_regions for {scene_name} -> Belt: {belt_yaw}, Bottle: {bottle_yaw}, Axis: {axis_choice}")
            belt_cx, belt_cy = 0.1, 0.0
            
            self.regions.update(
                self.get_region_dict(
                    region_centroid_xy=[belt_cx, belt_cy],
                    region_name="ghost_init_region",
                    target_name=self.workspace_name,
                    region_half_len=0.001,
                    yaw_rotation=(belt_yaw, belt_yaw + 0.0001)  
                )
            )
            
            perpendicular_dist = -0.3  
            group_tangent_shift = float(np.random.uniform(-0.1, 0.1)) 
            local_y_positions = [-0.15, -0.05, 0.05, 0.15]
            
            for idx, base_local_y in enumerate(local_y_positions):
                local_y = base_local_y + group_tangent_shift
                
                global_x = float(belt_cx + perpendicular_dist * np.cos(belt_yaw) - local_y * np.sin(belt_yaw))
                global_y = float(belt_cy + perpendicular_dist * np.sin(belt_yaw) + local_y * np.cos(belt_yaw))
                
                self.regions.update(
                    self.get_region_dict(
                        region_centroid_xy=[global_x, global_y],  
                        region_name=f"bottle_init_region_{idx}",
                        target_name=self.workspace_name,
                        region_half_len=0.02, 
                        yaw_rotation=(bottle_yaw, bottle_yaw + 0.0001)  
                    ) 
                )
            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

        @property
        def init_states(self):
            states = [
                ("On", "conveyor_ghost_1", "industry_workbench_ghost_init_region"),
            ]
            
            region_indices = [0, 1, 2, 3]
            np.random.shuffle(region_indices)
            
            for obj, region_idx in zip(OBJECTS, region_indices):
                states.append(("On", f"{obj}_1", f"industry_workbench_bottle_init_region_{region_idx}"))
            
            states.append(("LoadSlot", "conveyor_ghost_1", inlay_name, axis_choice, mode_name.lower(), str(sampled_speed)))
            
            return states

    # ==============================================================================
    # FIX: METACLASS CACHE BYPASS
    # Force the class to have a completely unique name every loop so LIBERO's 
    # internal registry cannot confuse it with previous iterations.
    # ==============================================================================
    unique_class_name = f"DynamicScene_{scene_name}"
    DynamicConveyorScene.__name__ = unique_class_name
    DynamicConveyorScene.__qualname__ = unique_class_name

    MU_DICT[scene_name.lower()] = DynamicConveyorScene
    SCENE_DICT["industry_workbench"] = [DynamicConveyorScene]

# ==============================================================================
# EXECUTION
# ==============================================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(VIDEO_OUT_DIR, exist_ok=True)
df_tracker = init_excel_tracker()

task_id = 1

print(f"Generating 24 BDDLs for S2D generalization. Tracking in {EXCEL_PATH}...")

for m_key, mode_name in MODES.items(): 
    for v_idx, speed_range in SPEED_RANGES.items(): 
        for p_idx, (inlay_name, precision_label) in INLAY_MAPPING.items(): 
            
            scene_name = f"S2D_{task_id:02d}_P{p_idx}V{v_idx}{m_key}"
            
            if task_id in df_tracker["Task ID"].values:
                if df_tracker.loc[df_tracker["Task ID"] == task_id, "Status"].iloc[0] == "Success":
                    print(f"Skipping Task {task_id:02d} | {scene_name} (Success)")
                    task_id += 1; continue

            print(f"Processing Task {task_id:02d} | {scene_name}...")
            task_utils.TASK_INFO.clear()
            
            # ==========================================================
            # EXACT SAMPLING FOR REPRODUCIBILITY
            # ==========================================================
            np.random.seed(task_id)
            sampled_speed = float(round(np.random.uniform(speed_range[0], speed_range[1]), 4))
            belt_yaw = float(round(np.random.uniform(-np.pi / 9, np.pi / 9), 4))
            bottle_yaw = float(round(np.random.uniform(-np.pi, np.pi), 4))
            axis_choice = "x" if task_id % 2 != 0 else "y"

            # ==========================================================
            # FIX: NUKE STALE BDDL FILES
            # Prevent the BDDL generator from silently skipping file creation 
            # by strictly deleting any old files with the current scene name.
            # ==========================================================
            for f in glob.glob(os.path.join(OUTPUT_FOLDER, f"{scene_name.upper()}*.bddl")):
                os.remove(f)
                print(f"[DEBUG] Purged stale BDDL file to force fresh generation: {os.path.basename(f)}")
            
            create_and_register_scene(scene_name, inlay_name, sampled_speed, mode_name, belt_yaw, bottle_yaw, axis_choice)
            
            goal_states = [("Routed", f"{obj}_1", "conveyor_ghost_1") for obj in OBJECTS]
            register_task_info(LANGUAGE, scene_name=scene_name, objects_of_interest=[], goal_states=goal_states)

            bddl_files, failures = generate_bddl_from_task_info(folder=OUTPUT_FOLDER)
            
            status, vid_filename, error_msg = "Fail", "", str(failures)
            if not failures and len(bddl_files) > 0:
                vid_filename = f"{scene_name}_val.mp4"
                
                ok, err = simulate_and_save_video(bddl_files[0], os.path.join(VIDEO_OUT_DIR, vid_filename), task_id)
                if ok: 
                    status, error_msg = "Success", ""
                else: 
                    error_msg = f"Sim Error: {err}"
            
            new_data = {
                "Task ID": task_id, 
                "Suite": "S2D", 
                "Precision (p)": precision_label, 
                "Speed Range (v)": SPEED_LABELS[v_idx],
                "Actual Speed": sampled_speed,          
                "Belt Yaw (rad)": belt_yaw,             
                "Bottle Yaw (rad)": bottle_yaw,         
                "Pattern": mode_name, 
                "Axis": axis_choice,       
                "kit": "baseline", 
                "Status": status, 
                "Video_File": vid_filename, 
                "Error_Msg": error_msg
            }
            
            df_tracker = df_tracker[df_tracker["Task ID"] != task_id]
            df_tracker = pd.concat([df_tracker, pd.DataFrame([new_data])], ignore_index=True)
            df_tracker.to_excel(EXCEL_PATH, index=False)
            
            task_id += 1

print("\n=== COMPLETE: 24 S2D BDDLs Generated ===")