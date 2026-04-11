import os
import glob
import traceback 

import numpy as np
import pandas as pd
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

OUTPUT_FOLDER = "/Users/cindy/experiments/LIBERO/libero/libero/bddl_files/libero_industry/D2D_cosmetics"
VIDEO_OUT_DIR = os.path.join(SCRIPT_DIR, "output_videos_d2d")
EXCEL_PATH = os.path.join(SCRIPT_DIR, "D2D_48_Task_Benchmark.xlsx")

MAX_FRAMES = 150  
VIDEO_FPS = 30

PRECISIONS = {
    1: ("cosmetics_inlay_a", "Novice"),
    2: ("cosmetics_inlay_b", "10%"),
    3: ("cosmetics_inlay_c", "5%"),
    4: ("cosmetics_inlay_d", "2%")
}

PATTERN_COMBOS = [
    ("Steady", "Steady"),
    ("Pulse", "Pulse"),
    ("Steady", "Pulse"), 
    ("Pulse", "Steady")   
]

SPEED_RANGES = {
    "V1": (0.02, 0.04),
    "V2": (0.04, 0.06),
    "V3": (0.06, 0.08)
}

# 3 Dynamic Configurations for 48 total tasks
DYNA_CONFIGS = [
    {"Label": "Parallel-Same-FastSrc", "Direction": "Same", "V_Src": "V2", "V_Dst": "V1", "Diff": "Medium"},
    {"Label": "Parallel-Same-FastDst", "Direction": "Same", "V_Src": "V1", "V_Dst": "V2", "Diff": "Medium"},
    {"Label": "Parallel-Opposite", "Direction": "Opposite", "V_Src": "V2", "V_Dst": "V2", "Diff": "Hard"}
]

OBJECTS = ["pump_bottle_flat", "serum_bottle_flat", "cream_bottle_flat", "cream_jar_flat"]
LANGUAGE = "place the cosmetic objects from the conveyor into the moving inlay"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def init_excel_tracker():
    columns = ["Task ID", "Suite", "Precision", "Pattern Combo", "Movement Profile",
               "Src Speed", "Dst Speed", "Src Yaw", "Dst Yaw", 
               "Status", "Video_File", "Error_Msg"]
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

def create_and_register_scene(scene_name, inlay_name, src_pattern, dst_pattern, 
                              src_speed, dst_speed, src_yaw):
    class DynamicConveyorScene(InitialSceneTemplates):
        def __init__(self):
            fixture_num_info = {"industry_workbench": 1, "conveyor_belt": 1, "conveyor_ghost": 1}
            object_num_info = {obj: 1 for obj in OBJECTS}
            super().__init__(workspace_name="industry_workbench", fixture_num_info=fixture_num_info, object_num_info=object_num_info)

        def define_regions(self):
            print(f"[DEBUG] define_regions for {scene_name} -> Src Yaw: {src_yaw}, Dst Yaw: {np.pi/2}")
            
            # Region for the Ghost Conveyor (Destination)
            # Locked to np.pi/2. Epsilon applied to bypass the 0-range BDDL default bug.
            self.regions.update(
                self.get_region_dict(
                    region_centroid_xy=[0.2, 0],
                    region_name="ghost_init_region",
                    target_name=self.workspace_name,
                    region_half_len=0.001,
                    yaw_rotation=(np.pi/2, np.pi/2 + 0.0001)  
                )
            )
            
            # Region for the starting position of the Pump Bottle (Source Belt)
            # Dynamic yaw based on configuration. Epsilon applied.
            self.regions.update(
                self.get_region_dict(
                    region_centroid_xy=[-0.3, 0],  
                    region_name="belt_init_region",
                    target_name=self.workspace_name,
                    region_half_len=0.03,                  
                    yaw_rotation=(src_yaw, src_yaw + 0.0001) 
                ) 
            )
            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

        @property
        def init_states(self):
            states = [
                ("On", "conveyor_belt_1", "industry_workbench_belt_init_region"),
                ("On", "conveyor_ghost_1", "industry_workbench_ghost_init_region"),
                ("AtSpeed", "conveyor_belt_1", str(src_speed), src_pattern.lower()),
                ("LoadSlot", "conveyor_ghost_1", inlay_name, "x", dst_pattern.lower(), str(dst_speed)),
            ]
            
            for idx, obj in enumerate(OBJECTS):
                states.append(("On", f"{obj}_1", f"conveyor_belt_1_offset_region_{idx+1}"))
                
            return states

    # Metaclass Cache Bypass
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

print(f"Generating 48 D2D Benchmark Tasks. Tracking in {EXCEL_PATH}...")

for p_idx, (inlay_name, precision_label) in PRECISIONS.items():
    for src_p, dst_p in PATTERN_COMBOS:
        for config in DYNA_CONFIGS:
            
            scene_name = f"D2D_{task_id:02d}_{precision_label.replace('%','')}_{src_p[0]}{dst_p[0]}_{config['Label']}"
            
            if task_id in df_tracker["Task ID"].values:
                if df_tracker.loc[df_tracker["Task ID"] == task_id, "Status"].iloc[0] == "Success":
                    print(f"Skipping Task {task_id:02d} | {scene_name} (Success)")
                    task_id += 1; continue

            print(f"\nProcessing Task {task_id:02d} | {scene_name}...")
            task_utils.TASK_INFO.clear()
            
            np.random.seed(task_id)
            
            src_speed_bounds = SPEED_RANGES[config["V_Src"]]
            dst_speed_bounds = SPEED_RANGES[config["V_Dst"]]
            sampled_src_speed = float(round(np.random.uniform(src_speed_bounds[0], src_speed_bounds[1]), 4))
            sampled_dst_speed = float(round(np.random.uniform(dst_speed_bounds[0], dst_speed_bounds[1]), 4))
            
            # Apply layout conditions
            if config["Direction"] == "Opposite":
                src_yaw = float(round(-np.pi / 2, 4))
            else:
                src_yaw = float(round(np.pi / 2, 4))

            # Nuke stale bddl files 
            for f in glob.glob(os.path.join(OUTPUT_FOLDER, f"{scene_name.upper()}*.bddl")):
                os.remove(f)
            
            create_and_register_scene(
                scene_name, inlay_name, src_p, dst_p, 
                sampled_src_speed, sampled_dst_speed, src_yaw
            )
            
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
                "Suite": "D2D_Industry", 
                "Precision": precision_label, 
                "Pattern Combo": f"{src_p}-{dst_p}", 
                "Movement Profile": config["Label"],
                "Src Speed": sampled_src_speed,
                "Dst Speed": sampled_dst_speed,
                "Src Yaw": src_yaw,
                "Dst Yaw": np.pi / 2,
                "Status": status, 
                "Video_File": vid_filename, 
                "Error_Msg": error_msg
            }
            
            df_tracker = df_tracker[df_tracker["Task ID"] != task_id]
            df_tracker = pd.concat([df_tracker, pd.DataFrame([new_data])], ignore_index=True)
            df_tracker.to_excel(EXCEL_PATH, index=False)
            
            task_id += 1

print("\n=== COMPLETE: 48 D2D BDDLs Generated ===")