import os
import numpy as np
import itertools
import pandas as pd
import openpyxl  # Added to ensure Excel reading/writing works
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

OUTPUT_FOLDER = "/Users/cindy/experiments/LIBERO/libero/libero/bddl_files/libero_industry/D2S_cosmetics"
VIDEO_OUT_DIR = os.path.join(SCRIPT_DIR, "output_videos")
EXCEL_PATH = os.path.join(SCRIPT_DIR, "D2S_generation_tracker.xlsx")

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

OBJECTS = ["pump_bottle_flat", "serum_bottle_flat", "cream_bottle_flat", "cream_jar_flat"]

OFFSET_REGIONS = [
    "conveyor_belt_1_offset_region_1",
    "conveyor_belt_1_offset_region_2",
    "conveyor_belt_1_offset_region_3",
    "conveyor_belt_1_offset_region_4"
]

LANGUAGE = "place the cosmetic products from conveyor belt to the inlay with objects front facing up"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def init_excel_tracker():
    columns = ["Task ID", "Suite", "Precision (p)", "Speed Range (v)", "Actual Speed", 
               "Belt Yaw (rad)", "Inlay Yaw (rad)", "Pattern", 
               "takt time", "kit", "Special Flag", "Status", "Video_File", "Error_Msg"]
    if os.path.exists(EXCEL_PATH):
        return pd.read_excel(EXCEL_PATH)
    else:
        return pd.DataFrame(columns=columns)

def simulate_and_save_video(bddl_path, video_save_path):
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
        # Note: env.seed(0) resets the global numpy random seed during simulation.
        env.seed(0)
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

def create_and_register_scene(scene_name, inlay_name, sampled_speed, mode_name, permutation, belt_yaw, inlay_yaw):
    class DynamicConveyorScene(InitialSceneTemplates):
        def __init__(self):
            fixture_num_info = {"industry_workbench": 1, "conveyor_belt": 1, inlay_name: 1}
            object_num_info = {obj: 1 for obj in OBJECTS}
            super().__init__(workspace_name="industry_workbench", fixture_num_info=fixture_num_info, object_num_info=object_num_info)

        def define_regions(self):
            self.regions.update(self.get_region_dict([0.1, 0], "conveyor_belt_init_region", region_half_len=0.05, yaw_rotation=(belt_yaw, belt_yaw)))
            self.regions.update(self.get_region_dict([-0.2, 0], "carrier_region", region_half_len=0.05, yaw_rotation=(inlay_yaw, inlay_yaw)))
            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

        @property
        def init_states(self):
            states = [
                ("On", "conveyor_belt_1", "industry_workbench_conveyor_belt_init_region"),
                ("AtSpeed", "conveyor_belt_1", str(sampled_speed), mode_name.lower()),
                ("On", f"{inlay_name}_1", "industry_workbench_carrier_region"),
            ]
            for obj_name, region_name in zip(permutation, OFFSET_REGIONS):
                states.append(("On", obj_name, region_name))
            return states

    MU_DICT[scene_name.lower()] = DynamicConveyorScene
    SCENE_DICT.setdefault("industry_workbench", []).append(DynamicConveyorScene)

# ==============================================================================
# EXECUTION
# ==============================================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(VIDEO_OUT_DIR, exist_ok=True)
df_tracker = init_excel_tracker()

# Create an independent random state to avoid interference from env.seed(0)
task_rng = np.random.RandomState(42)

# Pre-calculate 24 unique permutations
object_instances = [f"{obj}_1" for obj in OBJECTS]
PERMUTATIONS = list(itertools.permutations(object_instances))

task_id = 1
perm_idx = 0

print(f"Generating 24 BDDLs for generalization. Tracking in {EXCEL_PATH}...")

for m_key, mode_name in MODES.items(): 
    for v_idx, speed_range in SPEED_RANGES.items(): 
        for p_idx, (inlay_name, precision_label) in INLAY_MAPPING.items(): 
            
            scene_name = f"D2S_{task_id:02d}_P{p_idx}V{v_idx}{m_key}"
            
            # Skip if already successful AND files exist
            if task_id in df_tracker["Task ID"].values:
                row = df_tracker.loc[df_tracker["Task ID"] == task_id].iloc[0]
                if row["Status"] == "Success":
                    vid_file = row["Video_File"]
                    lang_suffix = "_".join(LANGUAGE.lower().split(" "))
                    bddl_file = f"{scene_name.upper()}_{lang_suffix}.bddl"
                    
                    vid_exists = os.path.exists(os.path.join(VIDEO_OUT_DIR, vid_file))
                    bddl_exists = os.path.exists(os.path.join(OUTPUT_FOLDER, bddl_file))
                    
                    if vid_exists and bddl_exists:
                        print(f"Skipping Task {task_id:02d} | {scene_name} (Success & Files Exist)")
                        task_id += 1; perm_idx += 1; continue
                    else:
                        print(f"Files missing for Task {task_id:02d} (BDDL: {bddl_exists}, Video: {vid_exists}), regenerating...")

            print(f"Processing Task {task_id:02d} | {scene_name}...")
            task_utils.TASK_INFO.clear()
            
            # ==========================================================
            # EXACT SAMPLING FOR REPRODUCIBILITY
            # We now use task_rng to prevent env.seed(0) from overwriting
            # ==========================================================
            sampled_speed = round(task_rng.uniform(speed_range[0], speed_range[1]), 4)
            belt_yaw = round(task_rng.uniform(4 * np.pi / 9, 5 * np.pi / 9), 4)
            inlay_yaw = round(task_rng.uniform(-np.pi, np.pi), 4)
            current_perm = PERMUTATIONS[perm_idx % len(PERMUTATIONS)]
            
            # Register with EXACT sampled values
            create_and_register_scene(scene_name, inlay_name, sampled_speed, mode_name, current_perm, belt_yaw, inlay_yaw)
            
            goal_states = [("On", f"{obj}_1", f"{inlay_name}_1_goal_area_{obj.replace('_flat','')}") for obj in OBJECTS]
            register_task_info(LANGUAGE, scene_name=scene_name, objects_of_interest=[], goal_states=goal_states)

            bddl_files, failures = generate_bddl_from_task_info(folder=OUTPUT_FOLDER)
            
            status, vid_filename, error_msg = "Fail", "", str(failures)
            if not failures and len(bddl_files) > 0:
                vid_filename = f"{scene_name}_val.mp4"
                ok, err = simulate_and_save_video(bddl_files[0], os.path.join(VIDEO_OUT_DIR, vid_filename))
                if ok: status, error_msg = "Success", ""
                else: error_msg = f"Sim Error: {err}"
            
            # Update Excel with EXACT numerical values
            new_data = {
                "Task ID": task_id, 
                "Suite": "D2S", 
                "Precision (p)": precision_label, 
                "Speed Range (v)": SPEED_LABELS[v_idx],
                "Actual Speed": sampled_speed,         
                "Belt Yaw (rad)": belt_yaw,             
                "Inlay Yaw (rad)": inlay_yaw,           
                "Pattern": mode_name, 
                "kit": "baseline", 
                "Status": status, 
                "Video_File": vid_filename, 
                "Error_Msg": error_msg
            }
            
            df_tracker = df_tracker[df_tracker["Task ID"] != task_id] # Drop old entry if retry
            df_tracker = pd.concat([df_tracker, pd.DataFrame([new_data])], ignore_index=True)
            df_tracker.to_excel(EXCEL_PATH, index=False)
            
            task_id += 1; perm_idx += 1

print("\n=== COMPLETE: 24 BDDLs Generated ===")