import os
import re
import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion

from libero.libero.envs.base_object import register_object

import pathlib

from libero.libero.envs.base_object import (
    register_visual_change_object,
    register_object,
)
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info

from libero.libero.envs import OffScreenRenderEnv
from IPython.display import display
from PIL import Image

import torch
import torchvision

# for visualization
import imageio
from IPython.display import HTML
from base64 import b64encode

import re
import numpy as np
from libero.libero.envs import objects
from libero.libero.utils.bddl_generation_utils import *
from libero.libero.envs.objects import OBJECTS_DICT
from libero.libero.utils.object_utils import get_affordance_regions
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates

@register_mu(scene_type="industry_workbench")
class ConveyorTestScene(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {
            "industry_workbench": 1,
            "conveyor_ghost": 1,
            # Removed the static inlay here, because our "Repeat" predicate 
            # will generate them dynamically.
        }

        object_num_info = {
            "pump_bottle_upright": 1,

        }

        super().__init__(
            workspace_name="industry_workbench",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        # Region for the Ghost Conveyor
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0, 0.1],
                region_name="ghost_init_region",
                target_name=self.workspace_name,
                region_half_len=0.001,
                yaw_rotation=(0, 0)  
            )
        )
        
        # Region for the starting position of the Pump Bottle
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.2, -0.3],  
                region_name="bottle_init_region",
                target_name=self.workspace_name,
                region_half_len=0.03,                  
                yaw_rotation=(0, 0)  
            ) 
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )
        
    @property
    def init_states(self):
        states = [
            ("On", "conveyor_ghost_1", "industry_workbench_ghost_init_region"),
            ("On", "pump_bottle_upright_1", "industry_workbench_bottle_init_region"),
            ("LoadSlot",  "conveyor_ghost_1","bottle_groove","x", "steady", "0.05"),
        ]
        return states


# Register and test
scene_name = "conveyor_test_scene"
language = "place the pump bottle into the moving inlay a"
register_task_info(language,
                   scene_name=scene_name,
                   objects_of_interest=[],
                   goal_states=[
                       # The BDDL parses success if it hits the Master Site of the ghost conveyor.
                       # Our custom Python step() function will handle the strict coordinate evaluation!
                        ("Routed", "pump_bottle_upright_1", "conveyor_ghost_1"),
                   ],
)

# Generate BDDL
YOUR_BDDL_FILE_PATH = "/Users/cindy/experiments/LIBERO/libero/libero/bddl_files/libero_industry/sorting"
bddl_file_names, failures = generate_bddl_from_task_info(folder=YOUR_BDDL_FILE_PATH)

# Test in environment
env_args = {
    "bddl_file_name": bddl_file_names[0],
    "camera_heights": 256,
    "camera_widths": 256,
    "has_renderer": True,           # Enable on-screen rendering
    "has_offscreen_renderer": False, # Disable offscreen
    "use_camera_obs": True,
    "control_freq": 20,
    "renderer": "mujoco"      
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
obs = env.reset()

# Display the scene
# from PIL import Image
# Image.fromarray(obs["agentview_image"][::-1]).show()

# Collect images
obs_tensors = []
dummy_action = [0.] * 7
for step in range(100):
    obs, reward, done, info = env.step(dummy_action)
    obs_tensors.append(obs["agentview_image"])
    if done:
        break

env.close()

# Save as MP4
images = [img[::-1] for img in obs_tensors]
fps = 30
writer = imageio.get_writer('tmp_video.mp4', fps=fps)
for image in images:
    writer.append_data(image)
writer.close()

# Display in notebook
video_data = open("tmp_video.mp4", "rb").read()
video_tag = f'<video controls alt="test" src="data:video/mp4;base64,{b64encode(video_data).decode()}">'
HTML(data=video_tag)

