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

## Define your own objects
# You may want to include more object meshes of yours in the procedural generation pipeline.
# One option is to include your assets and define your object directly inside the LIBERO codebase. But this can make the whole thing messy. 
# Alternatively, you can define the objects inside your custom project repo folder, and define the object classes accordingly. 
# Note that you need to import your defined object classes whenever you run your own stuff. Libero codebase cannot automatically import those that are defined outside its repo.
# In the next, we provide an example, assuming you have object meses defined in `custom_assets`. In this example, we assume the generated pddl file will be saved in `custom_pddl`.

class CustomObjects(MujocoXMLObject):
    def __init__(self, custom_path, name, obj_name, joints=[dict(type="free", damping="0.0005")]):
        # make sure custom path is an absolute path
        assert(os.path.isabs(custom_path)), "Custom path must be an absolute path"
        # make sure the custom path is also an xml file
        assert(custom_path.endswith(".xml")), "Custom path must be an xml file"
        super().__init__(
            custom_path,
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.object_properties = {"vis_site_names": {}}




@register_object
class LiberoMug(CustomObjects):
    def __init__(self,
                 name="libero_mug",
                 obj_name="libero_mug",
                 ):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                "./", "industry_assets", "libero_mug", "libero_mug.xml"
            )),
            name=name,
            obj_name=obj_name, 
        )

        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None



# Define the scene
# Now we define the scene to load the previously defined objects. 
# For more information about the scene genration, please look at `procedural_creation_walkthrough.ipynb`. 
import re
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
            "conveyor_belt": 1,

        }

        object_num_info = {
            "box": 5,
        }

        super().__init__(
            workspace_name="industry_workbench",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        # 移到桌面右侧，独立区域（无重叠）
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0, 0.2],
                region_name="conveyor_belt_init_region",
                target_name=self.workspace_name,
                region_half_len=0.05,
                yaw_rotation=(-np.pi/2, -np.pi/2)  
            )
        )
        
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0, 0],
                region_name="region_1",
                target_name=self.workspace_name,
                region_half_len=0.05,
                yaw_rotation=(0, -np.pi/2) 
            )
        )
        
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0, 0],
                region_name="region_2",
                target_name=self.workspace_name,
                region_half_len=0.05,
                yaw_rotation=(0,-np.pi/2) 
            )
        )
        
        #goal 
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0, 0.2],  
                region_name="goal_region",
                target_name=self.workspace_name,
                region_half_len=0.05,  
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )
        
    @property
    def init_states(self):
        states = [
            ("On", "conveyor_belt_1", "industry_workbench_conveyor_belt_init_region"),
            ("On", "box_1", "conveyor_belt_1_top_region_1"),
            ("On", "box_2", "conveyor_belt_1_top_region_2"),
            ("On", "box_3", "conveyor_belt_1_top_region_3"),
            ("On", "box_4", "conveyor_belt_1_top_region_4"),
            ("On", "box_5", "conveyor_belt_1_top_region_5"),
        ]
        return states




# Register and test
scene_name = "conveyor_test_scene"
language = "place box from conveyor belt to goal"
register_task_info(language,
                   scene_name=scene_name,
                   objects_of_interest=[],
                   goal_states=[
                       ("On", "box_1", "industry_workbench_goal_region"),
                       ("On", "box_2", "industry_workbench_goal_region"),
                       ("On", "box_3", "industry_workbench_goal_region"),
                       ("On", "box_4", "industry_workbench_goal_region"),
                       ("On", "box_5", "industry_workbench_goal_region"),

                   ],
)

# Generate BDDL
YOUR_BDDL_FILE_PATH = "./industry_pddl"
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
for step in range(200):
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

