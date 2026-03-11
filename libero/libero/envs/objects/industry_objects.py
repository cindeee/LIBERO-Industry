import os
import re
import numpy as np

from robosuite.models.objects import MujocoXMLObject

import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

from libero.libero.envs.base_object import (
    register_visual_change_object,
    register_object,
)

class IndustryObject(MujocoXMLObject):
    def __init__(self, name, obj_name, joints=[dict(type="free", damping="0.0005")]):
        super().__init__(
            os.path.join(
                str(absolute_path), f"assets/industry_objects/{obj_name}/{obj_name}.xml"
            ),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (0, 0)
        self.rotation_axis = None
        self.object_properties = {
            "vis_site_names": {}
            }

@register_object
class ConveyorBelt(IndustryObject):
    def __init__(
        self,
        name="conveyor_belt",
        obj_name="conveyor_belt",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {
            "x": (0, 0),
            "y": (0, 0),
            "z": (0, 0),
        }
        self.rotation_axis = None

@register_object
class ConveyorCurved(IndustryObject):
    def __init__(
        self,
        name="conveyor_curved",
        obj_name="conveyor_curved",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {
            "x": (0, 0),
            "y": (0, 0),
            "z": (0, 0),
        }
        self.rotation_axis = None
        
@register_object
class Box(IndustryObject):
    def __init__(
        self,
        name="box",
        obj_name="box",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {
            "x": (0, 0),
            "y": (0, 0),
            "z": (0, 2 * np.pi),
        }
        self.rotation_axis = None
        self.z_offset = 0.01


@register_object
class LiberoGreyTray(IndustryObject):
    def __init__(self,
                 name="grey_tray",
                 obj_name="grey_tray",
                 ):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                "./", "industry_assets", "grey_tray", "grey_tray.xml"
            )),
            name=name,
            obj_name=obj_name,
        )
        
        # Rotation settings (tray is typically flat)
        self.rotation = {
            "x": (0, 0),
            "y": (0, 0),
            "z": (0, 0),
        }
        self.rotation_axis = None
        
        # Optional: Initial z offset
        self.z_offset = 0.005