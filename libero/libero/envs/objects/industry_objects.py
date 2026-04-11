import os
import re
import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import string_to_array

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
        
        # --- NEW: Automatically extract geom size from XML ---
        geom = self.worldbody.find(".//geom")
        if geom is not None and geom.get("size") is not None:
            # size is usually half-extents: [x, y, z]
            self.parsed_size = string_to_array(geom.get("size"))
        else:
            self.parsed_size = np.array([0.05, 0.05, 0.01]) # Fallback

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
        self.z_offset = 0.098

# @register_object
# class ConveyorCurved(IndustryObject):
#     def __init__(
#         self,
#         name="conveyor_curved",
#         obj_name="conveyor_curved",
#         joints=[dict(type="free", damping="0.0005")],
#     ):
#         super().__init__(name, obj_name, joints)
#         self.rotation = {
#             "x": (0, 0),
#             "y": (0, 0),
#             "z": (0, 0),
#         }
#         self.rotation_axis = None

@register_object
class ConveyorGhost(IndustryObject):
    def __init__(
        self,
        name="conveyor_ghost",
        obj_name="conveyor_ghost",
        joints=[], 
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {
            "x": (0, 0),
            "y": (0, 0),
            "z": (0, 0),
        }
        self.rotation_axis = None
        self.z_offset = 0.005
        
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

# @register_object
# class PumpBottle(IndustryObject):
#     def __init__(
#         self,
#         name="pump_bottle",
#         obj_name="pump_bottle",
#         joints=[dict(type="free", damping="0.0005")],
#     ):
#         super().__init__(name, obj_name, joints)
#         # Force a 90-degree rotation around the X-axis upon initialization
#         self.rotation = {
#             "x": (0, 0), 
#             "y": (0, 0),
#             "z": (0, 2 * np.pi),
#         }
#         self.rotation_axis = None
#         # Give it a slightly higher offset so the side of the bottle clears the belt
#         self.z_offset = 0.02

@register_object
class PumpBottleUpright(IndustryObject):
    def __init__(
        self, 
        name="pump_bottle_upright", 
        obj_name="pump_bottle", 
        joints=[dict(type="free", 
        damping="0.0005")]
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 2 * np.pi)}
        self.rotation_axis = 'z'
        self.z_offset = 0.06



@register_object
class PumpBottleFlat(IndustryObject):
    def __init__(self, 
        name="pump_bottle_flat", 
        obj_name="pump_bottle", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, joints)
        # self.init_quat = np.array([0.7071068, 0, 0, 0.7071068])
        # Force the sampler to rotate it 90 degrees on the X-axis so it lies flat
        self.rotation = {"x": (np.pi/2, np.pi/2), "y": (0,  np.pi/2), "z": (0, 2 * np.pi)}
        
        self.z_offset = 0.02


@register_object
class CreamBottleUpright(IndustryObject):
    def __init__(
        self, 
        name="cream_bottle_upright", 
        obj_name="cream_bottle", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 2 * np.pi)}
        self.rotation_axis = 'z'
        self.z_offset = 0.046  # Half of 0.091

@register_object
class CreamBottleFlat(IndustryObject):
    def __init__(
        self, 
        name="cream_bottle_flat", 
        obj_name="cream_bottle", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, joints)
        # Force 90-degree X rotation to lie flat
        self.rotation = {"x": (np.pi/2, np.pi/2), "y": (0, np.pi/2), "z": (0, 2 * np.pi)}
        self.z_offset = 0.013  
@register_object
class CreamJarUpright(IndustryObject):
    def __init__(
        self, 
        name="cream_jar_upright", 
        obj_name="cream_jar", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 2 * np.pi)}
        self.rotation_axis = 'z'
        self.z_offset = 0.026  # Half of 0.0515

@register_object
class CreamJarFlat(IndustryObject):
    def __init__(
        self, 
        name="cream_jar_flat", 
        obj_name="cream_jar", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (np.pi/2, np.pi/2), "y": (0, np.pi/2), "z": (0, 2 * np.pi)}
        self.z_offset = 0.034  # Half of 0.0668 (diameter)

@register_object
class SerumBottleUpright(IndustryObject):
    def __init__(
        self, 
        name="serum_bottle_upright", 
        obj_name="serum_bottle", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 2 * np.pi)}
        self.rotation_axis = 'z'
        self.z_offset = 0.043  # Half of 0.0846

@register_object
class SerumBottleFlat(IndustryObject):
    def __init__(
        self, 
        name="serum_bottle_flat", 
        obj_name="serum_bottle", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (np.pi/2, np.pi/2), "y": (0, np.pi/2), "z": (0, 2 * np.pi)}
        self.z_offset = 0.014  # Half of 0.0273 (diameter)

@register_object
class CosmeticsInlayA(IndustryObject):
    def __init__(
        self,
        name="cosmetics_inlay_a", 
        obj_name="cosmetics_inlay_a",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.025  # Half of 0.049

@register_object
class CosmeticsInlayB(IndustryObject):
    def __init__(
        self,
        name="cosmetics_inlay_b", 
        obj_name="cosmetics_inlay_b",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.025

@register_object
class CosmeticsInlayC(IndustryObject):
    def __init__(
        self,
        name="cosmetics_inlay_c", 
        obj_name="cosmetics_inlay_c",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.025

@register_object
class CosmeticsInlayD(IndustryObject):
    def __init__(
        self,
        name="cosmetics_inlay_d", 
        obj_name="cosmetics_inlay_d",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.025

@register_object
class PumpBottlePlate(IndustryObject):
    def __init__(
        self,
        name="pump_bottle_plate", 
        obj_name="pump_bottle_plate",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        
        self.rotation = {
            "x": (0, 0),
            "y": (0, 0),
            "z": (0, 0),
        }
        self.rotation_axis = None
        self.z_offset = 0.01

# pump
@register_object
class PumpBottleInlayA(IndustryObject):
    def __init__(
        self,
        name="pump_bottle_inlay_a", 
        obj_name="pump_bottle_inlay_a",
        joints=[dict(type="free", damping="0.0005")],
    ):
        # Override the folder_name to group them together in the assets directory!
        super().__init__(name, obj_name, joints)
        
        self.rotation = {
            "x": (0, 0),
            "y": (0, 0),
            "z": (0, 0),
        }
        self.rotation_axis = None
        self.z_offset = 0.06

@register_object
class PumpBottleInlayADyn(IndustryObject):
    def __init__(
        self,
        name="pump_bottle_inlay_a_dyn", 
        obj_name="pump_bottle_inlay_a_dyn",
        # CHANGE THIS TO SLIDE JOINT ALONG Y-AXIS
        joints=[dict(type="slide", axis="0 1 0", damping="5.0")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.01 

@register_object
class PumpBottleInlayB(IndustryObject):
    def __init__(
        self,
        name="pump_bottle_inlay_b",
        obj_name="pump_bottle_inlay_b",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.06


@register_object
class PumpBottleInlayC(IndustryObject):
    def __init__(
        self,
        name="pump_bottle_inlay_c",
        obj_name="pump_bottle_inlay_c",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {
            "x": (0, 0),
            "y": (0, 0),
            "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.06


@register_object
class PumpBottleInlayD(IndustryObject):
    def __init__(
        self,
        name="pump_bottle_inlay_d",
        obj_name="pump_bottle_inlay_d",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.08

# 3C electronics
@register_object
class MouseNaga(IndustryObject):
    def __init__(
        self,
        name="mouse_naga",
        obj_name="mouse_naga",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.05


@register_object
class MouseTaipan(IndustryObject):
    def __init__(
        self,
        name="mouse_taipan",
        obj_name="mouse_taipan",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.05



@register_object
class MouseMat(IndustryObject):
    def __init__(
        self,
        name="bottle_groove",
        obj_name="bottle_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.05

@register_object
class MouseGroove(IndustryObject):
    def __init__(
        self,
        name="bottle_groove",
        obj_name="bottle_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.05

@register_object
class BottleGroove(IndustryObject):
    def __init__(
        self,
        name="bottle_groove",
        obj_name="bottle_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.05

@register_object
class MouseInlay(IndustryObject):
    def __init__(
        self,
        name="bottle_groove",
        obj_name="bottle_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        

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