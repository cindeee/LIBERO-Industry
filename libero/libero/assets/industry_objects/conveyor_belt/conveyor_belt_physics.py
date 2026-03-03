"""
Conveyor Belt Physics Implementation for LIBERO

Add this to your BDDLBaseDomain subclass to make the conveyor belt move objects.
"""

import numpy as np
import mujoco


class ConveyorBeltMixin:
    """
    Mixin to add conveyor belt functionality to LIBERO environments.
    
    Usage:
        class YourTask(ConveyorBeltMixin, BDDLBaseDomain):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.setup_conveyor_belt()
    """
    
    def setup_conveyor_belt(self, velocity=0.005, axis=(1, 0, 0), use_contact_detection=False):
        """
        Initialize conveyor belt parameters. Call in __init__ after super().__init__()
        
        Args:
            velocity: Belt velocity in m/s (default 0.005 = 0.5 cm/s)
            axis: Movement direction (default Y-axis)
            use_contact_detection: If True, use contact-based detection; if False, use site-based
        """
        self.conveyor_velocity = velocity
        self.conveyor_axis = np.array(axis)
        self.conveyor_body_name = None
        self.conveyor_site_names = [
            "conveyor_belt_1_contact_region"
        ]
        self.conveyor_use_contact = use_contact_detection
        self.conveyor_height_threshold = 0.04  # 6cm above belt
        
    def _pre_action(self, action, policy_step=False):
        """Override to apply conveyor belt physics before each action"""
        super()._pre_action(action, policy_step)
        self._apply_conveyor_belt_physics()
    
    def _apply_conveyor_belt_physics(self):
        """Apply velocity to objects in contact with conveyor belt"""
        if self.conveyor_use_contact:
            self._apply_contact_based_physics()
        else:
            self._apply_site_based_physics()
    
    def _apply_contact_based_physics(self):
        """Apply velocity using contact detection"""
        # Find all conveyor belt geoms
        conveyor_geom_ids = []
        for i in range(self.sim.model.ngeom):
            geom_name = self.sim.model.geom_id2name(i)
            if geom_name and 'conveyor_belt' in geom_name:
                conveyor_geom_ids.append(i)
        
        if not conveyor_geom_ids:
            return
        
        # Check each object for contact with conveyor
        for obj_name, obj in self.objects_dict.items():
            obj_body_id = self.obj_body_id[obj_name]
            
            # Check all contacts
            for i in range(self.sim.data.ncon):
                contact = self.sim.data.contact[i]
                
                # Check if contact involves conveyor geom and object
                geom1_body = self.sim.model.geom_bodyid[contact.geom1]
                geom2_body = self.sim.model.geom_bodyid[contact.geom2]
                
                if ((contact.geom1 in conveyor_geom_ids and geom2_body == obj_body_id) or
                    (contact.geom2 in conveyor_geom_ids and geom1_body == obj_body_id)):
                    self._apply_velocity_to_object(obj_body_id)
                    break
    
    def _apply_site_based_physics(self):
        """Apply velocity using site proximity detection"""
        site_positions = []
        for site_name in self.conveyor_site_names:
            try:
                site_id = self.sim.model.site_name2id(site_name)
                site_pos = self.sim.data.site_xpos[site_id]
                site_positions.append(site_pos)
            except:
                continue
        
        if not site_positions:
            return
        
        for obj_name, obj in self.objects_dict.items():
            obj_body_id = self.obj_body_id[obj_name]
            obj_pos = self.sim.data.body_xpos[obj_body_id]
            
            for site_pos in site_positions:
                horizontal_dist = np.linalg.norm(obj_pos[:2] - site_pos[:2])
                vertical_dist = abs(obj_pos[2] - site_pos[2])
                
                if horizontal_dist < 0.15 and vertical_dist < self.conveyor_height_threshold:
                    self._apply_velocity_to_object(obj_body_id)
                    break

    
    def _apply_velocity_to_object(self, obj_body_id):
        """Apply conveyor force to object"""
        force = self.conveyor_axis * self.conveyor_velocity * 50
        self.sim.data.xfrc_applied[obj_body_id, :3] = force
