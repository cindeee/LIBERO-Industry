"""
Conveyor Belt Physics Implementation for LIBERO
"""

import numpy as np


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
        Initialize conveyor belt parameters.
        
        Args:
            velocity: Belt velocity in m/s (default 0.005 = 0.5 cm/s)
            axis: Movement direction (default X-axis)
            use_contact_detection: If True, use contact-based; if False, use site-based
        """
        self.conveyor_velocity = velocity
        self.conveyor_axis = np.array(axis)
        self.conveyor_site_names = ["conveyor_belt_1_contact_region"]
        self.conveyor_use_contact = use_contact_detection
        self.conveyor_height_threshold = 0.06
        
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
        conveyor_geom_ids = []
        for i in range(self.sim.model.ngeom):
            geom_name = self.sim.model.geom_id2name(i)
            if geom_name and 'conveyor_belt' in geom_name:
                conveyor_geom_ids.append(i)
        
        if not conveyor_geom_ids:
            return
        
        for obj_name in self.objects_dict.keys():
            obj_body_id = self.obj_body_id[obj_name]
            
            for i in range(self.sim.data.ncon):
                contact = self.sim.data.contact[i]
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
        
        for obj_name in self.objects_dict.keys():
            obj_body_id = self.obj_body_id[obj_name]
            obj_pos = self.sim.data.body_xpos[obj_body_id]
            
            for site_pos in site_positions:
                horizontal_dist = np.linalg.norm(obj_pos[:2] - site_pos[:2])
                vertical_dist = abs(obj_pos[2] - site_pos[2])
                
                if horizontal_dist < 0.15 and vertical_dist < self.conveyor_height_threshold:
                    self._apply_velocity_to_object(obj_body_id)
                    break
    
    def _apply_velocity_to_object(self, obj_body_id):
        """Apply conveyor velocity to object"""
        joint_adr = self.sim.model.body_jntadr[obj_body_id]
        if joint_adr >= 0:
            qvel_adr = self.sim.model.jnt_qposadr[joint_adr]
            current_vel = self.sim.data.qvel[qvel_adr:qvel_adr+3].copy()
            target_vel = self.conveyor_axis * self.conveyor_velocity
            # Smooth blend: 90% current + 10% target
            self.sim.data.qvel[qvel_adr:qvel_adr+3] = 0.9 * current_vel + 0.1 * target_vel
