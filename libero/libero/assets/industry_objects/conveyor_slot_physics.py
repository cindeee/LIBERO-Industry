import numpy as np
import robosuite.utils.transform_utils as T

def _get_qpos_adr(sim, obj_body_id):
    return sim.model.jnt_qposadr[sim.model.body_jntadr[obj_body_id]]

def _get_qvel_adr(sim, obj_body_id):
    return sim.model.jnt_dofadr[sim.model.body_jntadr[obj_body_id]]

class ConveyorSlotMixin:
    # Add groove_bottom_offset to the inputs
    def setup_physical_track(self, conveyor_name, speed=0.02, mode="steady", groove_quat=[1, 0, 0, 0], groove_bottom_offset=-0.005, groove_step=None):
        self.target_conveyor_name = conveyor_name
        self.conveyor_speed = speed
        self.conveyor_mode = mode
        self.local_groove_quat = groove_quat 
        self.groove_bottom_offset = groove_bottom_offset
        self.groove_step = groove_step  
        self._track_initialized = False

    def _initialize_track(self):
        if not hasattr(self, 'dynamic_groove_names') or not self.dynamic_groove_names:
            return

        # 1. Grab exact world state of the ghost conveyor
        conv_id = self.obj_body_id[self.target_conveyor_name]
        conv_pos = self.sim.data.body_xpos[conv_id]
        conv_quat = self.sim.data.body_xquat[conv_id]
        conv_mat = T.quat2mat(T.convert_quat(conv_quat, to="xyzw"))

        # 2. DYNAMIC Z-HEIGHT CALCULATION (No hardcoding!)
        # Fetch the absolute table top height from the linked arena
        table_z = self.model.mujoco_arena.table_top_abs[2]
        hover_margin = 0.001 
        
        # Absolute Z = Table Top - (Negative Bottom Offset) + Hover
        self.abs_z_height = table_z - self.groove_bottom_offset + hover_margin

        # 3. Pull track length from the XML site
        site_name = f"{self.target_conveyor_name}_ghost_active_region"
        try:
            site_id = self.sim.model.site_name2id(site_name)
            site_size = self.sim.model.site_size[site_id]
            half_length = site_size[1]
        except Exception:
            half_length = 0.4

        self.belt_start_y_local = -half_length
        self.belt_end_y_local = half_length
        
        # 4. Calculate the global rotation
        local_quat_np = np.array(self.local_groove_quat)
        local_mat = T.quat2mat(T.convert_quat(local_quat_np, to="xyzw"))
        world_groove_mat = conv_mat.dot(local_mat)
        world_groove_quat_xyzw = T.mat2quat(world_groove_mat)
        self.world_groove_quat_wxyz = T.convert_quat(world_groove_quat_xyzw, to="wxyz")

        num_grooves = len(self.dynamic_groove_names)
        if hasattr(self, 'groove_step') and self.groove_step is not None:
            spacing = self.groove_step
            # Shrink the physical wrap-around bounds to perfectly fit the math
            perfect_length = num_grooves * spacing
            self.belt_start_y_local = -perfect_length / 2.0
            self.belt_end_y_local = perfect_length / 2.0
        else:
            spacing = (self.belt_end_y_local - self.belt_start_y_local) / num_grooves
        #
        print(f"\n[DEBUG PHYSICS] Belt Start: {self.belt_start_y_local}, Belt End: {self.belt_end_y_local}")
        print(f"[DEBUG PHYSICS] Placed {num_grooves} grooves with spacing: {spacing}")

        for i, g_name in enumerate(self.dynamic_groove_names):
            g_id = self.obj_body_id[g_name]
            qpos_adr = _get_qpos_adr(self.sim, g_id)
            qvel_adr = _get_qvel_adr(self.sim, g_id)
            
            # Local X/Y on the belt (Z is 0 locally because we use absolute Z later)
            local_y = self.belt_start_y_local + (i * spacing)
            local_xy = np.array([0, local_y, 0])
            
            # Transform local XY to global coordinates based on ghost position
            world_pos = conv_pos + conv_mat.dot(local_xy)
            
            # OVERRIDE the Z coordinate with our perfect absolute Z
            world_pos[2] = self.abs_z_height
            
            self.sim.data.qpos[qpos_adr:qpos_adr+3] = world_pos
            self.sim.data.qpos[qpos_adr+3:qpos_adr+7] = self.world_groove_quat_wxyz
            self.sim.data.qvel[qvel_adr:qvel_adr+6] = 0
            
        self.forward_vec = conv_mat.dot(np.array([0, 1, 0])) 
        self._track_initialized = True

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        if hasattr(self, 'conveyor_speed'):
            if not getattr(self, '_track_initialized', False):
                self._initialize_track()
            self._apply_physical_track()

    def _apply_physical_track(self):
        if not hasattr(self, 'dynamic_groove_names'):
            return

        current_speed = self.conveyor_speed
        if self.conveyor_mode == "pulse":
            current_speed = self.conveyor_speed if (self.sim.data.time % 2.0) < 1.0 else 0.0

        dt = self.sim.model.opt.timestep
        

        conv_id = self.obj_body_id[self.target_conveyor_name]
        conv_pos = self.sim.data.body_xpos[conv_id]

        for g_name in self.dynamic_groove_names:
            g_id = self.obj_body_id[g_name]
            qpos_adr = _get_qpos_adr(self.sim, g_id)
            qvel_adr = _get_qvel_adr(self.sim, g_id)
            
            world_pos = self.sim.data.qpos[qpos_adr:qpos_adr+3]
                        
            # CHECK BOUNDS
            relative_pos = world_pos - conv_pos
            local_y = np.dot(relative_pos, self.forward_vec)
            
            if local_y > self.belt_end_y_local:
                world_pos -= self.forward_vec * (self.belt_end_y_local - self.belt_start_y_local)
            elif local_y < self.belt_start_y_local:
                world_pos += self.forward_vec * (self.belt_end_y_local - self.belt_start_y_local)

            # Ensure Z never drifts from physics impacts
            world_pos[2] = self.abs_z_height

            self.sim.data.qpos[qpos_adr:qpos_adr+3] = world_pos
            self.sim.data.qpos[qpos_adr+3:qpos_adr+7] = self.world_groove_quat_wxyz
            
            vel_vec = self.forward_vec * current_speed
            self.sim.data.qvel[qvel_adr:qvel_adr+3] = vel_vec
            self.sim.data.qvel[qvel_adr+3:qvel_adr+6] = [0, 0, 0]