import numpy as np

def _get_qpos_adr(sim, obj_body_id):
    return sim.model.jnt_qposadr[sim.model.body_jntadr[obj_body_id]]

def _get_qvel_adr(sim, obj_body_id):
    return sim.model.jnt_dofadr[sim.model.body_jntadr[obj_body_id]]

class ConveyorSlotMixin:
    # 1. Add target_quat to the inputs
    def setup_physical_track(self, start_y, end_y, z_height, center_x, speed=0.02, mode="steady", target_quat=[1, 0, 0, 0]):
        self.belt_start_y = start_y
        self.belt_end_y = end_y
        self.belt_z = z_height
        self.belt_x = center_x
        self.conveyor_speed = speed
        self.conveyor_mode = mode
        self.target_quat = target_quat # Save the orientation
        self._track_initialized = False

    def _initialize_track(self):
        if not hasattr(self, 'dynamic_groove_names') or not self.dynamic_groove_names:
            return
            
        num_grooves = len(self.dynamic_groove_names)
        spacing = (self.belt_end_y - self.belt_start_y) / num_grooves

        for i, g_name in enumerate(self.dynamic_groove_names):
            g_id = self.obj_body_id[g_name]
            qpos_adr = _get_qpos_adr(self.sim, g_id)
            qvel_adr = _get_qvel_adr(self.sim, g_id)
            
            start_pos_y = self.belt_start_y + (i * spacing)
            
            self.sim.data.qpos[qpos_adr] = self.belt_x
            self.sim.data.qpos[qpos_adr+1] = start_pos_y
            self.sim.data.qpos[qpos_adr+2] = self.belt_z 
            
            # 2. Lock the initial orientation
            self.sim.data.qpos[qpos_adr+3:qpos_adr+7] = self.target_quat
            self.sim.data.qvel[qvel_adr:qvel_adr+6] = [0, 0, 0, 0, 0, 0]
            
        self._track_initialized = True

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        if hasattr(self, 'conveyor_speed'):
            if not getattr(self, '_track_initialized', False):
                self._initialize_track()
            self._apply_physical_track()

    def _apply_physical_track(self):
        """Forces the grooves to act like a rigid, unstoppable conveyor belt."""
        if not hasattr(self, 'dynamic_groove_names'):
            return

        current_speed = self.conveyor_speed
        if self.conveyor_mode == "pulse":
            current_speed = self.conveyor_speed if (self.sim.data.time % 2.0) < 1.0 else 0.0

        # --- NEW: Get the exact duration of the physics timestep ---
        dt = self.sim.model.opt.timestep

        for g_name in self.dynamic_groove_names:
            g_id = self.obj_body_id[g_name]
            qpos_adr = _get_qpos_adr(self.sim, g_id)
            qvel_adr = _get_qvel_adr(self.sim, g_id)
            
            # ==========================================
            # 1. KINEMATIC POSITION OVERRIDE
            # ==========================================
            # Force the Y position forward mathematically. MuJoCo cannot stop this.
            self.sim.data.qpos[qpos_adr+1] += current_speed * dt
            
            # Lock the other axes so it cannot tip over or derail
            self.sim.data.qpos[qpos_adr] = self.belt_x
            self.sim.data.qpos[qpos_adr+2] = self.belt_z
            self.sim.data.qpos[qpos_adr+3:qpos_adr+7] = self.target_quat
            
            # ==========================================
            # 2. SET VELOCITY FOR COLLISION ENGINE
            # ==========================================
            # We still set qvel so that if the robot or bottle hits the groove, 
            # MuJoCo knows how to calculate the impact force!
            self.sim.data.qvel[qvel_adr:qvel_adr+6] = [0, current_speed, 0, 0, 0, 0]
            
            # ==========================================
            # 3. LOOPING
            # ==========================================
            if self.sim.data.qpos[qpos_adr+1] > self.belt_end_y:
                self.sim.data.qpos[qpos_adr+1] = self.belt_start_y