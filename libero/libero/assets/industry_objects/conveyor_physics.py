import numpy as np


def _get_dof_adr(sim, obj_body_id):
    return sim.model.jnt_dofadr[sim.model.body_jntadr[obj_body_id]]


def _in_contact_with(sim, obj_body_id, geom_ids):
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1b = sim.model.geom_bodyid[c.geom1]
        g2b = sim.model.geom_bodyid[c.geom2]
        if (c.geom1 in geom_ids and g2b == obj_body_id) or \
           (c.geom2 in geom_ids and g1b == obj_body_id):
            return True
    return False


class ConveyorBeltMixin:
    """Straight conveyor belt: applies friction-like force via xfrc_applied."""

    def setup_conveyor_belt(self, velocity=0.01, axis=(1, 0, 0), kp=50.0):
        self.conveyor_velocity = velocity
        self.conveyor_axis = np.array(axis, dtype=float)
        self.conveyor_axis /= np.linalg.norm(self.conveyor_axis)
        self.conveyor_kp = kp

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        if hasattr(self, 'conveyor_velocity'):
            self._apply_conveyor_belt_physics()

    def _apply_conveyor_belt_physics(self):
        geom_ids = [
            i for i in range(self.sim.model.ngeom)
            if (name := self.sim.model.geom_id2name(i))
            and ("mesh_belt" in name or "contact_region" in name)
        ]
        if not geom_ids:
            return

        target_vel = self.conveyor_axis * self.conveyor_velocity
        for obj_name in self.objects_dict:
            obj_body_id = self.obj_body_id[obj_name]
            self.sim.data.xfrc_applied[obj_body_id] = 0.0
            if not _in_contact_with(self.sim, obj_body_id, geom_ids):
                continue
            dof = _get_dof_adr(self.sim, obj_body_id)
            vel_error = target_vel - self.sim.data.qvel[dof:dof + 3]
            mass = self.sim.model.body_mass[obj_body_id]
            self.sim.data.xfrc_applied[obj_body_id, :3] = self.conveyor_kp * mass * vel_error


class ConveyorCurvedMixin:
    """Curved conveyor belt using xfrc_applied for friction, torque for yaw,
    and centripetal force to prevent outward drift."""

    def setup_conveyor_curved(self, speed=0.02, arc_center_offset=(0.0, 0.0),
                              kp_linear=20.0, kp_angular=5.0):
        self.curved_speed = speed
        self.curved_center = np.array(arc_center_offset, dtype=float)
        self.curved_kp_linear = kp_linear
        self.curved_kp_angular = kp_angular
        self._curved_geom_ids = None

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        if hasattr(self, 'curved_speed'):
            self._apply_curved_conveyor_physics()

    def _get_curved_geom_ids(self):
        if self._curved_geom_ids is None:
            self._curved_geom_ids = [
                i for i in range(self.sim.model.ngeom)
                if (name := self.sim.model.geom_id2name(i)) and "contact_region" in name
            ]
        return self._curved_geom_ids

    def _apply_curved_conveyor_physics(self):
        geom_ids = self._get_curved_geom_ids()
        if not geom_ids:
            return

        try:
            conveyor_body_id = self.sim.model.body_name2id("conveyor_curved_1_conveyor_curved")
            conveyor_pos = self.sim.data.body_xpos[conveyor_body_id]
        except Exception:
            return

        for obj_name in self.objects_dict:
            obj_body_id = self.obj_body_id[obj_name]
            self.sim.data.xfrc_applied[obj_body_id] = 0.0
            if not _in_contact_with(self.sim, obj_body_id, geom_ids):
                continue

            obj_pos = self.sim.data.body_xpos[obj_body_id]
            dx = obj_pos[0] - conveyor_pos[0]
            dy = obj_pos[1] - conveyor_pos[1]
            r = np.sqrt(dx * dx + dy * dy)
            if r < 1e-4:
                continue

            omega = self.curved_speed / r
            target_linear_vel  = np.array([omega * dy, -omega * dx, 0.0])
            target_angular_vel = np.array([0.0, 0.0, -omega])

            mass = self.sim.model.body_mass[obj_body_id]
            dof  = _get_dof_adr(self.sim, obj_body_id)
            current_linear_vel  = self.sim.data.qvel[dof:dof + 3]
            current_angular_vel = self.sim.data.qvel[dof + 3:dof + 6]

            force  = mass * self.curved_kp_linear * (target_linear_vel - current_linear_vel)
            force += mass * (np.linalg.norm(current_linear_vel) ** 2 / r) * np.array([-dx/r, -dy/r, 0.0])
            torque = mass * self.curved_kp_angular * (target_angular_vel - current_angular_vel)

            self.sim.data.xfrc_applied[obj_body_id, :3] = force
            self.sim.data.xfrc_applied[obj_body_id, 3:6] = torque
