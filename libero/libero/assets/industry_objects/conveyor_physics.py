import numpy as np


def _get_dof_adr(sim, obj_body_id):
    return sim.model.jnt_dofadr[sim.model.body_jntadr[obj_body_id]]


def _get_body_geom_ids(sim, body_id):
    """Get all geom IDs belonging to a body and its children."""
    geom_ids = set()
    for i in range(sim.model.ngeom):
        bid = sim.model.geom_bodyid[i]
        while bid != 0:
            if bid == body_id:
                geom_ids.add(i)
                break
            bid = sim.model.body_parentid[bid]
    return geom_ids


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
    """Straight conveyor belt: directly sets qvel for uniform motion."""

    def setup_conveyor_belt(self, velocity=0.05, local_axis=(0, 1, 0)):
        """Drive objects along the belt using velocity in world frame.

        ``local_axis`` is the travel direction in the **conveyor_belt** body's
        local frame (the MJCF contact strip is long along local ``Y`` in
        ``conveyor_belt.xml``). It is rotated to world coordinates using that
        body's ``xmat`` each step (placement yaw, Blender alignment quat, etc.).
        """
        self.conveyor_velocity = velocity
        self._conveyor_local_axis = np.array(local_axis, dtype=float)
        n = np.linalg.norm(self._conveyor_local_axis)
        if n < 1e-8:
            raise ValueError("conveyor local_axis must be non-zero")
        self._conveyor_local_axis /= n
        self._belt_geom_ids = None
        self._belt_body_id = None

    def _get_belt_geom_ids(self):
        if self._belt_geom_ids is None:
            for name in self.fixtures_dict:
                if 'conveyor_belt' in name:
                    root_id = self.obj_body_id[name]
                    self._belt_geom_ids = _get_body_geom_ids(self.sim, root_id)
                    # Geoms sit on nested ``conveyor_belt``, not on free-joint ``main``; use that
                    # body's xmat so XML orientation (e.g. Blender fix quat) affects belt direction.
                    belt_bid = None
                    for gid in self._belt_geom_ids:
                        gname = self.sim.model.geom_id2name(gid)
                        if gname is not None and gname.endswith("contact_region"):
                            belt_bid = self.sim.model.geom_bodyid[gid]
                            break
                    if belt_bid is None and self._belt_geom_ids:
                        belt_bid = self.sim.model.geom_bodyid[min(self._belt_geom_ids)]
                    self._belt_body_id = belt_bid
                    break
        return self._belt_geom_ids

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        if hasattr(self, 'conveyor_velocity'):
            self._apply_conveyor_belt_physics()

    def _apply_conveyor_belt_physics(self):
        geom_ids = self._get_belt_geom_ids()
        if not geom_ids or self._belt_body_id is None:
            return

        R = np.array(self.sim.data.body_xmat[self._belt_body_id]).reshape(3, 3)
        world_axis = R @ self._conveyor_local_axis
        wn = np.linalg.norm(world_axis)
        if wn < 1e-8:
            return
        world_axis /= wn
        target_vel = world_axis * self.conveyor_velocity
        for obj_name in self.objects_dict:
            obj_body_id = self.obj_body_id[obj_name]
            if not _in_contact_with(self.sim, obj_body_id, geom_ids):
                continue
            dof = _get_dof_adr(self.sim, obj_body_id)
            # Directly set linear velocity along belt axis, preserve vertical
            self.sim.data.qvel[dof:dof + 3] = target_vel


class ConveyorCurvedMixin:
    """Curved conveyor belt: directly sets qvel for uniform arc motion."""

    def setup_conveyor_curved(self, speed=0.035, arc_center_offset=(0.0, 0.0)):
        self.curved_speed = speed
        self.curved_center = np.array(arc_center_offset, dtype=float)
        self._curved_geom_ids = None
        self._curved_body_id = None

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        if hasattr(self, 'curved_speed'):
            self._apply_curved_conveyor_physics()

    def _get_curved_geom_ids(self):
        if self._curved_geom_ids is None:
            for name in self.fixtures_dict:
                if 'conveyor_curved' in name:
                    self._curved_body_id = self.obj_body_id[name]
                    self._curved_geom_ids = _get_body_geom_ids(self.sim, self._curved_body_id)
                    break
        return self._curved_geom_ids

    def _apply_curved_conveyor_physics(self):
        geom_ids = self._get_curved_geom_ids()
        if not geom_ids or self._curved_body_id is None:
            return

        conveyor_pos = self.sim.data.body_xpos[self._curved_body_id]

        for obj_name in self.objects_dict:
            obj_body_id = self.obj_body_id[obj_name]
            if not _in_contact_with(self.sim, obj_body_id, geom_ids):
                continue

            obj_pos = self.sim.data.body_xpos[obj_body_id]
            dx = obj_pos[0] - conveyor_pos[0]
            dy = obj_pos[1] - conveyor_pos[1]
            r = np.sqrt(dx * dx + dy * dy)
            if r < 1e-4:
                continue

            omega = self.curved_speed / r
            dof = _get_dof_adr(self.sim, obj_body_id)
            # Tangential velocity for circular motion
            self.sim.data.qvel[dof:dof + 3] = [omega * dy, -omega * dx, 0.0]
            # Yaw rotation to follow the arc
            self.sim.data.qvel[dof + 3:dof + 6] = [0.0, 0.0, -omega]
