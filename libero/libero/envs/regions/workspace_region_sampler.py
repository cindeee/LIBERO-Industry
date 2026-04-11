import numpy as np

from .base_region_sampler import MultiRegionRandomSampler, InSiteRegionRandomSampler
from robosuite.utils.transform_utils import quat_multiply
from copy import copy
from robosuite.utils.errors import RandomizationError
from robosuite.utils.transform_utils import quat_multiply


class TableRegionSampler(MultiRegionRandomSampler):
    def __init__(
        self,
        object_name,
        mujoco_objects=None,
        x_ranges=None,
        y_ranges=None,
        rotation=(np.pi / 2, np.pi / 2),
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.01,
    ):
        name = f"table-middle-{object_name}"
        super().__init__(
            object_name,
            mujoco_objects,
            x_ranges,
            y_ranges,
            rotation,
            rotation_axis,
            ensure_object_boundary_in_range,
            ensure_valid_placement,
            reference_pos,
            z_offset,
        )

    def _sample_quat(self):
        """
        Samples the orientation for a given object
        Add multiple rotation options
        Returns:
            np.array: sampled (r,p,y) euler angle orientation
        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, tuple) or isinstance(self.rotation, list):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        # multiple rotations
        elif isinstance(self.rotation, dict):
            quat = np.array(
                [0.0, 0.0, 0.0, 1.0]
            )  # \theta=0, in robosuite, quat = (x, y, z), w
            for i in range(len(self.rotation.keys())):
                rotation_axis = list(self.rotation.keys())[i]
                rot_angle = np.random.uniform(
                    high=max(self.rotation[rotation_axis]),
                    low=min(self.rotation[rotation_axis]),
                )

                if rotation_axis == "x":
                    current_quat = np.array(
                        [np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "y":
                    current_quat = np.array(
                        [0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "z":
                    current_quat = np.array(
                        [0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)]
                    )

                quat = quat_multiply(current_quat, quat)

            return quat
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            return np.array([np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "y":
            return np.array([0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "z":
            return np.array([0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )


class Libero100TableRegionSampler(MultiRegionRandomSampler):
    def __init__(
        self,
        object_name,
        mujoco_objects=None,
        x_ranges=None,
        y_ranges=None,
        rotation=(np.pi / 2, np.pi / 2),
        rotation_axis="z",
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.01,
    ):
        name = f"table-middle-{object_name}"
        super().__init__(
            object_name,
            mujoco_objects,
            x_ranges,
            y_ranges,
            rotation,
            rotation_axis,
            ensure_object_boundary_in_range,
            ensure_valid_placement,
            reference_pos,
            z_offset,
        )

    def _sample_quat(self):
        """
        Samples the orientation for a given object
        Add multiple rotation options
        Returns:
            np.array: sampled (r,p,y) euler angle orientation
        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, tuple) or isinstance(self.rotation, list):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        # multiple rotations
        elif isinstance(self.rotation, dict):
            quat = np.array(
                [0.0, 0.0, 0.0, 1.0]
            )  # \theta=0, in robosuite, quat = (x, y, z), w
            for i in range(len(self.rotation.keys())):
                rotation_axis = list(self.rotation.keys())[i]
                rot_angle = np.random.uniform(
                    high=max(self.rotation[rotation_axis]),
                    low=min(self.rotation[rotation_axis]),
                )

                if rotation_axis == "x":
                    current_quat = np.array(
                        [np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "y":
                    current_quat = np.array(
                        [0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "z":
                    current_quat = np.array(
                        [0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)]
                    )

                quat = quat_multiply(current_quat, quat)

            return quat
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            return np.array([np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "y":
            return np.array([0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "z":
            return np.array([0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )


class ObjectBasedSampler(MultiRegionRandomSampler):
    def __init__(
        self,
        object_name,
        mujoco_objects=None,
        x_ranges=None,
        y_ranges=None,
        rotation=(np.pi / 2, np.pi / 2),
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.01,
    ):
        name = f"table-middle-{object_name}"
        super().__init__(
            object_name,
            mujoco_objects,
            x_ranges,
            y_ranges,
            rotation,
            rotation_axis,
            ensure_object_boundary_in_range,
            ensure_valid_placement,
            reference_pos,
            z_offset,
        )

    def _sample_quat(self):
        """
        Samples the orientation for a given object
        Add multiple rotation options
        Returns:
            np.array: sampled (r,p,y) euler angle orientation
        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, tuple) or isinstance(self.rotation, list):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        # multiple rotations
        elif isinstance(self.rotation, dict):
            quat = np.array(
                [0.0, 0.0, 0.0, 1.0]
            )  # \theta=0, in robosuite, quat = (x, y, z), w
            for i in range(len(self.rotation.keys())):
                rotation_axis = list(self.rotation.keys())[i]
                rot_angle = np.random.uniform(
                    high=max(self.rotation[rotation_axis]),
                    low=min(self.rotation[rotation_axis]),
                )

                if rotation_axis == "x":
                    current_quat = np.array(
                        [np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "y":
                    current_quat = np.array(
                        [0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "z":
                    current_quat = np.array(
                        [0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)]
                    )

                quat = quat_multiply(current_quat, quat)

            return quat
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            return np.array([np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "y":
            return np.array([0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "z":
            return np.array([0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )


class LiberoIndustrySampler(InSiteRegionRandomSampler):
    def __init__(
        self,
        object_name,
        mujoco_objects=None,
        x_ranges=None,
        y_ranges=None,
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.01,
    ):
        name = f"conveyor-{object_name}"
        super().__init__(
            name,
            mujoco_objects,
            x_ranges,
            y_ranges,
            rotation,
            rotation_axis,
            ensure_object_boundary_in_range,
            ensure_valid_placement,
            reference_pos,
            z_offset,
        )

    def sample(self, fixtures=None, reference=None, site_name="", on_top=True, sim=None):
        """
        Overrides the base sample method to make 'sim' an optional argument.
        """
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, ref_quat, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(
                base_offset
            )

        for obj in self.mujoco_objects:
            assert (
                obj.name not in placed_objects
            ), "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            
            # --- THE FIX: Safe sim check ---
            if sim is not None and site_name != "":
                site_offset = sim.data.get_site_xpos(site_name)
                if 'ref_quat' in locals():
                    site_x, site_y, site_z = T.quat2mat(
                        T.convert_quat(ref_quat, to="xyzw")
                    ) @ site_offset
                else:
                    site_x, site_y, site_z = site_offset
            else:
                # Fallback for static table placements where sim is missing
                site_x, site_y, site_z = 0.0, 0.0, 0.0
            # -------------------------------

            for i in range(5000):  # 5000 retries
                self.idx = np.random.randint(self.num_ranges)
                object_x = self._sample_x(0) + base_offset[0] + site_x
                object_y = self._sample_y(0) + base_offset[1] + site_y
                object_z = self.z_offset + base_offset[2] + site_z
                if on_top:
                    object_z -= bottom_offset[-1]

                # objects cannot overlap
                location_valid = True
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if (
                            np.linalg.norm((object_x - x, object_y - y))
                            <= other_obj.horizontal_radius + horizontal_radius
                        ) and (
                            object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]
                        ):
                            location_valid = False
                            break

                if location_valid:
                    quat = self._sample_quat()

                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)

                    # ========================================================
                    # CRITICAL FIX: THE LIBERO QUATERNION PATCH
                    # Shift [x, y, z, w] to [w, x, y, z] manually so the 
                    # BDDL loader doesn't accidentally flip the object 180 degrees!
                    # ========================================================
                    if len(obj.joints) > 0:
                        quat = np.array([quat[3], quat[0], quat[1], quat[2]])

                    # location is valid, put the object down
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                import pdb
                pdb.set_trace()
                raise RandomizationError("Cannot place all objects ):")

        return placed_objects