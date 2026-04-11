from typing import List
import numpy as np 

class Expression:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class UnaryAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, arg1):
        raise NotImplementedError


class BinaryAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, arg1, arg2):
        raise NotImplementedError


class MultiarayAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, *args):
        raise NotImplementedError


class TruePredicateFn(MultiarayAtomic):
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return True


class FalsePredicateFn(MultiarayAtomic):
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return False


class InContactPredicateFn(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg1.check_contact(arg2)


class In(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_contact(arg1) and arg2.check_contain(arg1)


class On(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_ontop(arg1)

        # if arg2.object_state_type == "site":
        #     return arg2.check_ontop(arg1)
        # else:
        #     obj_1_pos = arg1.get_geom_state()["pos"]
        #     obj_2_pos = arg2.get_geom_state()["pos"]
        #     # arg1.on_top_of(arg2) ?
        #     # TODO (Yfeng): Add checking of center of mass are in the same regions
        #     if obj_1_pos[2] >= obj_2_pos[2] and arg2.check_contact(arg1):
        #         return True
        #     else:
        #         return False


class Up(BinaryAtomic):
    def __call__(self, arg1):
        return arg1.get_geom_state()["pos"][2] >= 1.0


class Stack(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return (
            arg1.check_contact(arg2)
            and arg2.check_contain(arg1)
            and arg1.get_geom_state()["pos"][2] > arg2.get_geom_state()["pos"][2]
        )


class PrintJointState(UnaryAtomic):
    """This is a debug predicate to allow you print the joint values of the object you care"""

    def __call__(self, arg):
        print(arg.get_joint_state())
        return True


class Open(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_open()


class Close(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_close()


class TurnOn(UnaryAtomic):
    def __call__(self, arg):
        return arg.turn_on()


class TurnOff(UnaryAtomic):
    def __call__(self, arg):
        return arg.turn_off()

class AtSpeed(MultiarayAtomic):
    """
    Records the operational state of a conveyor.
    Usage in BDDL: (AtSpeed conveyor_belt_1 0.05 steady)
    """
    def __call__(self, *args):
        return True

class LoadSlot(MultiarayAtomic):
    """
    Records the operational state of a conveyor.
    Usage in BDDL: (LoadSlot conveyor_belt_1 0.05 steady)
    """
    def __call__(self, *args):
        return True

class ExactIn(BinaryAtomic):
    """Stricter "in": inside region AND bottom is resting on the region's surface.
    Upgraded for Dynamic Tasks: Evaluates Z-height relative to the target region, 
    allowing for placement on moving conveyors or static tables.
    """
    def __call__(self, arg1, arg2):
        # First: X/Y containment check
        try:
            contain_ok = arg2.check_contain(arg1)
        except Exception:
            contain_ok = False

        bottom_ok = False
        reg_type = getattr(arg2, 'object_state_type', '?')
        
        if reg_type == 'site':
            try:
                env = arg2.env
                site_name = arg2.object_name
                
                # 1. Get the target region's absolute Z height
                site_pos = env.sim.data.get_site_xpos(site_name)
                target_z = float(site_pos[2])

                # 2. Get the object's absolute bottom Z height
                bottom_site_name = f"{arg1.object_name}_bottom_site"
                try:
                    obj_bottom_world = env.sim.data.get_site_xpos(bottom_site_name)
                except Exception:
                    # Fallback to body COM if no bottom site exists
                    obj_bottom_world = env.sim.data.body_xpos[env.obj_body_id[arg1.object_name]]

                obj_bottom_z = float(obj_bottom_world[2])

                # 3. Compare them (Object bottom should be resting ON the region)
                delta_z_world = obj_bottom_z - target_z
                z_eps = 0.04 # 4cm tolerance

                bottom_ok = abs(delta_z_world) <= z_eps
            except Exception as e:
                print(f"[ExactIn WARNING] geometric calc failed: {e}")
                bottom_ok = arg2.check_ontop(arg1)
        else:
            # Non-site fallback
            try:
                bottom_ok = arg2.check_ontop(arg1)
            except Exception:
                bottom_ok = False

        return contain_ok and bottom_ok

class MovingWith(BinaryAtomic):
    """Checks if arg1 is dynamically stable relative to arg2
    for ensuring objects aren't sliding, bouncing, or rolling off a conveyor.
    """
    def __call__(self, arg1, arg2):
        try:
            env = arg1.env
            
            # Get linear velocities
            v1 = env.sim.data.body_xvelp[env.obj_body_id[arg1.object_name]]
            
            if getattr(arg2, 'object_state_type', None) == 'site':
                # Sites don't have direct velocity in MuJoCo, use parent body
                parent_body_name = env.object_sites_dict[arg2.object_name].parent_name
                v2 = env.sim.data.body_xvelp[env.sim.model.body_name2id(parent_body_name)]
            else:
                v2 = env.sim.data.body_xvelp[env.obj_body_id[arg2.object_name]]

            # Calculate relative velocity magnitude
            relative_vel = np.linalg.norm(v1 - v2)
            
            # If relative velocity is less than 0.01 m/s, they are moving together
            return relative_vel < 0.01
        except Exception:
            return False