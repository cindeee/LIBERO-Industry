import numpy as np

def _get_dof_adr(sim, obj_body_id):
    return sim.model.jnt_dofadr[sim.model.body_jntadr[obj_body_id]]

class GhostConveyorMixin:
    """Invisible conveyor that dynamically calculates, resizes, and tracks virtual inlays."""

    def setup_ghost_conveyor(self, conveyor_name, groove_width, groove_length, velocity=0.02, mode="steady", precision_threshold=0.03):
        self.ghost_conveyor_name = conveyor_name  
        self.ghost_velocity = velocity
        self.groove_width = groove_width
        self.groove_length = groove_length
        self.conveyor_mode = mode
        self.precision_threshold = precision_threshold
        
        self.virtual_inlays = {}
        self.completed_objects = set()
        self._inlays_initialized = False

    def _get_ghost_region_bounds(self):
        """Get the spatial bounding box of the ghost region site."""
        # Prepend the prefix to the search target
        site_target = f"{self.ghost_conveyor_name}_ghost_active_region"
        
        site_id = self.sim.model.site_name2id(site_target)
        site_pos = self.sim.data.site_xpos[site_id]
        site_size = self.sim.model.site_size[site_id] 
        
        return {
            'x_center': site_pos[0],
            'x_min': site_pos[0] - site_size[0], 'x_max': site_pos[0] + site_size[0],
            'y_min': site_pos[1] - site_size[1], 'y_max': site_pos[1] + site_size[1],
            'z_min': site_pos[2] - site_size[2], 'z_max': site_pos[2] + site_size[2],
            'z_surface': site_pos[2] 
        }

    def _initialize_dynamic_inlays(self):
        bounds = self._get_ghost_region_bounds()
        conveyor_length = bounds['y_max'] - bounds['y_min']
        num_inlays_needed = int(conveyor_length / self.groove_width)
        
        self.virtual_inlays = {}
        
        for i in range(num_inlays_needed):
            # Prepend the prefix to the pool target
            pool_target = f"{self.ghost_conveyor_name}_groove_pool_{i}"
            
            if pool_target not in self.sim.model.site_names:
                print(f"WARNING: Reached limit of XML site pool at {pool_target}.")
                break
                
            site_id = self.sim.model.site_name2id(pool_target)
            
            self.sim.model.site_size[site_id] = [self.groove_length / 2.0, self.groove_width / 2.0, 0.005]
            self.sim.model.site_rgba[site_id][3] = 0.8
            
            start_y = bounds['y_min'] + (i * self.groove_width) + (self.groove_width / 2.0)
            self.sim.model.site_pos[site_id] = [bounds['x_center'], start_y, bounds['z_surface']]
            
            self.virtual_inlays[i] = {
                'y_pos': start_y, 
                'held_object': None,
                'site_id': site_id
            }
            
        self._inlays_initialized = True
        print(f"Ghost Conveyor Initialized: Molded {num_inlays_needed} grooves to size {self.groove_width}x{self.groove_length}")

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        if hasattr(self, 'ghost_velocity'):
            if not self._inlays_initialized:
                self._initialize_dynamic_inlays()
                
            self._update_virtual_inlays()
            self._apply_ghost_physics()

    def _update_virtual_inlays(self):
        dt = self.sim.model.opt.timestep
        bounds = self._get_ghost_region_bounds()
        
        # Handle 'steady' vs 'pulse' mode
        if self.conveyor_mode == "pulse":
            sim_time = self.sim.data.time
            is_moving = 1.0 if (sim_time % 2.0) < 1.0 else 0.0
            current_velocity = self.ghost_velocity * is_moving
        else:
            current_velocity = self.ghost_velocity

        for idx, inlay in self.virtual_inlays.items():
            inlay['y_pos'] += current_velocity * dt

            # Update visual site position
            if inlay['site_id'] is not None:
                self.sim.model.site_pos[inlay['site_id']][1] = inlay['y_pos']

            # Wrap around: If an inlay falls off the end, loop it back to the beginning
            if inlay['y_pos'] > bounds['y_max']:
                # If it had an object, score it and clear it
                if inlay['held_object'] is not None:
                    self.completed_objects.add(inlay['held_object'])
                    
                    # Teleport object out of the way (under the table)
                    obj_body_id = self.obj_body_id[inlay['held_object']]
                    dof = _get_dof_adr(self.sim, obj_body_id)
                    self.sim.data.qpos[dof + 2] = -5.0 
                    
                    inlay['held_object'] = None
                
                # Loop back to the start of the conveyor
                inlay['y_pos'] = bounds['y_min']

    def _apply_ghost_physics(self):
        bounds = self._get_ghost_region_bounds()

        for obj_name in self.objects_dict:
            if obj_name in self.completed_objects:
                continue

            obj_body_id = self.obj_body_id[obj_name]
            obj_pos = self.sim.data.body_xpos[obj_body_id]

            # Check if object is inside the spatial bounding box
            in_x = bounds['x_min'] <= obj_pos[0] <= bounds['x_max']
            in_y = bounds['y_min'] <= obj_pos[1] <= bounds['y_max']
            in_z = bounds['z_min'] <= obj_pos[2] <= bounds['z_max']

            if in_x and in_y and in_z:
                dof = _get_dof_adr(self.sim, obj_body_id)
                
                for idx, inlay in self.virtual_inlays.items():
                    # If this object is already locked to this inlay, keep moving it
                    if inlay['held_object'] == obj_name:
                        # Determine current velocity based on mode
                        vel = self.ghost_velocity if self.conveyor_mode == "steady" else (self.ghost_velocity if (self.sim.data.time % 2.0) < 1.0 else 0.0)
                        self.sim.data.qvel[dof:dof+3] = [0, vel, 0]
                        break
                    
                    # If empty, check if placed accurately enough
                    elif inlay['held_object'] is None:
                        # Check distance to the center of the virtual groove
                        y_distance = abs(obj_pos[1] - inlay['y_pos'])
                        x_distance = abs(obj_pos[0] - bounds['x_center'])
                        
                        # If within threshold...
                        if y_distance < self.precision_threshold and x_distance < (self.groove_length / 2.0):
                            # Lock it in!
                            inlay['held_object'] = obj_name
                            
                            # Snap object perfectly into the visual groove
                            self.sim.data.qpos[dof] = bounds['x_center'] 
                            self.sim.data.qpos[dof+1] = inlay['y_pos']
                            
                            vel = self.ghost_velocity if self.conveyor_mode == "steady" else (self.ghost_velocity if (self.sim.data.time % 2.0) < 1.0 else 0.0)
                            self.sim.data.qvel[dof:dof+3] = [0, vel, 0]
                            break