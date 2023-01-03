import numpy as np


class LocalPlanner:

    def __init__(self, world, step_resolution=1e-3, exploring_distance_max=20.):
        self.world = world
        self.step_resolution = step_resolution
        self.exploring_distance_max = exploring_distance_max

    def plan(self, x_nearest, x_free, distance=None):
        # TODO: could use nearest distance function for this
        distance_st = np.linalg.norm(x_nearest - x_free)
        x_coll_free = x_nearest
        beta = 0
        distance_st_explore = self.step_resolution
        exploring_distance_max = distance or self.exploring_distance_max
        distance_st_max = min(exploring_distance_max, distance_st)
        status = True
        beta_max = distance_st_max/distance_st
        while beta < beta_max and status:
            beta = distance_st_explore / distance_st
            x_check = (1-beta) * x_nearest + beta * x_free
            status = self.world.is_state_collision_free(x_check)
            x_coll_free = x_check if status else x_coll_free
            distance_st_explore += self.step_resolution
        return status, x_coll_free


class LocalPlannerVelocityConstrained(LocalPlanner):

    def __init__(self, *args, speed_max, **kwargs):
        super().__init__(*args, **kwargs)
        self.speed_max = speed_max

    def project_state_within_speed_cone(self, x_src, x_dst):
        t_0 = x_src[-1]
        dq = x_dst[:-1] - x_src[:-1]
        distance = np.linalg.norm(dq)
        dt_needed = x_dst[-1] - x_src[-1]
        speed_needed = distance / dt_needed
        speed = min(self.speed_max, speed_needed)
        dt = distance / speed
        x = np.append(dq, t_0 + dt)
        return x

    def plan(self, x_nearest, x_free, distance=None):
        x_free_projected = self.project_state_within_speed_cone(x_nearest, x_free)
        return super().plan(x_nearest, x_free_projected, distance)
