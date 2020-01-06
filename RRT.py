import numpy as np
from utils import add_circles, OBSTACLE_REGION, GOAL_REGION, BasePathPlanner, viz_path

class RRT(BasePathPlanner):
    
    def __init__(self, max_tree_size= 500, max_local_planner_distance=5):
        self.MAX_T_SIZE = max_tree_size
        self.LOCAL_PLANNER_MAX_DISTANCE = max_local_planner_distance
    
    def _get_x_sample(self, X_goal, Y_goal, X, Y):
        x_sample = np.random.normal((X_goal, Y_goal), (X, Y)).astype(int).reshape(2,1)
        while not np.all(x_sample>0) or x_sample[0,0]>X or x_sample[1,0]>Y or np.all(x_sample<0):
            x_sample = np.random.normal((X_goal, Y_goal), (X*0.5, Y*0.5)).astype(int).reshape(2,1)    
        return x_sample
    
    def _plan_locally(self, x_nearest, x_sample, T, grid):
        d = self.LOCAL_PLANNER_MAX_DISTANCE
        alphas = np.arange(1, 5) * 0.05
        x_path = x_nearest * (1-alphas) + x_sample * alphas
        x_path = x_path.astype(int)
        i_mask = (np.linalg.norm(x_path-x_nearest, axis=0) < d)
        i_mask = i_mask & (grid[x_path[0,:],x_path[1,:]] != OBSTACLE_REGION)
        i_ok, = np.where(i_mask)
        if len(i_ok) == 0:
            return None
        return x_path[:,i_ok[-1]].reshape(-1,1)    
        
    def plan_path(self, grid, x_start, X, Y, X_goal, Y_goal):
        T = np.c_[x_start]
        edges = [0]
        while T.shape[1] < self.MAX_T_SIZE:
            x_sample = self._get_x_sample(X_goal, Y_goal, X, Y)
            i_nearest = np.argmin(np.linalg.norm(x_sample - T, axis=0))
            x_nearest = T[:,i_nearest].reshape(-1,1)
            x_new = self._plan_locally(x_nearest, x_sample, T, grid)
            if x_new is None:
                continue            
            edges.append(i_nearest)
            T = np.c_[T, x_new]    
            if grid[x_new[0], x_new[1]] == GOAL_REGION:
                return self._generate_path_to_goal(edges, T)
        return []
    
    def _generate_path_to_goal(self, edges, T, i_parent_start=-1):
        i_parent = edges[i_parent_start]
        if i_parent_start == -1:
            i_current = len(edges) - 1
        else:
            i_current = i_parent_start
        path = np.c_[T[:,i_current]]
        indx = set()
        while i_parent != 0 and i_parent not in indx:
            indx.add(i_parent)
            i_current = i_parent
            i_parent = edges[i_parent]
            path = np.c_[T[:,i_current], path]
        return path

if __name__ == "__main__":
    np.random.seed(1)
    X, Y = 150, 150
    grid = np.zeros((X,Y))
    goal_region  = (X-20, Y-20, 10)
    X_goal, Y_goal, goal_R  = goal_region
    X_start, Y_start = 0, 0
    obstacle_dims = [(30,30,20), (60,80,30)]    
    add_circles(grid, obstacle_dims, X, Y, OBSTACLE_REGION)
    add_circles(grid, [goal_region], X, Y, GOAL_REGION)    
    x_start = np.array([X_start,Y_start]).reshape(2,1)    
    agent = RRT()
    path = agent.plan_path(grid, x_start, X, Y, X_goal, Y_goal)    
    viz_path(grid, path)
    
    Vx, Vy = 1, 1
    V = np.sqrt(2)    
    path_d = np.diff(path, axis=1)    
    dTs = np.linalg.norm(path_d, axis=0)/(V*0.5)
    Ts = np.r_[0, np.cumsum(dTs)]
    Vs = np.c_[path_d[0::]/dTs, np.zeros((2,1))]
    
    # In[]
    import matplotlib.pyplot as plt
    plt.plot(path[0,:],path[1,:])
    # In[]
    from trajectory import trajectory_generate_via_points_cubic_spline
    ts, vs = trajectory_generate_via_points_cubic_spline(path, Ts, Vs)
    # In[]
    plt.plot(vs[0,:],vs[1,:])
    plt.plot(path[0,:],path[1,:])
    # In[]
    plt.plot(Ts, path[0,:],marker=".")
    plt.plot(ts, vs[0,:])
    
    # In[]
    plt.plot(ts[0:-1], np.diff(vs[0,:])/np.diff(ts))
    plt.plot(ts[0:-1], np.diff(vs[1,:])/np.diff(ts))
        
        
        
        
        
        
        
        
        