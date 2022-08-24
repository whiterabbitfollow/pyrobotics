from pyrb.mp.planners.rrt import LocalPlannerSpaceTime, RRTPlanner
from pyrb.mp.planners.rrt_connect import RRTConnectPlanner
from pyrb.mp.utils.trees.tree import Tree
from pyrb.mp.utils.trees.tree_rewire import TreeRewireSpaceTime


def compile_all_planners(world, state_space_start, state_space_goal):
    planner_kwargs = [
        ("rrt", make_rrt, {"state_space": state_space_start}),
        ("rrt_connect", make_rrt_connect, {"state_space_start": state_space_start, "state_space_goal": state_space_goal})
    ]
    planners = {}
    for planner_name, make_func, make_func_kwargs in planner_kwargs:
        planner = make_func(world, **make_func_kwargs)
        planners[planner_name] = planner
    return planners


def make_rrt(world, state_space):
    local_planner = LocalPlannerSpaceTime(
        min_path_distance=0.2,
        min_coll_step_size=0.05,
        max_distance=(1.0, 20),
        max_actuation=world.robot.max_actuation
    )
    planner = RRTPlanner(
        space=state_space,
        tree=Tree(max_nr_vertices=int(1e4), vertex_dim=state_space.dim),
        local_planner=local_planner
    )
    return planner


def make_rrt_connect(world, state_space_start, state_space_goal):
    local_planner = LocalPlannerSpaceTime(
        min_path_distance=0.2,
        min_coll_step_size=0.05,
        max_distance=(1.0, 5),
        max_actuation=world.robot.max_actuation
    )

    tree_start = Tree(
        space=state_space_start,
        max_nr_vertices=int(1e4),
        vertex_dim=state_space_start.dim
    )

    tree_goal = Tree(
        space=state_space_goal,
        max_nr_vertices=int(1e4),
        vertex_dim=state_space_goal.dim
    )

    planner = RRTConnectPlanner(
        local_planner=local_planner,
        tree_start=tree_start,
        tree_goal=tree_goal
    )

    return planner


def make_rrt_star(world, state_space):
    local_planner = LocalPlannerSpaceTime(
        world.robot.max_actuation,
        nr_time_steps=5,
        min_path_distance=.3,
        min_coll_step_size=0.05,
        max_distance=(1.0, 10)
    )

    planner = RRTPlanner(
        space=state_space,
        tree=TreeRewireSpaceTime(
            local_planner=local_planner,
            space=state_space,
            nearest_radius=1.0,
            nearest_time_window=10,
            max_nr_vertices=int(1e4)
        ),
        local_planner=local_planner
    )
    return planner


def make_rrt_star_connect(world, state_space_start, state_space_goal):
    local_planner = LocalPlannerSpaceTime(
        min_path_distance=0.2,
        min_coll_step_size=0.05,
        max_distance=(1.0, 5),
        max_actuation=world.robot.max_actuation
    )

    tree_start = TreeRewireSpaceTime(
        local_planner=local_planner,
        max_nr_vertices=int(1e3),
        nearest_radius=.2,
        nearest_time_window=10,
        space=state_space_start
    )

    tree_goal = TreeRewireSpaceTime(
        local_planner=local_planner,
        max_nr_vertices=int(1e3),
        nearest_radius=.2,
        nearest_time_window=10,
        space=state_space_goal
    )

    planner = RRTConnectPlanner(
        local_planner=local_planner,
        tree_start=tree_start,
        tree_goal=tree_goal
    )

    return planner