import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from examples.utils import render_tree
from pyrb.mp.utils.spaces import RealVectorTimeSpace
from pyrb.mp.utils.trees.tree_rewire import TreeRewireSpaceTime
from pyrb.mp.utils.constants import LocalPlannerStatus


class DummyLocalPlanner:

    def __init__(self):
        self.max_actuation = 2

    def plan(self, space, state_src, state_dst, max_distance=None):
        return LocalPlannerStatus.REACHED, state_dst.reshape(1, -1)


class DummyRobot:

    def __init__(self):
        self.max_actuation = 2


class DummyWorld:

    def __init__(self):
        self.robot = DummyRobot()


local_planner = DummyLocalPlanner()
world = DummyWorld()
space = RealVectorTimeSpace(world=world, dim=1, limits=[-1, 1], max_time=10, gamma=0.1)
tree = TreeRewireSpaceTime(
    local_planner=local_planner,
    max_nr_vertices=100,
    nearest_radius=.2,
    nearest_time_window=5,
    space=space
)


TIME_HORIZON = 6
tree.set_root_vertex(np.array([0, 0]))                       # 0

state = np.array([0, 2])


vs = [
    ([1.5, 1], 0),  # 1
    ([4.5, 3], 1),  # 2
    ([3, 4], 2),    # 3
    ([1.5, 5], 3),  # 4
    ([0.0, 6], 4),  # 5
    ([0.0, 10], 5), # 6
    ([0.4, 12], 6), # 7
    ([-0.5, 1], 0)  # 8
]

for coords, i_parent in vs:
    vertex = np.array(coords)
    edge_cost = space.transition_cost(vertex, tree.vertices[i_parent])
    tree.append_vertex_without_rewiring(
        vertex,
        i_parent=i_parent,
        edge_cost=edge_cost
    )

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
render_tree(ax1, tree.get_vertices(), tree.get_edges())



ax1.scatter(state[0], state[1])

verts = tree.get_vertices()
for i_vert, vert in enumerate(verts):
    cost = tree.cost_to_verts[i_vert]
    ax1.annotate(f"{cost:0.2f}", vert)


polygon_values = np.array([
    tuple(state),
    (state[0] + world.robot.max_actuation * TIME_HORIZON, state[1] - TIME_HORIZON),
    (state[0] - world.robot.max_actuation * TIME_HORIZON, state[1] - TIME_HORIZON),
])
ax2.add_patch(
    Polygon(polygon_values, alpha=0.1)
)

# polygon_values = np.array([
#     (state[0] + world.robot.max_actuation * TIME_HORIZON, state[1]),
#     (state[0] + world.robot.max_actuation * TIME_HORIZON, state[1] - TIME_HORIZON),
#     (state[0] - world.robot.max_actuation * TIME_HORIZON, state[1] - TIME_HORIZON),
#     (state[0] - world.robot.max_actuation * TIME_HORIZON, state[1])
# ])
#
# ax2.add_patch(
#     Polygon(polygon_values, alpha=0.1)
# )

ax2.scatter(state[0], state[1])

verts = tree.get_vertices()
edges = tree.get_edges()
render_tree(ax2, tree.get_vertices(), tree.get_edges())

indxs_past = tree.get_nearest_past_states_indices(state)
verts_past = verts[indxs_past]
ax2.scatter(verts_past[:, 0], verts_past[:, 1], c="red")

for i_vert, vert in zip(indxs_past, verts_past):
    cost = tree.cost_to_verts[i_vert]
    ax2.annotate(f"{cost:0.2f}", vert)

indxs_past_coll_free = tree.get_collision_free_past_nearest_indices(state, indxs_past)
verts_past = verts[indxs_past_coll_free]

ax2.scatter(verts_past[:, 0], verts_past[:, 1], c="orange")
ax2.set_title("Rewiring past states")

tree.wire_new_through_nearest(state, indxs_past_coll_free)
render_tree(ax3, tree.get_vertices(), tree.get_edges())
ax3.add_patch(
    Polygon(polygon_values, alpha=0.1)
)
for i_vert, vert in enumerate(verts):
    cost = tree.cost_to_verts[i_vert]
    ax3.annotate(f"{cost:0.2f}", vert)

verts, edges = tree.get_vertices(), tree.get_edges()

indxs_future = tree.get_nearest_future_states_indices(state)
verts_future = verts[indxs_future]

render_tree(ax4, verts, edges)
ax4.scatter(verts_future[:, 0], verts_future[:, 1], c="orange")
# ax3.set_title("Rewiring past states")

indxs_future_all = (verts[:, -1] > state[-1]).nonzero()[0]
verts_future_all = verts[indxs_future_all]
for i_vert, vert in zip(indxs_future_all, verts_future_all):
    cost = tree.cost_to_verts[i_vert]
    ax4.annotate(f"{cost:0.2f}", vert)

polygon_values = np.array([
    tuple(state),
    (state[0] + world.robot.max_actuation * TIME_HORIZON, state[1] + TIME_HORIZON),
    (state[0] - world.robot.max_actuation * TIME_HORIZON, state[1] + TIME_HORIZON),
])

ax4.add_patch(
    Polygon(polygon_values, alpha=0.1)
)

i_new = tree.vert_cnt - 1
tree.rewire_nearest_through_new(i_new, state, indxs_future)

render_tree(ax5, verts, edges)

ax5.scatter(verts_future[:, 0], verts_future[:, 1], c="orange")

indxs_future_all = (verts[:, -1] > state[-1]).nonzero()[0]
verts_future_all = verts[indxs_future_all]
for i_vert, vert in zip(indxs_future_all, verts_future_all):
    cost = tree.cost_to_verts[i_vert]
    ax5.annotate(f"{cost:0.2f}", vert)


# ax3.set_title("Rewiring past states")

polygon_values = np.array([
    tuple(state),
    (state[0] + world.robot.max_actuation * TIME_HORIZON, state[1] + TIME_HORIZON),
    (state[0] - world.robot.max_actuation * TIME_HORIZON, state[1] + TIME_HORIZON),
])

ax5.add_patch(
    Polygon(polygon_values, alpha=0.1)
)

polygon_values = np.array([
    tuple(state),
    (state[0] + world.robot.max_actuation * TIME_HORIZON, state[1] + TIME_HORIZON),
    (state[0] - world.robot.max_actuation * TIME_HORIZON, state[1] + TIME_HORIZON),
])
ax3.scatter(state[0], state[1])
ax3.add_patch(
    Polygon(polygon_values, alpha=0.1)
)
ax3.set_title("Rewiring future states")

plt.show()