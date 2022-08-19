import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from pyrb.mp.utils.tree import TreeRewire


class DummyLocalPlanner:

    def plan(self, state_src, state_dst, state_global_goal=None, full_plan=False):
        return state_dst.reshape(1, -1)


local_planner = DummyLocalPlanner()
tree = TreeRewire(local_planner=local_planner, max_nr_vertices=100, nearest_radius=.2, vertex_dim=2)

tree.add_vertex(np.array([0, 0]))                       # 0
tree.append_vertex(np.array([0.2, 0.8]), i_parent=0)    # 1
tree.append_vertex(np.array([0.5, 0.6]), i_parent=1)    # 2
tree.append_vertex(np.array([0.5, 0.5]), i_parent=0)    # 3


def render_tree(ax, tree):
    color = "black"
    verts = tree.get_vertices()
    ax.scatter(verts[:, 0], verts[:, 1], color=color)
    for i_parent, indxs_children in tree.edges_parent_to_children.items():
        for i_child in indxs_children:
            q = np.stack([verts[i_parent], verts[i_child]], axis=0)
            ax.plot(q[:, 0], q[:, 1], color=color)


def render_nearest_and_new_vert(ax, new_vert, tree, nearest_vertices_indices):
    nearest_vertices = vertices[nearest_vertices_indices, :]
    nearest_vertices_costs = tree.cost_to_verts[nearest_vertices_indices]
    ax.scatter(new_vert[0], new_vert[1], c=new_vert_color)
    ax.add_patch(Circle(tuple(new_vert), tree.nearest_radius, color=new_vert_color, alpha=0.1))
    for nearest_vertex, cost in zip(nearest_vertices, nearest_vertices_costs):
        ax.scatter(nearest_vertex[0], nearest_vertex[1], label=f"cost {cost:.2f}")


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

vertices = tree.get_vertices()
new_vert = np.array([0.6, 0.6])
new_vert_color = "cyan"
i_new = tree.vert_cnt

indices = tree.get_collision_free_nearest_indices(new_vert)


render_tree(ax1, tree)
render_nearest_and_new_vert(ax1, new_vert, tree, indices)
ax1.set_aspect("equal")
ax1.set_title("Before wiring new through nearest")
ax1.legend(loc="best")

tree.wire_new_through_nearest(i_new, new_vert, indices)

render_tree(ax2, tree)
render_nearest_and_new_vert(ax2, new_vert, tree, indices)
ax2.set_aspect("equal")
ax2.set_title("After wiring new through nearest")
ax2.legend(loc="best")


tree.rewire_nearest_through_new(i_new, new_vert, indices)
render_tree(ax3, tree)
render_nearest_and_new_vert(ax3, new_vert, tree, indices)
ax3.set_aspect("equal")
ax3.set_title("After rewiring through new")
ax3.legend(loc="best")

plt.show()
