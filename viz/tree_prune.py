import matplotlib.pyplot as plt
import numpy as np

from examples.utils import render_tree
from pyrb.mp.utils.tree import Tree

tree = Tree(max_nr_vertices=10, vertex_dim=2)

tree.add_vertex(np.array([0, 0]))                       # 0
tree.append_vertex(np.array([0.0, 0.1]), i_parent=0)    # 1
tree.append_vertex(np.array([0.0, 0.2]), i_parent=1)    # 2
tree.append_vertex(np.array([0.1, 0.3]), i_parent=2)    # 3
tree.append_vertex(np.array([0.1, 0.4]), i_parent=3)    # 4
tree.append_vertex(np.array([-0.1, 0.3]), i_parent=2)    # 5
tree.append_vertex(np.array([-0.1, 0.4]), i_parent=5)    # 6
tree.append_vertex(np.array([0.1, 0.2]), i_parent=1)    # 6
tree.append_vertex(np.array([-0.1, 0.2]), i_parent=1)    # 6

fig, (ax1, ax2) = plt.subplots(1, 2)

vertices, edges = tree.get_vertices(), tree.get_edges()
render_tree(ax1, vertices, edges)
for i_vert, vert in enumerate(vertices):
    i_parent = edges[i_vert]
    ax1.annotate(f"{i_vert}: {i_parent}", vert)
ax1.set_aspect("equal")
ax1.set_title("Before pruning")
# ax1.legend(loc="best")

#
# print(tree.is_vertex_leaf(2))
# print(tree.is_vertex_leaf(4))
#
#

tree.prune_vertex(2)
vertices, edges = tree.get_vertices(), tree.get_edges()

render_tree(ax2, vertices, edges)
for i_vert, vert in enumerate(vertices):
    i_parent = edges[i_vert]
    ax2.annotate(f"{i_vert}: {i_parent}", vert)

ax2.set_aspect("equal")
ax2.set_title("After pruning")


plt.show()
