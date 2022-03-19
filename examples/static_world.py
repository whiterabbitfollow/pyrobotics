import pyrb
from pyrb.world import StaticBoxesWorld

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib
import numpy as np

matplotlib.rc("font", size=16)

world = StaticBoxesWorld()
world.reset()
robot = world.robot

rectangles = []
angle_rad_total = 0
for angle_rad, link in zip(robot.config, robot.links):
    angle_rad_total += angle_rad
    angle = np.rad2deg(angle_rad_total)
    height = link.height
    width = link.width
    lower_left_corner_local = np.array([0, -height / 2, 0])
    lower_left_corner_global = pyrb.kin.SE3_mul(link.transform, lower_left_corner_local)
    xy = lower_left_corner_global[:2]
    rectangles.append(Rectangle(xy, width, height, angle=angle))

static_obstacles = []
for obs in world.obstacle_region.obstacles:
    p_local = np.array([-obs.width / 2, -obs.height / 2, 0])
    p_global = pyrb.kin.SE3_mul(obs.transform, p_local)
    static_obstacles.append(Rectangle(tuple(p_global)[:2], obs.width, obs.height, angle=np.rad2deg(obs.angle)))

fig, (ax, ax1) = plt.subplots(1, 2, figsize=(15, 6))
ax.add_collection(PatchCollection(rectangles, color="blue", edgecolor="black"))
ax.add_collection(PatchCollection(rectangles, color="blue", edgecolor="black"))
for angle_rad, link in zip(robot.config, robot.links):
    xy = link.transform[:2, 3]
    ax.scatter(xy[0], xy[1], color="black")

ax.add_collection(PatchCollection(static_obstacles, color="green"))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("World, $\mathcal{W} = \mathbb{R}^2$")

thetas_raw = np.linspace(-np.pi, np.pi, 100)
theta_grid_1, theta_grid_2 = np.meshgrid(thetas_raw, thetas_raw)
thetas = np.stack([theta_grid_1.ravel(), theta_grid_2.ravel()], axis=1)

collision_mask = []

for theta in thetas:
    robot.set_config(theta)
    collision = robot.collision_manager.in_collision_other(world.obstacle_region.collision_manager)
    collision_mask.append(not collision)

collision_mask = np.array(collision_mask).reshape(100, 100)
ax1.pcolormesh(theta_grid_1, theta_grid_2, collision_mask)
ax1.set_title("Configuration space, $\mathcal{C}$")
ax1.set_xlabel(r"$\theta_1$")
ax1.set_ylabel(r"$\theta_2$")

plt.show()
