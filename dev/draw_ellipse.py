import matplotlib.pyplot as plt
import numpy as np


x_goal = np.array([1, 1])
x_start = np.array([0, 0])

x_center = (x_start + x_goal) / 2
x_dir = x_goal - x_start
c_min = np.linalg.norm(x_dir)
a_1 = (x_dir / c_min).reshape(-1, 1)
id_1 = np.zeros((a_1.size, 1))
id_1[0, 0] = 1
M = a_1 @ id_1.T

U, s, V_t = np.linalg.svd(M)
diag_vals = np.append(np.ones((s.size - 1,)), np.linalg.det(U) * np.linalg.det(V_t))
C = U @ np.diag(diag_vals) @ V_t

c_max = c_min * 1.05

r = np.zeros((2,))

r[0] = c_max / 2
r[1:] = np.sqrt(c_max ** 2 - c_min ** 2) / 2
L = np.diag(r)
CL = C @ L


thetas = np.linspace(0, np.pi * 2, 100)
xs = np.vstack([np.cos(thetas), np.sin(thetas)])
x_center = x_center.reshape(-1, 1)

xs = CL @ xs + x_center
xs = xs.T

plt.figure()
plt.plot(xs[:, 0], xs[:, 1])
plt.scatter(x_center[0], x_center[1])
plt.scatter(x_start[0], x_start[1])
plt.scatter(x_goal[0], x_goal[1])
plt.axis("equal")
plt.show()

