from problems.util import *

limits = [[0, 60], [0, 60]]
Obstacles = []

Thetas = [[20, 48], [30, 48], [40, 48],
          [20, 12], [30, 12], [40, 12],
          [12, 20], [12, 30], [12, 40],
          [48, 20], [48, 30], [48, 40]]

A_rect = np.array([[-1, 0],
                   [1, 0],
                   [0, -1],
                   [0, 1]])
Goals = []
lG = [2, 2]
for pG in [[20, 12], [30, 12], [40, 12],
           [20, 48], [30, 48], [40, 48],
           [48, 20], [48, 30], [48, 40],
           [12, 20], [12, 30], [12, 40]]:
    b = np.array([[-(pG[0] - lG[0])], [pG[0] + lG[0]], [-(pG[1] - lG[1])], [pG[1] + lG[1]]])
    Goals.append([A_rect, b])

file_name = "ground/problem.yaml"
name = "ground"
size = 1
velocity = 1
k = [0.5, 0.5, 0.5]
agents = [Car(size, velocity, k) for _ in range(12)]

write_problem(file_name, name, limits, Obstacles, agents, Thetas, Goals)
