from problems.util import *

A_rect = np.array([[-1,0],
				   [1,0],
				   [0,-1],
				   [0,1]])

b01 = np.array([[-25], [80], [-45], [50]])
b02 = np.array([[-8], [58], [-8], [19]])

b1 = np.array([[-40], [41], [-30], [40]])
b2 = np.array([[-1], [41], [-29], [30]])
b3 = np.array([[-1], [2], [-1], [29]])
b4 = np.array([[-2], [59], [-1], [2]])
b5 = np.array([[-58], [59], [-2], [40]])
b6 = np.array([[-58], [89], [-40], [41]])
b7 = np.array([[-88], [89], [-41], [58]])
b8 = np.array([[-15], [89], [-58], [59]])
b9 = np.array([[-15], [16], [-40], [58]])
b10 = np.array([[-15], [40], [-39], [40]])

Obstacles = [(A_rect, b01),
			 (A_rect, b02),
			 (A_rect, b1),
			 (A_rect, b2),
			 (A_rect, b3),
			 (A_rect, b4),
			 (A_rect, b5),
			 (A_rect, b6),
			 (A_rect, b7),
			 (A_rect, b8),
			 (A_rect, b9),
			 (A_rect, b10)]

Thetas = [[35, 55],
		  [50, 55],
		  [65, 55]]


Goals = []
A_rect = np.array([[-1,0],
				   [1,0],
				   [0,-1],
				   [0,1]])

pG1 = [45, 5]
lG1 = [10, 10]
b3 = np.array([[-(pG1[0] - lG1[0])], [pG1[0] + lG1[0]], [-(pG1[1] - lG1[1])], [pG1[1] + lG1[1]]])
Goals.append((A_rect, b3))

pG2 = [30, 5]
lG2 = [10, 10]
b4 = np.array([[-(pG2[0] - lG2[0])], [pG2[0] + lG2[0]], [-(pG2[1] - lG2[1])], [pG2[1] + lG2[1]]])
Goals.append((A_rect, b4))

pG3 = [15, 5]
lG3 = [10, 10]
b5 = np.array([[-(pG3[0] - lG3[0])], [pG3[0] + lG3[0]],  [-(pG3[1] - lG3[1])], [pG3[1] + lG3[1]]])
Goals.append((A_rect, b5))

limits = [[0, 90], [0, 60]]

file_name = "problem.yaml"
name = "parking"

size = 2.5
velocity = 1
k = [0.5, 0.5, 0.5]
agents = [Car(size, velocity, k) for _ in range(3)]
write_problem(file_name, name, limits, Obstacles, agents, Thetas, Goals)