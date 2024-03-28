from util import *
from problems.util import *

A_rect = np.array([[-1,0],
				   [1,0],
				   [0,-1],
				   [0,1]])
A_tri1 = np.array([[-1,-1],
				   [1,-1],
				   [0,1]])
A_tri2 = np.array([[-1,1],
				   [1,1],
				   [0,-1]])
b1 = np.array([[5], [20], [0]])
b2 = np.array([[-10], [35], [0]])
b3 = np.array([[-30], [0], [30]])
b4 = np.array([[-15], [-15], [30]])
b5 = np.array([[-45], [15], [30]])

b6 = np.array([[16], [51], [1], [0]])
b7 = np.array([[16], [51], [-30], [31]])
b8 = np.array([[16], [-15], [0], [30]])
b9 = np.array([[-50], [51], [0], [30]])

Obstacles = [(A_tri2, b1),
			 (A_tri2, b2),
			 (A_tri1, b3),
			 (A_tri1, b4),
			 (A_tri1, b5),
			 (A_rect, b6),
			 (A_rect, b7),
			 (A_rect, b8),
			 (A_rect, b9)]


Thetas = [[40, 5],
		  [40, 20],
		  [-10, 5],
		  [-10, 20]]



Goals = []

pG1 = [-10, 20]
lG1 = [2, 2]
b1 = np.array([[-(pG1[0] - lG1[0])], [pG1[0] + lG1[0]], [-(pG1[1] - lG1[1])], [pG1[1] + lG1[1]]])
Goal1 = (A_rect, b1)
Goals.append(Goal1)

pG2 = [-10, 5]
lG2 = [2, 2]
b2 = np.array([[-(pG2[0] - lG2[0])], [pG2[0] + lG2[0]], [-(pG2[1] - lG2[1])], [pG2[1] + lG2[1]]])
Goal2 = (A_rect, b2)
Goals.append(Goal2)

pG3 = [40, 20]
lG3 = [2, 2]
b3 = np.array([[-(pG3[0] - lG3[0])], [pG3[0] + lG3[0]],  [-(pG3[1] - lG3[1])], [pG3[1] + lG3[1]]])
Goal3 = (A_rect, b3)
Goals.append(Goal3)

pG4 = [40, 5]
lG4 = [2, 2]
b4 = np.array([[-(pG4[0] - lG4[0])], [pG4[0] + lG4[0]],  [-(pG4[1] - lG4[1])], [pG4[1] + lG4[1]]])
Goal4 = (A_rect, b4)
Goals.append(Goal4)


limits = [[-16, 51], [-1, 31]]


file_name = "problem.yaml"
name = "zigzag"

size = 1.5
velocity = 1
k = [0.5, 0.5, 0.5]

agents = [Car(size, velocity, k) for _ in range(4)]
write_problem(file_name, name, limits, Obstacles, agents, Thetas, Goals)



