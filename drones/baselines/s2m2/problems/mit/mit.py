from util import *
from problems.util import *

Obstacles = [make_box_line(-12, -8, 2, 2, 0, 5.8),
			 make_box_line(8, 12, 2, 2, 0, 5.8)]

Thetas = [[-6, -2, 3],
		  [-4, -2, 3],
		  [-2, -2, 3],
		  [0, -2, 3],
		  [2, -2, 3],
		  [4, -2, 3],
		  [6, -2, 3],

		  [-2, -4, 3],
		  [-2, -6, 3],
		  [-2, -8, 3],
		  [-4, -8, 3],
		  [-6, -8, 3],

		  [2, -4, 3],
		  [2, -6, 3],
		  [2, -8, 3],
		  [4, -8, 3],
		  [6, -8, 3],
		  ]

Goals = [make_box_center(-14, 8, 1, 2, 2, 2),
		 make_box_center(-12, 8, 3, 2, 2, 2),
		 make_box_center(-10, 8, 5, 2, 2, 2),
		 make_box_center(-8, 8, 3, 2, 2, 2),
		 make_box_center(-6, 8, 1, 2, 2, 2),
		 make_box_center(-4, 8, 3, 2, 2, 2),
		 make_box_center(-2, 8, 5, 2, 2, 2),
		 make_box_center(0, 8, 3, 2, 2, 2),
		 make_box_center(2, 8, 1, 2, 2, 2),

		 make_box_center(6, 8, 1, 2, 2, 2),
		 make_box_center(6, 8, 3, 2, 2, 2),
		 make_box_center(6, 8, 5, 2, 2, 2),

		 make_box_center(10, 8, 5, 2, 2, 2),
		 make_box_center(12, 8, 5, 2, 2, 2),
		 make_box_center(12, 8, 3, 2, 2, 2),
		 make_box_center(12, 8, 1, 2, 2, 2),
		 make_box_center(14, 8, 5, 2, 2, 2),]


limits = [[-16, 16], [-10, 10], [0.5, 6]]

size = 0.5
velocity = 1
k = [2, 2, 2]
file_name = "problem.yaml"
name = "mit"

agents = [AUV(size, velocity, k) for _ in range(17)]
write_problem(file_name, name, limits, Obstacles, agents, Thetas, Goals)

