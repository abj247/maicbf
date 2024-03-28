from algs.decentralized import decentralized_algo
from algs.ref2traj import ref2traj
from problems.util import *
from timeit import *
from util import *

from viz.plot import *
from viz.animate import *
from viz.util import *

def test(env, problem_path, config_path):
    name, limits, Obstacles, agents, Thetas, Goals = read_problem(problem_path)
    min_segs, max_segs, obs_steps = read_configuration(config_path)



    start = default_timer()
    refs = decentralized_algo(agents, Thetas, Goals, limits, Obstacles, min_segs, max_segs, obs_steps, 0)
    end = default_timer()
    print("Total Time = ", end - start)
    name = '[%s]'%(env)

    trajs = ref2traj(refs)
    # plot_results(agents, limits, Obstacles, Thetas, Goals, trajs, name, refs=refs)

    return refs

test("zigzag", "problems/zigzag/problem.yaml", "problems/zigzag/config.yaml")
