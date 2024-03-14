import yaml
from models.agent import *
from models.auv import *
from models.car import *
from models.hovercraft import *


def write_results(file_name, paths, problem_path = None):

    data = dict()

    if problem_path is not None:
        name, limits, Obstacles, agents, Thetas, Goals = read_problem(problem_path)
        data['name'] = name
        data["limits"] = limits
        data["obstacles"] = [write_polytope(obs) for obs in Obstacles]
        data["agents"] = [write_agent(agent) for agent in agents]
        data["starts"] = Thetas
        data["goals"] = [write_polytope(goal) for goal in Goals]

    data["paths"] = paths

    yaml.Dumper.ignore_aliases = lambda *args: True
    with open(file_name, "w") as file:
        yaml.dump(data, file, indent=4)

def read_configuration(file_name):

    with open(file_name, "r") as file:
        data = yaml.load(file)

        min_segs = data["min_segs"]
        max_segs = data["max_segs"]
        obstacle_segs = data["obstacle_segs"]

    return [min_segs, max_segs, obstacle_segs]

def read_problem(file_name):

    with open(file_name, "r") as file:
        data = yaml.load(file)
        name = data["name"]
        limits = data["limits"]
        Obstacles = [read_polytope(obs) for obs in data["obstacles"]]
        agents = [read_agent(agent) for agent in data["agents"]]
        Thetas = data["starts"]
        Goals = [read_polytope(goal) for goal in data["goals"]]

    return [name, limits, Obstacles, agents, Thetas, Goals]

def write_problem(file_name, name, limits, Obstacles, agents, Thetas, Goals):
    data = dict()
    data['name'] = name
    data["limits"] = limits
    data["obstacles"] = [write_polytope(obs) for obs in Obstacles]
    data["agents"] = [write_agent(agent) for agent in agents]
    data["starts"] = Thetas
    data["goals"] = [write_polytope(goal) for goal in Goals]
    yaml.Dumper.ignore_aliases = lambda *args: True
    with open(file_name, "w") as file:
        yaml.dump(data, file, indent=4)

def write_agent(agent):
    agent_dict = dict()
    agent_dict["k"] = agent.k
    agent_dict["velocity"] = agent.velocity
    agent_dict["size"] = agent.size
    if isinstance(agent, Car):
        agent_dict["type"] = "car"
    elif isinstance(agent, AUV):
        agent_dict["type"] = "auv"
    elif isinstance(agent, Hovercraft):
        agent_dict["type"] = "hovercraf"

    return agent_dict

def read_agent(agent):
    type = agent["type"]
    k = agent["k"]
    velocity = agent["velocity"]
    size = agent["size"]
    if agent["type"] == "car":
        return Car(size, velocity, k)
    elif type == "auv":
        return AUV(size, velocity, k)
    elif type == "hovercraft":
        return Hovercraft(size, velocity, k)

def write_polytope(poly):
    A, b = poly
    lines = []
    for idx in range(len(A)):
        lines.append(A[idx].tolist()+b[idx].tolist())
    return lines

def read_polytope(poly):
    A = []
    b = []
    for idx in range(len(poly)):
        line = list(poly[idx])
        A.append(line[:-1])
        b.append([line[-1]])
    return [np.array(A), np.array(b)]
