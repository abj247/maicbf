from algs.xref import *
from algs.collision import *
from queue import *
from networkx import *
from copy import deepcopy
from timeit import *

# Main controller algorithm
def decentralized_algo(models, thetas, goals, limits, obstacles,
                       min_segs, max_segs, obs_steps, min_dur, timeout = 300):
    agent_num = len(models)
    params = [models, thetas, goals, limits, obstacles, min_segs, max_segs, obs_steps, min_dur]

    print("\n----- 0 -----")
    root = Node(agent_num)
    for idx in range(agent_num):
        plan = root.update_plan(idx, params)
        if plan == None:
            return None
    q = PriorityQueue()
    q.put(root)
    times = 0
    start_time = default_timer()
    while not q.empty():
        if default_timer() - start_time > timeout:
            print("Timeout!")
            return None
        times = times + 1
        node = q.get()
        print("\n-----", times, "-----")
        print("Orders:", node.G.edges)
        print("Makespan =", node.makespan)
        # Check Collision
        colliding_agents = node.earliest_colliding_agents(params)
        print("Colliding Agents:", colliding_agents)
        # Solution Found
        if colliding_agents == None:
            print("[*] Solution Found!")
            return node.plans
        # Resolve Collision
        for (j, i) in [colliding_agents, (colliding_agents[1], colliding_agents[0])]:
            # Update Node
            print("\n[*] Split on (%s < %s)"%(j, i))
            new_node = node.copy()
            new_node = new_node.update_node((j, i), params)
            if new_node == None: print("Infeasiblae")
            else:
                print("Feasible and Push")
                q.put(new_node)
    return None

class Node:
    def __init__(self, num, plans = [], makespan = 0,
                G = DiGraph(), orders = set(),
                collisions = {}):
        # params [num]
        self.num = num
        # plans [plans, makespan]
        if plans == []: plans = [None for i in range(num)]
        self.plans = plans
        self.makespan = makespan
        # orders [G]
        G.add_nodes_from(list(range(num)))
        if orders != set():  G.add_edges_from(orders)
        self.G = G
        # collisions
        self.collisions = collisions # only specified for agents with certain orderigns

    def __lt__(self, other):
        # if (self.makespan < other.makespan): return True
        # elif (self.makespan > other.makespan): return False
        # else: return len(self.G) < len(other.G)

        if len(self.G.edges) > len(other.G.edges): return True
        elif len(self.G.edges) < len(other.G.edges): return False
        else:
            return self.makespan < other.makespan

    def copy(self, memodict={}):
        return Node(self.num,
                    plans = deepcopy(self.plans), makespan = self.makespan,
                    G = self.G.copy(), collisions = deepcopy(self.collisions))

    def agents(self):
        return set(list(range(self.num)))

    def higher_agents(self, idx):
        return ancestors(self.G, idx)

    def lower_agents(self, idx):
        return descendants(self.G, idx)

    def sorted_agents(self, nodes):
        H = self.G.subgraph(nodes)
        return list(topological_sort(H))

    def add_order(self, order):
        higher_agents = self.higher_agents(order[1])
        lower_agents = self.lower_agents(order[1])
        if order[0] in higher_agents: status = "existing"
        elif order[0] in lower_agents: status = "inconsistent"
        else:
            self.G.add_edge(*order)
            status = "added"
        return status

    def earliest_colliding_agents(self, params):

        models, _, _, _, _, _, _, _, _ = params

        earliest_time = 1e6
        colliding_agents = None
        for i in range(self.num):
            for j in self.agents() - set([i]):
                time = 1e8
                # print(" -- Check Collison for (%s, %s)" % (i, j))
                if ("%s-%s" % (i, j) in self.collisions
                        and self.collisions["%s-%s" % (i, j)] == "free"):
                    1
                    # print(" -- [Collision Free] Previously Checked.")
                elif ("%s-%s" % (i, j) in self.collisions
                      and self.collisions["%s-%s" % (i, j)] != "free"):
                    # print(" -- [Collision Found] Previously Checked.")
                    time = self.collisions["%s-%s" % (i, j)]
                else:
                    area, _ = find_collision(self.plans[i], models[i],
                                             self.plans[j], models[j])
                    if area != None:
                        # print(" -- [Collision Found].")
                        self.collisions["%s-%s" % (i, j)] = area[0]
                        self.collisions["%s-%s" % (j, i)] = area[0]
                        time = area[0]
                    else:
                        # print(" -- [Collision Free].")
                        self.collisions["%s-%s" % (i, j)] = "free"
                        self.collisions["%s-%s" % (j, i)] = "free"
                if time < earliest_time:
                    earliest_time = time
                    colliding_agents = (i, j)
                    print(" -- [\] Update Colliding Agents.", i, "-", j)
        return colliding_agents

    def update_node(self, order, params):
        # Check Ordering Status
        status = self.add_order(order)
        print("Adding Order Status: ", status)
        # assert status == "added"
        if status == "inconsistent": return None
        # Plan for All the Affected Agents
        idx = order[1]
        models, _, _, _, _, _, _, _, _ = params
        agents = [idx] # Only Update Affected Agents
        # agents = [idx] + self.sorted_agents(self.lower_agents(idx))
        print("Agents to Update:", agents)
        for k in agents:
            print("- Update Plan for Agent", k)
            higher_agents = self.higher_agents(k)
            # check colliding
            replan = False
            for j in higher_agents:
                # print(" -- Check Collison for (%s, %s)" % (k, j))
                if ("%s-%s" % (k, j) in self.collisions
                        and self.collisions["%s-%s" % (k, j)] == "free"):
                    1
                    # print(" -- [Collision Free] Previously Checked.")
                elif ("%s-%s" % (k, j) in self.collisions
                      and self.collisions["%s-%s" % (k, j)] != "free"):
                    # print(" -- [Collision Found] Previously Checked.")
                    replan = True
                    break
                else:
                    if self.plans[j] != None:
                        self.collisions["%s-%s" % (k, j)] != area[0]
                        self.collisions["%s-%s" % (j, k)] != area[0]
                        area, _ = find_collision(self.plans[k], models[k],
                                                 self.plans[j], models[j])
                    if area != None:
                        self.collisions["%s-%s" % (k, j)] != "free"
                        self.collisions["%s-%s" % (j, k)] != "free"
                        # print(" -- [Collision Found].")
                        replan = True
                        break
                    # else: print(" -- [Collision Free].")

            # Update If colliding
            if k == idx or replan:
                plan = self.update_plan(k, params)
                if plan == None:
                    print("- Update Fails for Agent", k)
                    return None
                print("- Update Succeed for Agent", k)
            else: print("- No need to update for Agent", k)
        return self

    def update_plan(self, k, params):

        models, thetas, goals, limits, obstacles, min_segs, max_segs, obs_steps, min_dur = params

        print("- Find Collision and Reroute")
        MO = []
        for j in self.higher_agents(k):
            MO = MO + traj2obs(self.plans[j], models[j], steps = obs_steps)
        # if self.plans[k] == None:
        #     Nmin, Nmax = min_segs, max_segs
        # else:
        #     Nmin = len(self.plans[k]) + 1
        #     Nmax = Nmin + 2
        Nmin, Nmax = min_segs, max_segs
        new_plan = get_gb_xref([models[k]], [thetas[k]], [goals[k]],
                            limits, obstacles, MO, Nmin, Nmax, min_dur = min_dur)
        if new_plan == None:
            print("- No Plan.")
            return None
        else:
            # Update Plan and Makespan
            plan = new_plan[0]
            self.plans[k] = plan
            self.makespan = max(plan[-1][0], self.makespan)
            self.makespan = sum([plan[-1][0] for plan in self.plans if plan != None])

            print("- Feasible Plan Found with Makespan", plan[-1][0])
            for j in self.higher_agents(k):
                self.collisions["%s-%s" % (k, j)] = "free"
                self.collisions["%s-%s" % (j, k)] = "free"
            # Unfreeze Collision Indicator
            for j in self.agents() - self.higher_agents(k):
                if "%s-%s" % (k, j) in self.collisions: self.collisions.pop("%s-%s" % (k, j))
                if "%s-%s" % (j, k) in self.collisions: self.collisions.pop("%s-%s" % (j, k))
            return plan
