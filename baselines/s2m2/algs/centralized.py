from algs.xref import *

# Main controller algorithm
def centralized_algo(velocities, sizes, bloating_functions, Thetas, goals,
                     limits, obstacles, min_segs, max_segs):
    agent_num = len(sizes)
    collisions = [[]] * agent_num
    # Get the reference nodes
    print("Search Begins")
    ma_nodes = get_gb_xref(Thetas, goals, obstacles, limits,
                           velocities, sizes, bloating_functions, collisions,
                           min_segs, max_segs)
    if ma_nodes != None:
        print("Solution Found!")
        return ma_nodes
    print("Fail!")
    return None
