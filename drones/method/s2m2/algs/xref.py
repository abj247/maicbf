from __future__ import division

from math import *
import polytope as pc
from gurobi import *
from numpy import linalg
from models.agent import *
from timeit import *

def add_initial_constraints(problem, vars, Theta):
    # print("Add Intiail Constraint for Agent")
    dim = len(Theta)
    problem.addConstrs(vars[i] == Theta[i] for i in range(dim))

def add_final_constraints(problem, vars, Goal, r):
    # print("Add Final Constraint for Agent")
    Af = Goal[0]
    bf = Goal[1]

    # Make sure that ending point is at the center of the goal set
    poly = pc.Polytope(Af, bf)
    pt = poly.chebXc  # Find the center of the goal set
    problem.addConstrs(vars[i] == pt[i] for i in range(len(pt)))

    # # Additionally, make sure all the error is contained within the goal set
    # for row in range(len(Af)):
    #     b = bf[row][0] - linalg.norm(Af[row]) * r
    #     problem.addConstr(LinExpr(Af[row], vars) <= b)

def add_obstacle_constraints(problem, p1, p2, O, r):
    # print("Add Reference Constraints for Agent")
    if O == []: return None
    # Add the constraints to avoid the obstacles
    for (Ao, bo) in O:
        flags = problem.addVars(len(Ao), vtype=GRB.BINARY)
        problem.addConstr(flags.sum() <= len(Ao) - 1)
        for row in range(len(Ao)):
            b = bo[row] + linalg.norm(Ao[row]) * r
            problem.addConstr(LinExpr(Ao[row], p1) + 1e5 * flags[row] >= b)
            problem.addConstr(LinExpr(Ao[row], p2) + 1e5 * flags[row] >= b)

def add_moving_obstacle_constraints(problem, p1, p2, t1, t2, MO, r):
    # print("Add Reference Constraints for Agent")
    if MO == []: return None
    # Add the constraints to avoid the obstacles

    for lb, ub, Ao, bo in MO:
       # Distance Constraints or Our of Time Window
        if t2 == None and ub == None:
            flags = problem.addVars(len(Ao), vtype=GRB.BINARY)
            problem.addConstr(flags.sum() <= len(Ao)-1)
            for row in range(len(Ao)):
                b = bo[row] + linalg.norm(Ao[row]) * r
                problem.addConstr(LinExpr(Ao[row], p1) + 1e5 * flags[row] >= b)
                problem.addConstr(LinExpr(Ao[row], p2) + 1e5 * flags[row] >= b)
        elif t2 == None:
            flags = problem.addVars(len(Ao) + 1, vtype=GRB.BINARY)
            problem.addConstr(flags.sum() <= len(Ao))
            problem.addConstr(t1 + 1e5 * flags[len(Ao)] >= ub)
            for row in range(len(Ao)):
                b = bo[row] + linalg.norm(Ao[row]) * r
                problem.addConstr(LinExpr(Ao[row], p1) + 1e5 * flags[row] >= b)
                problem.addConstr(LinExpr(Ao[row], p2) + 1e5 * flags[row] >= b)
        elif ub == None:
            flags = problem.addVars(len(Ao) + 1, vtype=GRB.BINARY)
            problem.addConstr(flags.sum() <= len(Ao))
            problem.addConstr(t2 - 1e5 * flags[len(Ao)] <= lb)
            for row in range(len(Ao)):
                b = bo[row] + linalg.norm(Ao[row]) * r
                problem.addConstr(LinExpr(Ao[row], p1) + 1e5 * flags[row] >= b)
                problem.addConstr(LinExpr(Ao[row], p2) + 1e5 * flags[row] >= b)
        else:
            flags = problem.addVars(len(Ao) + 2, vtype=GRB.BINARY)
            problem.addConstr(flags.sum() <= len(Ao) + 1)
            problem.addConstr(t1 + 1e5 * flags[len(Ao)] >= ub)
            problem.addConstr(t2 - 1e5 * flags[len(Ao)+1] <= lb)
            for row in range(len(Ao)):
                b = bo[row] + linalg.norm(Ao[row]) * r
                problem.addConstr(LinExpr(Ao[row], p1) + 1e5 * flags[row] >= b)
                problem.addConstr(LinExpr(Ao[row], p2) + 1e5 * flags[row] >= b)

def add_avoiding_constraints_linear(problem, pA1, pA2, pB1, pB2, r):
    basis_num = 2
    dim = len(pA1)
    bases = ball_bases(basis_num, dim = dim)
    # distance variables LAs, LBs
    lAs = problem.addVars(basis_num, vtype=GRB.CONTINUOUS, lb=0)
    lBs = problem.addVars(basis_num, vtype=GRB.CONTINUOUS, lb=0)
    for i in range(basis_num):
        basis = bases[i]
        # distance lower bound
        problem.addConstr(lAs[i] >= LinExpr(basis, pA1) - LinExpr(basis, pA2))
        problem.addConstr(lAs[i] >= LinExpr(basis, pA2) - LinExpr(basis, pA1))
        problem.addConstr(lBs[i] >= LinExpr(basis, pB1) - LinExpr(basis, pB2))
        problem.addConstr(lBs[i] >= LinExpr(basis, pB2) - LinExpr(basis, pB1))
    # distance variables LAB
    lAB2 = problem.addVar(vtype=GRB.CONTINUOUS, lb=0)
    lAB_flags = problem.addVars(2 * basis_num, vtype=GRB.BINARY)

    for i in range(basis_num):
        basis = bases[i]
        # distance upper bound
        problem.addConstr(lAB2 - 1e5 * lAB_flags[2 * i]
                        <= LinExpr(basis, pA1) + LinExpr(basis, pA2)
                        - LinExpr(basis, pB1) - LinExpr(basis, pB2))
        problem.addConstr(lAB2 - 1e5 * lAB_flags[2 * i + 1]
                        <= - LinExpr(basis, pA1) - LinExpr(basis, pA2)
                        + LinExpr(basis, pB1) + LinExpr(basis, pB2))
    # Distance Constraints
    problem.addConstr(lAB2 >= lAs.sum() + lBs.sum() + 2 * r)

def add_constraints(problem, vars, N, dim,
                    Thetas, Goals, O, MO,
                    velocities, sizes, bloat_funcs, min_dur):
    # Get all the constraints in one function
    agent_num = len(Thetas)
    add_time_constraints(problem, vars, agent_num, velocities, N, dim, min_dur)
    # Add constraints for each agent
    for idx in range(agent_num):
        errors = [bloat_funcs[idx](n) + sizes[idx] for n in range(N+1)]
        p0 = [vars['(A%sN%sD%s)' % (idx, 0, k)] for k in range(dim)]
        pN = [vars['(A%sN%sD%s)' % (idx, N, k)] for k in range(dim)]
        add_initial_constraints(problem, p0, Thetas[idx])
        add_final_constraints(problem, pN, Goals[idx], errors[N])
        for n in range(N):
            p1 = [vars['(A%sN%sD%s)' % (idx, n, k)] for k in range(dim)]
            p2 = [vars['(A%sN%sD%s)' % (idx, n+1, k)] for k in range(dim)]
            t1, t2 = vars['T%s'%(n)], vars['T%s'%(n+1)]
            add_obstacle_constraints(problem, p1, p2, O, errors[n])
            add_moving_obstacle_constraints(problem, p1, p2, t1, t2, MO, errors[n])
        pN = [vars['(A%sN%sD%s)' % (idx, N, k)] for k in range(dim)]
        tN = vars['T%s'%(N)]
        add_moving_obstacle_constraints(problem, pN, pN, tN, None, MO, sizes[idx])
    for i in range(agent_num):
        for j in range(i):
            for n in range(N):
                pA1 = [vars['(A%sN%sD%s)' % (i, n, k)] for k in range(dim)]
                pA2 = [vars['(A%sN%sD%s)' % (i, n+1, k)] for k in range(dim)]
                pB1 = [vars['(A%sN%sD%s)' % (j, n, k)] for k in range(dim)]
                pB2 = [vars['(A%sN%sD%s)' % (j, n+1, k)] for k in range(dim)]
                rA = bloat_funcs[i](n)
                rB = bloat_funcs[j](n)
                r = rA + rB + sizes[i] + sizes[j]
                add_avoiding_constraints_linear(problem, pA1, pA2, pB1, pB2, r)

def get_variables(problem, agent_num, N, dim, limits):
    vars = {}
    for i in range(agent_num):
        for n in range(N + 1):
            for k in range(dim):
                lb, ub = limits[k]
                name = '(A%sN%sD%s)' % (i, n, k)
                vars[name] = problem.addVar(vtype=GRB.CONTINUOUS, name= name, lb=lb, ub=ub)
    for n in range(N+1): vars['T%s'%(n)] = problem.addVar(vtype=GRB.CONTINUOUS,
                                                        name = 'T%s'%(n), lb=0, ub=1e3)
    return vars

def add_time_constraints(problem, vars, agent_num, velocities, N, dim, min_dur):
    problem.addConstr(vars['T0'] == 0)
    for idx in range(agent_num):
        v = velocities[idx]
        for n in range(N):
            t1 = v * vars['T%s'%(n)]
            t2 = v * vars['T%s'%(n + 1)]
            p1 = [vars['(A%sN%sD%s)' % (idx, n, k)] for k in range(dim)]
            p2 = [vars['(A%sN%sD%s)' % (idx, n+1, k)] for k in range(dim)]
            problem.addConstr(t2 - t1 >= min_dur)
            if dim == 2:
                for basis in ball_bases(N = 3, dim = dim):
                    problem.addConstr(v * t2 - v * t1 >= LinExpr(basis, p1) - LinExpr(basis, p2))
                    problem.addConstr(v * t2 - v * t1 >= LinExpr(basis, p2) - LinExpr(basis, p1))
  
            elif dim == 3:
                problem.addConstr(v * t2 - v * t1 >= p1[0]-p2[0]+p1[1]-p2[1]+p1[2]-p2[2])
                problem.addConstr(v * t2 - v * t1 >= p1[0]-p2[0]+p1[1]-p2[1]+p2[2]-p1[2])
                problem.addConstr(v * t2 - v * t1 >= p1[0]-p2[0]+p2[1]-p1[1]+p1[2]-p2[2])
                problem.addConstr(v * t2 - v * t1 >= p1[0]-p2[0]+p2[1]-p1[1]+p2[2]-p1[2])
                problem.addConstr(v * t2 - v * t1 >= p2[0]-p1[0]+p1[1]-p2[1]+p1[2]-p2[2])
                problem.addConstr(v * t2 - v * t1 >= p2[0]-p1[0]+p1[1]-p2[1]+p2[2]-p1[2])
                problem.addConstr(v * t2 - v * t1 >= p2[0]-p1[0]+p2[1]-p1[1]+p1[2]-p2[2])
                problem.addConstr(v * t2 - v * t1 >= p2[0]-p1[0]+p2[1]-p1[1]+p2[2]-p1[2])

def set_objective(problem, vars, N):
    problem.setObjective(vars['T%s'%(N)], GRB.MINIMIZE)

def get_gb_xref(models, Thetas, Goals, limits, O, MO, N_min, N, min_dur = 0):
    # print("Start Searching for Reference Trajectories")
    agent_num = len(models)
    velocities = [model.velocity for model in models]
    sizes = [model.size for model in models]
    bloat_funcs = [model.bloating for model in models]
    dim = len(Goals[0][0][0])
    for n in range(N_min, N):
        # print("--------")
        # print("Horizon", n)
        problem = Model("xref")
        problem.setParam(GRB.Param.OutputFlag, 0)
        problem.setParam(GRB.Param.IntFeasTol, 1e-9)
        problem.setParam(GRB.Param.MIPGap, 0.5)
        # problem.setParam(GRB.Param.NonConvex, 2) # Temporary
        vars = get_variables(problem, agent_num, n, dim, limits)
        add_constraints(problem, vars, n, dim,
                        Thetas, Goals, O, MO,
                        velocities, sizes, bloat_funcs, min_dur)
        set_objective(problem, vars, n)
        # problem.write("test.lp")
        start = default_timer()
        problem.optimize()
        end = default_timer()
        if problem.status == GRB.OPTIMAL:
            # for v in problem.getVars(): print(v.varName, v.x)
            ma_nodes = []
            front_time = -1
            for idx in range(agent_num):
                nodes = []
                for i in range(n + 1):
                    time = vars['T%s'%(i)].x
                    x_new = [time]
                    for k in range(dim):
                        x_new = x_new + [vars['(A%sN%sD%s)' % (idx, i, k)].x]
                    if x_new[0] > front_time:
                        front_time = x_new[0]
                        nodes = nodes + [x_new]
                ma_nodes = ma_nodes + [nodes]
            problem.dispose()
            print_solution(ma_nodes, end-start)
            return ma_nodes
        else:
            # print('No solution')
            problem.dispose()

def ball_bases(N=3, dim=2):
    if dim == 2:
        p = pi / N
        bases = [[cos(n * p), sin(n * p)] for n in range(N)]
        # print(bases)
        return bases
    elif dim == 3:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

def print_solution(ma_nodes, time):
    agent_num = len(ma_nodes)
    for idx in range(agent_num):
        print("Printing Solution with Solving Time ", time, "s")
        # print("Agent %s Solution"%(idx))
        nodes = ma_nodes[idx]
        seg_num = len(nodes)
        for i in range(seg_num): print("[%s] %s"%(i, nodes[i]))