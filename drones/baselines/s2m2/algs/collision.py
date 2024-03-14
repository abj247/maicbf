import numpy as np
from numpy import linalg, float16, float64
from math import *
import pypoman as ppm


from models.agent import *

def traj2obs(ref, model, e = 0.001, steps = 10):
    O = []
    time_step = (ref[-1][0] - ref[0][0]) / steps
    rref = ref.copy()
    # print("Start", ref[0][0], "End", ref[-1][0])


    for j in range(1,steps):
        t = j * time_step + rref[0][0]
        # print(j, t)
        index = next((i for i in range(len(rref) - 1)
                      if rref[i][0] <= t and t <= rref[i + 1][0]), None)
        assert index != None
        t1, t2 = rref[index][0], rref[index+1][0]
        p1, p2 = np.array(rref[index][1:]), np.array(rref[index+1][1:])
        k = (p2 - p1) / (t2 - t1)

        p = p1 + k * (t - t1)
        rref.insert(index+1, [t] + [x for x in p])

    # Each Segment To Obstacle
    for i in range(len(rref)-1):
        # print("Traj Seg[%s/%s] to Moving Obstacles"%(i, len(rref)-1))
        t1, t2 = rref[i][0], rref[i+1][0]
        p1, p2 = np.array(rref[i][1:]), np.array(rref[i+1][1:])
        R = model.size + model.bloating(i) + e
        # print(R)
        # Static during This Segment
        if t2 == t1: k = np.array([0] * len(p1))
        # Moving during This Segment
        else: k = (p2 - p1) / (t2 - t1)
        obs = seg2obs(t1, t2, k, R, p1)
        O.append(obs)
    # Final Position as A Staic Obstacle
    # print("Traj Seg[%s/%s] to Moving Obstacles" % (len(rref)-1, len(rref)-1))
    obs = seg2obs(rref[-1][0], None, np.array([0] * len(p1)), model.size + e, np.array(rref[-1][1:]))
    O.append(obs)

    return O

def seg2obs(lb, ub, k, R, p):
    for idx in range(len(k)):
        if -0.01 < k[idx] < 0.01: k[idx] = 0

    norm = linalg.norm(k)

    if len(k) == 2:
        if linalg.norm(k) != 0:
            k = k / linalg.norm(k)
            vk = np.array([-k[1], k[0]])
            # print(linalg.norm(k))
            #  print("Direction", k, "Vertical Direction", vk)
            vertices = [p + R * (- k + vk), p + R * (- k - vk),
                        p + k * norm * (ub - lb) + R * (k + vk),
                        p + k * norm * (ub - lb) + R * (k - vk)]
        else:
            vertices = [p + np.array([R, R]),
                        p + np.array([R, -R]),
                        p + np.array([-R, R]),
                        p + np.array([-R, -R])]
        # TODO CDD LIB cannot Handle Float64

    elif len(k) == 3:
        if linalg.norm(k) != 0:
            k = k / linalg.norm(k)
            k1, k2 = [0] * 3, [0] * 3
            for ia, ib, ic in [[0, 1, 2], [0, 2, 1], [2, 0, 1]]:
                k1[ia], k1[ib], k1[ic] = [-k[ic], -k[ic], k[ia] + k[ib]]
                if linalg.norm(np.array(k1)) > 1e-3: break
            assert(linalg.norm(np.array(k1)) > 1e-3)
            k2[ia], k2[ib], k2[ic] = [-k[ia]*k[ib] - k[ib]*k[ib] - k[ic]*k[ic],
                                     k[ia]*k[ia] + k[ia]*k[ib] + k[ic]*k[ic],
                                    k[ia]*k[ic] - k[ib]*k[ic]]
            k1 = np.array(k1)
            k2 = np.array(k2)
            k1 = float64(k1 / linalg.norm(k1))
            k2 = float64(k2 / linalg.norm(k2))
            vertices = [p + R * (- k + k1 + k2),
                        p + R * (- k + k1 - k2),
                        p + R * (- k - k1 + k2),
                        p + R * (- k - k1 - k2),
                        p + k * norm * (ub - lb) + R * (- k + k1 + k2),
                        p + k * norm * (ub - lb) + R * (- k + k1 - k2),
                        p + k * norm * (ub - lb) + R * (- k - k1 + k2),
                        p + k * norm * (ub - lb) + R * (- k - k1 - k2)]
        else:
            vertices = [p + np.array([R, R, R]),
                        p + np.array([R, R, -R]),
                        p + np.array([R, -R, R]),
                        p + np.array([R, -R, -R]),
                        p + np.array([-R, R, R]),
                        p + np.array([-R, R, -R]),
                        p + np.array([-R, -R, R]),
                        p + np.array([-R, -R, -R])]
        # TODO CDD LIB cannot Handle Float64
    vertices = [float64(v) for v in vertices]
    #print(lb, ub, k, R, p)
    #print(vertices)
    # print("\n")
    A, b = ppm.compute_polytope_halfspaces(vertices)
    return lb, ub, A, b

def find_collision(refA, modelA, refB, modelB, request = 'first'):
    rA = modelA.size
    rB = modelB.size
    bloatingA = modelA.bloating
    bloatingB = modelB.bloating

    refA = refA + [np.array([1e4] + list(refA[-1][1:]))]
    refB = refB + [np.array([1e4] + list(refB[-1][1:]))]
    areaA, areaB = [], []

    timesA = [refA[i][0] for i in range(len(refA))]
    timesB = [refB[i][0] for i in range(len(refB))]
    times = sorted(list(set().union(timesA, timesB)))

    # print("A, Times", timesA)
    # print("B, Times", timesB)
    # print("Times", times)

    for i in range(len(times)-1):
        start_time, end_time = times[i], times[i+1]
        # print("\n-- [%s, %s]"%(start_time, end_time))

        indexA = next((i for i in range(len(timesA)-1)
                       if timesA[i] <= start_time and end_time <= timesA[i+1]), None)
        assert indexA != None
        tA1, tA2 = timesA[indexA], timesA[indexA+1]
        pA1, pA2 = np.array(refA[indexA][1:]), np.array(refA[indexA+1][1:])
        RA = rA
        if indexA + 1 != len(refA) - 1: RA = RA + + bloatingA(indexA)

        kA = (pA2 - pA1) / (tA2 - tA1)
        bA = pA1 + kA * (start_time - tA1) # positions at start_time of this segment

        indexB = next((i for i in range(len(timesB) - 1)
                       if timesB[i] <= start_time and timesB[i+1] >= end_time), None)
        assert indexB != None
        tB1, tB2 = timesB[indexB], timesB[indexB+1]
        pB1, pB2 = np.array(refB[indexB][1:]), np.array(refB[indexB+1][1:])
        RB = rB
        if indexB + 1 != len(refB) - 1: RB = RB + + bloatingB(indexB)
        kB = (pB2 - pB1) / (tB2 - tB1)
        bB = pB1 + kB * (start_time - tB1)

        kAB = kA - kB
        bAB = bA - bB
        R = RA + RB
        if len(kAB) == 2:
            a = kAB[0] ** 2 + kAB[1] ** 2
            b = 2 * kAB[0] * bAB[0] + 2 * kAB[1] * bAB[1]
            c = bAB[0] ** 2 + bAB[1] ** 2 - R ** 2
        elif len(kAB) == 3:
            a = kAB[0] ** 2 + kAB[1] ** 2 + kAB[2] ** 2
            b = 2 * kAB[0] * bAB[0] + 2 * kAB[1] * bAB[1] + 2 * kAB[2] * bAB[2]
            c = bAB[0] ** 2 + bAB[1] ** 2 + bAB[2] ** 2 - R ** 2

        # print("IndexA: %s, IndexB: %s"%(indexA, indexB))
        # print("A: [%s -> %s]"% (refA[indexA], refA[indexA+1]))
        # print("B: [%s -> %s]"% (refB[indexB], refB[indexB+1]))
        # print("R:", R, "RA:", RA, "RB", RB)
        # print("kAB:", kAB, "kA:", kA, "kB:", kB)
        # print("bAB:", bAB, "bA:", bA, "bB:", bB)
        # print("a = %s, b = %s, c = %s"%(a, b, c))
        if a == 0:
            # print("Constant Relative Distance")
            assert b == 0
            if c <= 0:
                lb, ub = start_time, end_time
                areaA.append([lb, ub, kA, RA, bA])
                areaB.append([lb, ub, kB, RB, bB])
                # print("Collision Found")
                # print("A", areaA)
                # print("B", areaB)
                if request == 'first':
                    return [areaA[0], areaB[0]]
        elif b ** 2 - 4 * a * c >= 0:
            d = sqrt(b ** 2 - 4 * a * c)
            roots = sorted([(-b - d) / (2 * a), (-b + d) / (2 * a)])
            lb, ub = np.array(roots) + np.array([start_time, start_time])  # add time offset
            # print("Meet during [%s, %s]"%(lb, ub))
            if start_time <= ub and lb <= end_time:
                lb = max(start_time, lb)
                ubA = min(end_time, ub)
                ubB = min(end_time, ub)
                areaA.append([lb, ubA, kA, RA, bA + kA * (lb - start_time)])
                areaB.append([lb, ubB, kB, RB, bB + kB * (lb - start_time)])
                # print("Collision Found")
                # print("A", areaA)
                # print("B", areaB)
                if request == 'first': return [areaA[0], areaB[0]]

    if request == 'first': return [None, None]
    elif request == 'all': return [areaA, areaB]