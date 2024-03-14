from numpy import linalg
import numpy as np
from scipy.spatial import ConvexHull
import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt
import pypoman as ppm
from shapely.geometry.polygon import Polygon
from models.auv import *
from models.hovercraft import *
from util import *

def plot_env(map_limits, Obstacles):
    fig = plt.figure()
    if len(map_limits) == 2:
        # Configure
        xmin, xmax = map_limits[0]
        ymin, ymax = map_limits[1]
        plt.xlim(xmin-2, xmax+2)
        plt.ylim(ymin-2, ymax+2)
        # # viz obstacles
        for A, b in Obstacles:
            poly = Polygon(ppm.duality.compute_polytope_vertices(A, b))
            x, y = poly.exterior.xy
            plt.fill(x, y, facecolor='r', alpha=0.3)
        ax = fig.gca()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.margins(0, 0)

    elif len(map_limits) == 3:
        ax = fig.add_subplot(111, projection='3d')
        xmin, xmax = map_limits[0]
        ymin, ymax = map_limits[1]
        zmin, zmax = map_limits[2]
        ax.set_xlim3d(xmin, xmax)
        ax.set_ylim3d(ymin, ymax)
        ax.set_zlim3d(zmin, zmax)
        # set_equal TODO
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_zaxis().set_visible(False)
        plt.margins(0, 0, 0)
        for obs in Obstacles:
            A, b = obs[0], obs[1]
            if len(A) == 4:
                A = np.array(A.tolist() + [[0, 0, -1], [0, 0, 1]])
                b = np.array(b.tolist() + [[-map_limits[2][0]], [map_limits[2][1]]])
            plot_polytope_3d(A, b, ax=ax, color='yellow', trans=0.2)

    return fig, ax

def plot_goals(Goals):
    True
    for A, b in Goals:
        poly = Polygon(ppm.duality.compute_polytope_vertices(A, b))
        x, y = poly.exterior.xy
        plt.fill(x, y, facecolor='g', alpha=0.3)

def plot_thetas(thetas):
    for x, y in thetas:
        A, b = make_rectangle_center(x, y, 5, 5)
        poly = Polygon(ppm.duality.compute_polytope_vertices(A, b))
        x, y = poly.exterior.xy
        plt.fill(x, y, facecolor='blue', alpha=0.3)

def extract_paths(models, ma_thetas, ma_segs):
    # viz initial states and trajectories
    agent_num = len(models)
    paths = []
    for idx in range(agent_num):
        theta = ma_thetas[idx]
        segs = ma_segs[idx]

        if len(theta) == 2:
            ref_x, ref_y, ref_theta, tru_x, tru_y, tru_theta, times =  [], [], [], [], [], [], []
            q0 = theta + [0]
            for seg in segs:
                t, qref, uref = seg
                ref_x = ref_x + [qref[i][0] for i in range(len(qref))]
                ref_y = ref_y + [qref[i][1] for i in range(len(qref))]
                ref_theta = ref_theta + [qref[i][2] for i in range(len(qref))]
                times = times + [t[i] for i in range(len(t))]

                run = models[idx].run_model
                q = run(q0, t, qref, uref)
                q0 = q[-1]
                tru_x = tru_x + [q[i][0] for i in range(len(q)-1)]
                tru_y = tru_y + [q[i][1] for i in range(len(q)-1)]
                tru_theta = tru_theta + [q[i][2] for i in range(len(q)-1)]
            paths.append([ref_x, ref_y, ref_theta, tru_x, tru_y, tru_theta, times])

        elif len(theta) == 3:

            if isinstance(models[0], AUV):
                q0 = theta + [0, 0, 0]
                ref_x, ref_y, ref_z, times = [], [], [], []
                tru_x, tru_y, tru_z, tru_thi, tru_theta, tru_psi = [], [], [], [], [], []
                for seg in segs:
                    t, qref, uref = seg
                    ref_x = ref_x + [qref[i][0] for i in range(len(qref))]
                    ref_y = ref_y + [qref[i][1] for i in range(len(qref))]
                    ref_z = ref_z + [qref[i][2] for i in range(len(qref))]
                    times = times + [t[i] for i in range(len(t))]

                    run = models[idx].run_model
                    q = run(q0, t, qref, uref)
                    q0 = q[-1]
                    tru_x = tru_x + [q[i][0] for i in range(len(q)-1)]
                    tru_y = tru_y + [q[i][1] for i in range(len(q)-1)]
                    tru_z = tru_z + [q[i][2] for i in range(len(q)-1)]
                    tru_thi = tru_thi + [q[i][3] for i in range(len(q)-1)]
                    tru_theta = tru_theta + [q[i][4] for i in range(len(q)-1)]
                    tru_psi = tru_psi + [q[i][5] for i in range(len(q)-1)]

                paths.append([ref_x, ref_y, ref_z,
                              tru_x, tru_y, tru_z, tru_thi, tru_theta, tru_psi, times])

            elif isinstance(models[0], Hovercraft):
                q0 = theta + [0, 0, 0]
                ref_x, ref_y, ref_z, tru_x, tru_y, tru_z, times = [], [], [], [], [], [], []

                for seg in segs:
                    t, qref, uref = seg
                    ref_x = ref_x + [qref[i][0] for i in range(len(qref))]
                    ref_y = ref_y + [qref[i][1] for i in range(len(qref))]
                    ref_z = ref_z + [qref[i][2] for i in range(len(qref))]
                    times = times + [t[i] for i in range(len(t))]

                    run = models[idx].run_model
                    q = run(q0, t, qref, uref)
                    q0 = q[-1]
                    tru_x = tru_x + [q[i][0] for i in range(len(q)-1)]
                    tru_y = tru_y + [q[i][1] for i in range(len(q)-1)]
                    tru_z = tru_z + [q[i][2] for i in range(len(q)-1)]
                paths.append([ref_x, ref_y, ref_z, tru_x, tru_y, tru_z, times])

    return paths

class Faces():
    def __init__(self,tri, sig_dig=12, method="convexhull"):
        self.method=method
        self.tri = np.around(np.array(tri), sig_dig)
        self.grpinx = list(range(len(tri)))
        norms = np.around([self.norm(s) for s in self.tri], sig_dig)
        _, self.inv = np.unique(norms,return_inverse=True, axis=0)

    def norm(self,sq):
        cr = np.cross(sq[2]-sq[0],sq[1]-sq[0])
        return np.abs(cr/np.linalg.norm(cr))

    def isneighbor(self, tr1,tr2):
        a = np.concatenate((tr1,tr2), axis=0)
        return len(a) == len(np.unique(a, axis=0))+2

    def order(self, v):
        if len(v) <= 3:
            return v
        v = np.unique(v, axis=0)
        n = self.norm(v[:3])
        y = np.cross(n,v[1]-v[0])
        y = y/np.linalg.norm(y)
        c = np.dot(v, np.c_[v[1]-v[0],y])
        if self.method == "convexhull":
            h = ConvexHull(c)
            return v[h.vertices]
        else:
            mean = np.mean(c,axis=0)
            d = c-mean
            s = np.arctan2(d[:,0], d[:,1])
            return v[np.argsort(s)]

    def simplify(self):
        for i, tri1 in enumerate(self.tri):
            for j,tri2 in enumerate(self.tri):
                if j > i:
                    if self.isneighbor(tri1,tri2) and \
                       self.inv[i]==self.inv[j]:
                        self.grpinx[j] = self.grpinx[i]
        groups = []
        for i in np.unique(self.grpinx):
            u = self.tri[self.grpinx == i]
            u = np.concatenate([d for d in u])
            u = self.order(u)
            groups.append(u)
        return groups

def plot_polytope_3d(A, b, ax = None, color = 'red', trans = 0.2):
    verts = np.array(ppm.compute_polytope_vertices(A, b))
    # compute the triangles that make up the convex hull of the data points
    hull = ConvexHull(verts)
    triangles = [verts[s] for s in hull.simplices]
    # combine co-planar triangles into a single face
    faces = Faces(triangles, sig_dig=1).simplify()
    # viz
    if ax == None:
        ax = a3.Axes3D(plt.figure())

    pc = a3.art3d.Poly3DCollection(faces,
                                   facecolor=color,
                                   edgecolor="k", alpha=trans)
    ax.add_collection3d(pc)
    # define view
    yllim, ytlim = ax.get_ylim()
    xllim, xtlim = ax.get_xlim()
    zllim, ztlim = ax.get_zlim()
    x = verts[:,0]
    x = np.append(x, [xllim, xtlim])
    y = verts[:,1]
    y = np.append(y, [yllim, ytlim])
    z = verts[:,2]
    z = np.append(z, [zllim, ztlim])
    ax.set_xlim(np.min(x)-1, np.max(x)+1)
    ax.set_ylim(np.min(y)-1, np.max(y)+1)
    ax.set_zlim(np.min(z)-1, np.max(z)+1)
