import yaml
import numpy as np
import csv

def save_reftrajs(trajs, path):
    data = dict()
    for idx in range(len(trajs)):
        data["Agent %s"%(idx)] = trajs[idx]
    yaml.Dumper.ignore_aliases = lambda *args: True
    with open(path, "w") as file:
        yaml.dump(data, file, indent=4)

def write_line(path, line):
    f = open(path, "a+")
    f.write(line)
    f.close()

def write_csv(path, list):
    with open(path, 'a+') as file:
        writer = csv.writer(file)
        writer.writerows([list])

def make_rectangle_line(xmin, xmax, ymin, ymax):
    A_rect = np.array([[-1, 0],
                       [1, 0],
                       [0, -1],
                       [0, 1]])
    b = np.array([[-xmin], [xmax], [-ymin], [ymax]])
    return (A_rect, b)

def make_rectangle_center(x, y, xl, yl):
    A_rect = np.array([[-1, 0],
                       [1, 0],
                       [0, -1],
                       [0, 1]])
    b = np.array([[-(x - xl/2)], [x + xl/2], [-(y - yl/2)], [y + yl/2]])

    return (A_rect, b)

def make_box_line(xmin, xmax, ymin, ymax, zmin, zmax):

    if not zmin and not zmax:
        A_cube = np.array([[-1, 0, 0],
                           [1, 0, 0],
                           [0, -1, 0],
                           [0, 1, 0]])
        b = np.array([[-xmin], [xmax], [-ymin], [ymax]])
    else:
        A_cube = np.array([[-1, 0, 0],
                           [1, 0, 0],
                           [0, -1, 0],
                           [0, 1, 0],
                           [0, 0, -1],
                           [0, 0, 1]])
        b = np.array([[-xmin], [xmax], [-ymin], [ymax], [-zmin], [zmax]])

    return (A_cube, b)

def make_box_center(x, y, z, xl, yl, zl):
    if not z and not zl:
        A_cube = np.array([[-1, 0, 0],
                           [1, 0, 0],
                           [0, -1, 0],
                           [0, 1, 0]])
        b = np.array([[-(x - xl / 2)], [x + xl / 2],
                      [-(y - yl / 2)], [y + yl / 2]])
    else:
        A_cube = np.array([[-1, 0, 0],
                           [1, 0, 0],
                           [0, -1, 0],
                           [0, 1, 0],
                           [0, 0, -1],
                           [0, 0, 1]])
        b = np.array([[-(x - xl / 2)], [x + xl / 2],
                      [-(y - yl / 2)], [y + yl / 2],
                      [-(z - zl / 2)], [z + zl / 2]])

    return (A_cube, b)
