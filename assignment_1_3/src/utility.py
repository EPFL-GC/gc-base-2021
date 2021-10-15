import numpy as np
import cv2 as cv
import triangle as tr

def triangulate_mesh(Ps, Holes, area_density):
    '''
    Triangulates a set of points given by P using Delaunay scheme

    Input:
    - P : list of N lists of 2 elements giving the vertices positions
    - Holes: list of M lists of 2 elements giving the holes positions
    - area_density: the maximum triangle area

    Output:
    - V : array of shape (nNodes, 3) containing the vertices position
    - F : array of vertices ids (list of length nTri of lists of length 3). The ids are such that the algebraic area of
          each triangle is positively oriented
    '''

    num_points = 0
    points = []
    segments = []
    for k, _ in enumerate(Ps):
        P = np.array(Ps[k])
        N = P.shape[0]
        points.append(P)
        index = np.arange(N)
        seg = np.stack([index, index + 1], axis=1) % N + num_points
        segments.append(seg)
        num_points += N

    points = np.vstack(points)
    segments = np.vstack(segments)

    data = []
    if Holes == [] or Holes == [[]] or Holes == None:
        data = dict(vertices=points, segments=segments)
    else:
        data = dict(vertices=points, segments=segments, holes = Holes)

    tri = tr.triangulate(data, 'qpa{}'.format(area_density))

    V = np.array([[v[0], v[1], 0] for v in tri["vertices"]])
    F = np.array([[f[0], f[1], f[2]] for f in tri["triangles"]])
    return [V, F]