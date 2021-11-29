import igl
import numpy as np
import remesher_helper
from matplotlib import cm
import meshplot as mp
cmap = cm.get_cmap('PiYG')


def normalize_area(v, f):
    cm = np.mean(v, axis=0)
    v -= cm
    v /= (np.sum(igl.doublearea(v, f))/2)**0.5
    v += cm


def has_zero_area_triangle(v, f):
    if np.min(igl.doublearea(v, f))/2 < 1e-10:
        return True
    return False


def reorder_mesh(raw_v, raw_f):
    np.random.shuffle(raw_f)
    bdry_vx_idx = np.array(list(set((igl.boundary_facets(raw_f)).flatten())))
    num_bdry_vx = len(bdry_vx_idx)
    num_intr_vx = len(raw_v) - num_bdry_vx
    reorder_vx_map = {}
    map_to_original = {}
    boundary_vertices = []
    interior_vertices = []

    for i in range(len(raw_v)):
        if i in bdry_vx_idx:
            reorder_vx_map[i] = len(boundary_vertices)
            map_to_original[len(boundary_vertices)] = i
            boundary_vertices.append(i)
        else:
            reorder_vx_map[i] = len(interior_vertices) + num_bdry_vx
            map_to_original[len(interior_vertices) + num_bdry_vx] = i
            interior_vertices.append(i)

    v = np.array([raw_v[map_to_original[i]] for i in range(len(raw_v))])
    f = np.array([[reorder_vx_map[i] for i in face] for face in raw_f])
    return v, f, num_bdry_vx, num_intr_vx


def parse_input_mesh(name):
    raw_v, raw_f = igl.read_triangle_mesh(name)
    v, f, num_bdry_vx, num_intr_vx = reorder_mesh(raw_v, raw_f)
    # normalize_area(v, f)
    return v, f, num_bdry_vx, num_intr_vx


def remesh(input_name, output_name, mesh_size = 0.03):
    vr, fr = remesher_helper.remesh(input_name, mesh_size)
    igl.write_obj(output_name, vr, fr.astype('int64'))


def get_diverging_colors(values, percentile = 95):
    max_val = np.percentile(np.abs(values), percentile)
    return(cmap(values / max_val * 0.5 * -1 + 0.5)[:, :3])

def plot_directions(x, F, d_1, d_2, scale=0.1):
    color = np.ones((len(x), 3))
    shading_options = {
        "flat": False,
        "wireframe":False,
        "metalness": 0.05,
    }
    p = mp.plot(x, F, c=color, shading=shading_options)
    p.add_lines(x+d_1*scale, x-d_1*scale, shading={"line_color": "red"})
    p.add_lines(x+d_2*scale, x-d_2*scale, shading={"line_color": "blue"})
    p.update_object()
