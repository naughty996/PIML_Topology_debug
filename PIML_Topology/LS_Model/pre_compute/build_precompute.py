import numpy as np
from LS_Model.set_bc import set_case

# 不进入循环，先计算完
def _compute_unique_pattern(rows: np.ndarray, cols: np.ndarray):
    pairs = np.stack([rows, cols], axis=1)
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return unique_pairs[:, 0].astype(np.int64), unique_pairs[:, 1].astype(np.int64), inverse.astype(np.int64)


def _regular_q4_density_shape_matrix(sample_n: int, include_boundary: bool) -> np.ndarray:
    """Precompute Q4 shape values on a regular x-by-x sampling grid.

    The matrix depends only on the sampling rule, not on the current level-set
    values.  Rows are sampling points and columns are the four Q4 nodes ordered
    consistently with set_ele_node_connect: [bottom-left, bottom-right,
    top-right, top-left].
    """
    n = max(1, int(sample_n))
    if bool(include_boundary) and n > 1:
        pts = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    else:
        # Interior midpoint-style samples.  For n=1, this gives s=t=0.
        idx = np.arange(n, dtype=np.float64)
        pts = -1.0 + (2.0 * idx + 1.0) / float(n)

    s_grid, t_grid = np.meshgrid(pts, pts, indexing='ij')
    s = s_grid.reshape(-1)
    t = t_grid.reshape(-1)
    return np.stack([
        0.25 * (1.0 - s) * (1.0 - t),
        0.25 * (1.0 + s) * (1.0 - t),
        0.25 * (1.0 + s) * (1.0 + t),
        0.25 * (1.0 - s) * (1.0 + t),
    ], axis=1).astype(np.float64)


def build_precompute(case, case_info, opt_info):
    cache = {}
    length = case_info['length']
    height = case_info['height']
    nx = case_info['nx']
    ny = case_info['ny']
    dx = length / nx
    dy = height / ny

    node_coordinate, ele_node, ele_dofs, fixeddofs, load = set_case(case, case_info)

    Ke0 = cal_q4_Ke(dx, dy, nu=0.3, plane='stress')

    nd = (nx + 1) * (ny + 1)
    ndof = 2 * nd

    fixeddofs = fixeddofs.flatten()
    all_dofs = np.arange(ndof, dtype=np.int64)
    is_fixed = np.zeros(ndof, dtype=bool)
    is_fixed[fixeddofs] = True
    freedofs = all_dofs[~is_fixed]

    cache['node_coordinate'] = node_coordinate
    cache['ele_node'] = ele_node
    cache['ele_dofs'] = ele_dofs
    cache['fixeddofs'] = fixeddofs
    cache['freedofs'] = freedofs
    cache['load'] = load
    cache['Ke0'] = Ke0

    return cache


def cal_q4_Ke(dx, dy, nu=0.3, plane='stress'):
    if plane == 'stress':
        D = (1.0 / (1.0 - nu ** 2)) * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - nu)]
        ], dtype=np.float64)
    elif plane == 'strain':
        D = (1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu))) * np.array([
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)]
        ], dtype=np.float64)
    else:
        raise ValueError("plane must be 'stress' or 'strain'")

    coords = np.array([
        [0.0, 0.0],
        [dx, 0.0],
        [dx, dy],
        [0.0, dy]
    ], dtype=np.float64)

    g = 1.0 / np.sqrt(3.0)
    gauss_points = [(-g, -g, 1.0), (g, -g, 1.0), (g, g, 1.0), (-g, g, 1.0)]

    Ke0 = np.zeros((8, 8), dtype=np.float64)

    for xi, eta, w in gauss_points:
        N = 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta)
        ], dtype=np.float64)

        dN_nat = 0.25 * np.array([
            [-(1 - eta), -(1 - xi)],
            [(1 - eta), -(1 + xi)],
            [(1 + eta), (1 + xi)],
            [-(1 + eta), (1 - xi)]
        ], dtype=np.float64)

        J = coords.T @ dN_nat
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        dN_xy = dN_nat @ invJ.T

        B = np.zeros((3, 8), dtype=np.float64)
        for i_node in range(4):
            dN_dx = dN_xy[i_node, 0]
            dN_dy = dN_xy[i_node, 1]
            B[0, 2 * i_node] = dN_dx
            B[1, 2 * i_node + 1] = dN_dy
            B[2, 2 * i_node] = dN_dy
            B[2, 2 * i_node + 1] = dN_dx

        Nm = np.zeros((2, 8), dtype=np.float64)
        for i_node in range(4):
            Nm[0, 2 * i_node] = N[i_node]
            Nm[1, 2 * i_node + 1] = N[i_node]

        Ke0 += (B.T @ D @ B) * detJ * w


    return Ke0
