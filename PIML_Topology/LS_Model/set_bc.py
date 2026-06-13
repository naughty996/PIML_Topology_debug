import numpy as np


def set_case(case, case_info):
    length, height = case_info['length'], case_info['height']
    nx, ny = case_info['nx'], case_info['ny']
    boundary_box = [0.0, length, 0.0, height]
    load_size =case_info['load_size']

    node_coordinate = set_fem_domain(0.0, length, nx, 0.0, height, ny)
    ele_node, ele_dofs = set_ele_node_connect(nx, ny)
    fixeddofs = set_constrain(node_coordinate, boundary_box, case)
    load = set_load(node_coordinate, boundary_box, case, nstep, load_size)
    u0 = np.zeros((2 * (nx + 1) * (ny + 1), 1), dtype=np.float64)
    v0 = np.zeros((2 * (nx + 1) * (ny + 1), 1), dtype=np.float64)
    return node_coordinate, ele_node, ele_dofs, fixeddofs, load, u0, v0

def set_fem_domain(min_x, length, nx, min_y, height, ny):
    ndx = nx + 1
    ndy = ny + 1
    lin_x = np.linspace(min_x, length, ndx)
    lin_y = np.linspace(min_y, height, ndy)
    node_coordinate = np.zeros((ndx * ndy, 2), dtype=np.float64)
    col_id = 0
    for coord_x in np.nditer(lin_x):
        tb = ndy * col_id
        te = tb + ndy
        node_coordinate[tb:te, 0] = coord_x
        node_coordinate[tb:te, 1] = lin_y
        col_id += 1
    return node_coordinate


def initial_ls_func(nx, ny, domain, d, radio):
    X, Y = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
    if domain == 'single_hole':
        r = ny * radio
        hX = np.array([nx * 0.5])
        hY = np.array([ny * 0.5])
        dX = X[:, :, None] - hX[None, None, :]
        dY = Y[:, :, None] - hY[None, None, :]
        phi = np.min(np.sqrt(dX ** 2 + dY ** 2) - r, axis=2)
        return np.clip(phi, -d, d)
    if domain == 'multi_holes':
        r = ny * radio
        hX = nx * np.array([
            1 / 6, 5 / 6, 1 / 6, 5 / 6, 1 / 6, 5 / 6,
            0, 1 / 3, 2 / 3, 1, 0, 1 / 3, 2 / 3, 1, 1 / 2
        ])
        hY = ny * np.array([
            0, 0, 1 / 2, 1 / 2, 1, 1,
            1 / 4, 1 / 4, 1 / 4, 1 / 4, 3 / 4, 3 / 4, 3 / 4, 3 / 4, 1 / 2
        ])
        dX = X[:, :, None] - hX[None, None, :]
        dY = Y[:, :, None] - hY[None, None, :]
        phi = np.min(np.sqrt(dX ** 2 + dY ** 2) - r, axis=2)
        return np.clip(phi, -d, d)
    if domain == 'no_hole':
        return d * np.ones((ny + 1, nx + 1), dtype=np.float64)
    raise ValueError("domain must be 'single_hole', 'multi_holes', or 'no_hole'")


def set_load(node, bbox, case, nstep=100, amplitude=1000.0):
    """Build the time load history and its time-independent spatial pattern(s).

    The QMDM static enrichment in the SIAD-QMDM paper uses the load spatial
    distribution F_hat in F(t)=F_hat*xi(t).  Returning load_spatial_patterns
    here makes that information explicit, instead of recovering it later from
    the discrete time-load matrix.  For the current cantilever benchmark there
    is one spatial pattern: a downward force at the loaded DOF.
    """
    xmin, xmax, ymin, ymax = bbox
    eps = 0.1 * np.sqrt((xmax - xmin) * (ymax - ymin) / node.shape[0])
    theta = np.linspace(0.0, np.pi, nstep + 1)
    fx = np.zeros(nstep + 1, dtype=np.float64)
    fy = np.zeros(nstep + 1, dtype=np.float64)

    ndof = 2 * node.shape[0]

    if case == 'cantilever':
        candidates = np.where((np.abs(node[:, 0] - xmax) < eps) & (np.abs(node[:, 1] - ymax / 2) < eps))[0]
        if len(candidates) == 0:
            raise ValueError('No loaded node found for cantilever case.')
        loaded_node = int(candidates[0])
        # Time history: F(t) = F_hat * sin(theta), with F_hat carrying the
        # spatial location, direction and amplitude scale.  The sign matters;
        # the absolute scale is later removed by basis orthonormalization.
        # 改成静载荷
        fy = -amplitude * np.sin(theta)
    else:
        raise ValueError(f'Unsupported case: {case}')
    # 载荷和位移都是列向量
    load = np.zeros((ndof, 1), dtype=np.float64)
    load[2 * loaded_node, :] = fx
    load[2 * loaded_node + 1, :] = fy
    return load


def set_constrain(node, bbox, case):
    xmin, xmax, ymin, ymax = bbox
    eps = 0.1 * np.sqrt((xmax - xmin) * (ymax - ymin) / node.shape[0])
    if case != 'cantilever':
        raise ValueError(f'Unsupported case: {case}')
    left_nodes = np.where(np.abs(node[:, 0] - xmin) < eps)[0]
    dofs_x = 2 * left_nodes
    dofs_y = 2 * left_nodes + 1
    fixeddofs = np.concatenate([dofs_x, dofs_y])
    return np.sort(fixeddofs).reshape(-1, 1)


def set_ele_node_connect(nx, ny):
    ndx, ndy = nx + 1, ny + 1
    nd = ndx * ndy
    node_set = np.array(np.split(np.arange(nd), ndx)).T
    node_set_1 = node_set[0:ny, 0:nx]
    node_set_4 = node_set[1:ndy, 0:nx]
    node_set_3 = node_set[1:ndy, 1:ndx]
    node_set_2 = node_set[0:ny, 1:ndx]
    ele_nodes = np.concatenate(
        (
            node_set_1.T.flatten()[:, None],
            node_set_2.T.flatten()[:, None],
            node_set_3.T.flatten()[:, None],
            node_set_4.T.flatten()[:, None],
        ),
        axis=1,
    )
    ele_dofs = np.zeros((nx * ny, 8), dtype=np.int64)
    ele_dofs[:, 0::2] = 2 * ele_nodes
    ele_dofs[:, 1::2] = 2 * ele_nodes + 1
    return ele_nodes, ele_dofs
