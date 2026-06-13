import numpy as np
from set_bc import set_case

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
    thickness = case_info['thickness']
    nx = case_info['nx']
    ny = case_info['ny']
    E0 = case_info['E']
    nu = case_info['nu']
    rho = case_info['rho']
    dx = length / nx
    dy = height / ny

    case_data = set_case(case, case_info)
    if len(case_data) == 8:
        node_coordinate, ele_node, ele_dofs, fixeddofs, load, u0, v0, load_spatial_patterns = case_data
    else:
        node_coordinate, ele_node, ele_dofs, fixeddofs, load, u0, v0 = case_data
        # Backward-compatible fallback for old set_bc.py files.  It is better
        # to provide the true time-independent load spatial pattern directly
        # from set_bc.py.
        col_norm = np.linalg.norm(load, axis=0)
        nz = np.flatnonzero(col_norm > 0)
        load_spatial_patterns = load[:, nz[:1]] if nz.size else np.zeros((load.shape[0], 0), dtype=np.float64)

    Ke0, Me0 = cal_q4_Ke_Me(dx, dy, nu, plane='stress')
    Ke = Ke0.reshape(-1, order='F') * thickness * E0
    Me = Me0.reshape(-1, order='F') * thickness * rho

    nd = (nx + 1) * (ny + 1)
    ndof = 2 * nd
    ne = nx * ny

    fixeddofs = fixeddofs.flatten()
    all_dofs = np.arange(ndof, dtype=np.int64)
    is_fixed = np.zeros(ndof, dtype=bool)
    is_fixed[fixeddofs] = True
    freedofs = all_dofs[~is_fixed]

    # element contributions in column-major local matrix order
    i = np.tile(ele_dofs, (1, 8)).reshape(-1)
    j = np.repeat(ele_dofs, 8, axis=1).reshape(-1)

    # global unique sparsity pattern for K / M / C
    full_row, full_col, full_inverse = _compute_unique_pattern(i, j)

    # free-free unique sparsity pattern for Kff / Mff / Cff / M1ff
    free_map = -np.ones(ndof, dtype=np.int64)
    free_map[freedofs] = np.arange(freedofs.size, dtype=np.int64)
    rows_local = free_map[i]
    cols_local = free_map[j]
    free_mask = (rows_local >= 0) & (cols_local >= 0)
    free_entry_ids = np.flatnonzero(free_mask).astype(np.int64)
    free_row, free_col, free_inverse = _compute_unique_pattern(rows_local[free_mask], cols_local[free_mask])

    cache['node_coordinate'] = node_coordinate
    cache['ele_node'] = ele_node
    cache['ele_dofs'] = ele_dofs
    cache['fixeddofs'] = fixeddofs
    cache['freedofs'] = freedofs
    cache['load'] = load
    cache['load_spatial_patterns'] = load_spatial_patterns
    cache['u0'] = u0
    cache['v0'] = v0
    cache['i'] = i
    cache['j'] = j
    cache['Ke'] = Ke
    cache['Me'] = Me
    cache['ne'] = ne
    cache['nd'] = nd
    cache['ndof'] = ndof
    cache['ndof_free'] = freedofs.size
    cache['ar'] = np.asarray(case_info['Rayleigh'], dtype=np.float64)
    cache['ele_area'] = np.full((ne,), dx * dy, dtype=np.float64)
    cache['t_max'] = case_info['t_max']
    cache['n_step'] = case_info['n_step']
    cache['alpha'] = case_info['alpha']
    cache['beta'] = case_info['beta']
    cache['gamma'] = case_info['gamma']
    cache['H_alpha'] = opt_info['H_alpha']
    cache['H_beta'] = opt_info['H_beta']
    cache['H_d'] = opt_info['H_d']
    cache['density_sample_n'] = opt_info.get('density_sample_n', 3)
    cache['density_sample_include_boundary'] = int(bool(opt_info.get('density_sample_include_boundary', False)))
    cache['density_shape_matrix'] = _regular_q4_density_shape_matrix(
        cache['density_sample_n'],
        bool(cache['density_sample_include_boundary']),
    )
    cache['cpu_assembly'] = int(bool(opt_info.get('cpu_assembly', True)))
    cache['cpu_spmv_backend_is_csr'] = int(str(opt_info.get('cpu_spmv_backend', 'csr')).lower() == 'csr')
    cache['cpu_csr_use_numba'] = int(bool(opt_info.get('cpu_csr_use_numba', True)))
    cache['hht_use_free_rhs_spmv'] = int(bool(opt_info.get('hht_use_free_rhs_spmv', True)))
    cache['energy_use_free_spmv'] = int(bool(opt_info.get('energy_use_free_spmv', True)))

    # SIAD-QMDM ROM options copied into the FEM cache because fem.pre_compute()
    # receives cache only. Keep scalar/string values lightweight and explicit.
    for _key in [
        'rom_modal_count',
        'rom_update_interval',
        'rom_initial_siad_steps',
        'rom_include_static_load_basis',
        'rom_static_load_basis_count',
        'rom_static_load_basis_mode',
        'rom_include_output_basis',
        'rom_output_basis_max',
        'rom_energy_residual_basis_enable',
        'rom_energy_residual_basis_types',
        'rom_energy_residual_max_residual_count',
        'rom_energy_residual_max_energy_error_count',
        'rom_energy_error_candidate_count',
        # Backward-compatible old name. If present, fem_dynamic.py uses it as
        # fallback for rom_energy_error_candidate_count.
        'rom_energy_residual_candidate_count',
        'rom_energy_residual_unique_timesteps',
        'rom_energy_residual_trigger_tol',
        'rom_energy_residual_update_interval',
        'rom_energy_residual_skip_initial',
        'rom_mass_orth_tol',
        'rom_symmetrize_reduced_matrices',
        'rom_use_reduced_lu_factor',
        'rom_cache_projected_loads',
        'rom_zero_initial_state_tol',
        'rom_verbose',
        'linear_solver_backend',
        'linear_solver_method',
        'linear_solver_atol',
        'linear_solver_maxiter',
        'reuse_factorization_type',
        'superlu_permc_spec',
        'seed',
    ]:
        if _key in opt_info:
            cache[_key] = opt_info[_key]

    # new precomputed sparse patterns
    cache['full_row'] = full_row
    cache['full_col'] = full_col
    cache['full_inverse'] = full_inverse
    cache['n_full_nnz'] = np.array([full_row.size], dtype=np.int64)

    cache['free_row'] = free_row
    cache['free_col'] = free_col
    cache['free_inverse'] = free_inverse
    cache['free_entry_ids'] = free_entry_ids
    cache['n_free_nnz'] = np.array([free_row.size], dtype=np.int64)
    return cache


def cal_q4_Ke_Me(dx, dy, nu=0.3, plane='stress'):
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
    Me0 = np.zeros((8, 8), dtype=np.float64)

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
        Me0 += (Nm.T @ Nm) * detJ * w

    return Ke0, Me0
