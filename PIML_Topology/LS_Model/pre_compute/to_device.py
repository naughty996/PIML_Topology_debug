import torch


PASSTHROUGH_KEYS = {
    'linear_solver_backend', 'linear_solver_method', 'reuse_factorization_type',
    'superlu_permc_spec', 'rom_static_load_basis_mode', 'rom_energy_residual_basis_types',
}

LONG_KEYS = {
    'ele_node', 'ele_dofs', 'fixeddofs', 'freedofs', 'i', 'j', 'ne', 'nd', 'ndof',
    'ndof_free', 'n_step', 'full_row', 'full_col', 'full_inverse', 'n_full_nnz',
    'free_row', 'free_col', 'free_inverse', 'free_entry_ids', 'n_free_nnz',
    'cpu_spmv_backend_is_csr', 'cpu_csr_use_numba', 'hht_use_free_rhs_spmv', 'energy_use_free_spmv'
}


def cache_to_torch(cache, device='cpu', dtype=torch.float64):
    out = {}
    device = torch.device(device)
    for k, v in cache.items():
        if k in PASSTHROUGH_KEYS or isinstance(v, str):
            out[k] = v
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device)
            continue
        if k in LONG_KEYS:
            out[k] = torch.as_tensor(v, device=device, dtype=torch.long)
        else:
            out[k] = torch.as_tensor(v, device=device, dtype=dtype)
    return out
