import torch


PASSTHROUGH_KEYS = {
    'linear_solver_backend', 'linear_solver_method', 'reuse_factorization_type',
    'superlu_permc_spec', 'rom_static_load_basis_mode', 'rom_energy_residual_basis_types',
}

LONG_KEYS = {
    'ele_node', 'ele_dofs', 'fixeddofs', 'freedofs', 'Ke0', 'node_coordinate',
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
