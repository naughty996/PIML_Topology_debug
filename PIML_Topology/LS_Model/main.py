
import os
import random
import numpy as np
import torch

import config
from initial_ls_net import relative_l2_error, run_initial_levelset_prediction
from ls_net import LevelSetNet
from set_bc import initial_ls_func
from pre_compute import build_precompute, cache_to_torch
from utils import get_torch_dtype, write_pretrain_summary

def set_global_seed(seed: int = 2026):
    """Fix ordinary RNG sources used by optional pretraining and optimizers.

    Sparse FEM assembly, sparse solves and COO SpMV are handled by the CPU FEM
    backend.  This seed is still useful for optional initial model training and
    for Adam initialization/state creation.
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def get_path(case_name):
    if case_name == 'cantilever':
        return config.cantilever, {
            'phi': './result/cantilever/phi/',
            'iter_hist': './result/cantilever/iter_hist/',
            'root': './result/cantilever/'
        }
    raise ValueError(f'Unsupported case: {case_name}')



if __name__ == '__main__':
    seed = int(config.network_paras.get('seed', 2026))
    set_global_seed(seed)
    # Use one global seed for the network, optional initial LS fitting, and
    # the deterministic random initial SIAD modal subspace.
    config.optimize_paras['seed'] = seed

    case_idx = 'cantilever'
    config_data, save_path = get_path(case_idx)
    for path in save_path.values():
        os.makedirs(path, exist_ok=True)

    device = config.network_paras.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    dtype = get_torch_dtype(config.network_paras.get('dtype', 'float64'))
    torch.set_default_dtype(dtype)

    nx = config_data['nx']
    ny = config_data['ny']
    domain_type = config_data['domain']

    cache_np = build_precompute(case_idx, config_data, config.optimize_paras)

    phi = initial_ls_func(nx, ny, domain_type, config.optimize_paras['H_d'], config_data['ls_r']).reshape(-1, 1, order='F')
    node_coordinate = cache_np['node_coordinate']#改到gpu上

    pretrain_dir = config_data['model_path']
    os.makedirs(pretrain_dir, exist_ok=True)
    model_name = f"mp_{domain_type}_mesh{nx + 1}_{ny + 1}_net{config.network_paras['neu_hidden'][0]}" \
                 f"_{config.network_paras['neu_hidden'][1]}_r{config_data['ls_r']}"
    trained_model_path = os.path.join(pretrain_dir, f'{model_name}.pt')

    generated_this_run = False
    if not os.path.exists(trained_model_path):
        print('[InitialModel] Pretrained parameters not found. Training initial level-set network...', flush=True)
        results = run_initial_levelset_prediction(
            coords=node_coordinate,
            phi=phi,
            ndx=nx + 1,
            ndy=ny + 1,
            in_dim=config.network_paras['neu_in'],
            out_dim=config.network_paras['neu_out'],
            activation=config.network_paras['act_func'],
            tol=config.network_paras['ls_tol'],
            save_dir=pretrain_dir,
            model_name=model_name,
            hidden_dim=config.network_paras['neu_hidden'][1],
            num_hidden_layers=config.network_paras['neu_hidden'][0],
            epochs=config.network_paras['max_epoch'][0],
            lr=config.network_paras['lr'],
            device=device,
            dtype=config.network_paras['dtype'],
        )
        trained_model_path = results['best_model_path']
        generated_this_run = True
        summary_path = write_pretrain_summary(
            save_dir=pretrain_dir,
            model_name=model_name,
            case_name=case_idx,
            domain_type=domain_type,
            nx=nx,
            ny=ny,
            network_paras=config.network_paras,
            device=device,
            results=results,
        )
        print(f'[InitialModel] Pretraining summary saved to: {summary_path}', flush=True)
    else:
        print('[InitialModel] Pretrained parameters available. Loading fixed initial model.', flush=True)

    topo_model = LevelSetNet(
        in_dim=config.network_paras['neu_in'],
        out_dim=config.network_paras['neu_out'],
        hidden_dim=config.network_paras['neu_hidden'][1],
        num_hidden_layers=config.network_paras['neu_hidden'][0],
        activation=config.network_paras['act_func'],
        pretrained_path=trained_model_path,
        device=device,
    ).to(device=device, dtype=dtype)

    loaded_rel_l2, _ = relative_l2_error(topo_model, node_coordinate, phi)
    n_params = sum(p.numel() for p in topo_model.parameters() if p.requires_grad)
    print(f'[InitialModel] Loaded topology model relative L2 error = {loaded_rel_l2:.6e}', flush=True)
    print(f'[InitialModel] Trainable parameters = {n_params}', flush=True)
    #调用opt
    #phi最终结果传过来，其余用self
    #obj，vol变化图，每一步的时间（迭代历史），loss，
    #日志信息，obj，vol，avetime，time