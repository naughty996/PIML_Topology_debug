import os
import torch


def get_torch_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {'float64', 'double', 'torch.float64'}:
        return torch.float64
    if name in {'float32', 'float', 'torch.float32'}:
        return torch.float32
    raise ValueError(f'Unsupported dtype: {name}')


def write_pretrain_summary(
    save_dir,
    model_name,
    case_name,
    domain_type,
    nx,
    ny,
    network_paras,
    device,
    results,
):
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, f"{model_name}_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('Pretraining Summary\n')
        f.write('===================\n')
        f.write(f'Case name            : {case_name}\n')
        f.write(f'Initial topology     : {domain_type}\n')
        f.write(f'Grid size            : {nx + 1} x {ny + 1}\n')
        f.write(f'Input dimension      : {network_paras["neu_in"]}\n')
        f.write(f'Output dimension     : {network_paras["neu_out"]}\n')
        f.write(f'Activation           : {network_paras["act_func"]}\n')
        f.write(f'Hidden layers        : {network_paras["neu_hidden"][0]}\n')
        f.write(f'Hidden dimension     : {network_paras["neu_hidden"][1]}\n')
        f.write(f'Epochs               : {network_paras["max_epoch"][0]}\n')
        f.write(f'Learning rate        : {network_paras["lr"]}\n')
        f.write(f'Dtype                : {network_paras["dtype"]}\n')
        f.write(f'Device               : {device}\n')
        f.write(f'Best relative L2     : {results["best_rel_l2"]:.6e}\n')
        f.write(f'Model path           : {results["best_model_path"]}\n')
        f.write(f'Phi prediction (.mat): {results["phi_pred_mat_path"]}\n')
    return summary_path
