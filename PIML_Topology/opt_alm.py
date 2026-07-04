import os
import time
import csv
import torch
import torch.optim as optim
import numpy as np
import scipy.io as sio
import sys
from pathlib import Path

from core.config import config
from core.geometry import LevelSetNet, get_node_coords
from core.fem_darcy_solver import solve_darcy_fsi

sys.path.append(str(Path(__file__).resolve().parent / "Model"))
from network import PIMLNet, FieldAssembler


def phi_to_rho_subgrid_area(phi_nodes, nx, ny, device):
    idx_x = torch.arange(nx, device=device)
    idx_y = torch.arange(ny, device=device)
    I, J = torch.meshgrid(idx_x, idx_y, indexing='ij')

    n1 = I * (ny + 1) + J
    n2 = (I + 1) * (ny + 1) + J
    n3 = (I + 1) * (ny + 1) + (J + 1)
    n4 = I * (ny + 1) + (J + 1)

    phi_1 = phi_nodes[n1.flatten()].squeeze()
    phi_2 = phi_nodes[n2.flatten()].squeeze()
    phi_3 = phi_nodes[n3.flatten()].squeeze()
    phi_4 = phi_nodes[n4.flatten()].squeeze()

    s = torch.linspace(-1, 1, 21, device=device, dtype=torch.float64)
    t = torch.linspace(-1, 1, 21, device=device, dtype=torch.float64)
    S, T = torch.meshgrid(s, t, indexing='ij')
    S, T = S.flatten(), T.flatten()

    N1 = (1 - S) * (1 - T) / 4.0
    N2 = (1 + S) * (1 - T) / 4.0
    N3 = (1 + S) * (1 + T) / 4.0
    N4 = (1 - S) * (1 + T) / 4.0

    tmpPhi = (phi_1.unsqueeze(1) * N1 + phi_2.unsqueeze(1) * N2 +
              phi_3.unsqueeze(1) * N3 + phi_4.unsqueeze(1) * N4)

    subgrid_solid = torch.sigmoid(100.0 * tmpPhi)
    eleVol = subgrid_solid.mean(dim=1)
    return eleVol


def apply_darcy_bc_mask(raw_density, nx, ny):
    rho_grid = raw_density.view(nx, ny).clone()
    rho_grid[0, :] = 1.0

    inlet_s, inlet_e = nx // 2 - 2, nx // 2 + 2
    rho_grid[inlet_s:inlet_e, -1] = 0.0

    rho_flat = torch.clamp(rho_grid.flatten(), min=1e-4, max=1.0)
    return rho_flat.unsqueeze(1).repeat(1, 25)


def run_optimization():
    device = torch.device(config['optimization']['device'] if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64 if config['optimization']['dtype'] == 'float64' else torch.float32
    print(f"[*] 启动 PIML-Darcy 拓扑优化流控中心 | 算力节点: {device}")

    print("[*] 正在挂载 PIMLNet 预训练数字大脑...")
    piml_net = PIMLNet().to(device=device, dtype=dtype)
    piml_net.load_state_dict(torch.load(config['data']['piml_model_path'], map_location=device, weights_only=True))
    piml_net.eval()
    for param in piml_net.parameters():
        param.requires_grad = False
    assembler = FieldAssembler(config['data']['labels_path'], device)

    nx, ny = config['physics']['nx'], config['physics']['ny']
    print(f"[*] 解析宏观设计域网格: {nx} x {ny}")
    node_inputs = get_node_coords(nx, ny, device=device)

    ls_net = LevelSetNet(
        in_dim=config['ls_network']['neu_in'],
        out_dim=config['ls_network']['neu_out'],
        hidden_dim=config['ls_network']['neu_hidden'][1],
        num_hidden_layers=config['ls_network']['neu_hidden'][0],
        activation=config['ls_network']['act_func'],
        pretrained_path=config['data']['init_ls_weights'],
        device=device
    ).to(device=device, dtype=dtype)
    ls_net.train()

    lr = config['optimization']['opt_lr']
    optimizer = optim.Adam(ls_net.parameters(), lr=lr)

    max_iter = config['optimization']['max_iter']
    vol_target = config['optimization']['vf_target']
    N_conv = config['optimization']['N_conv']
    conv_tol = config['optimization']['con_tol']



    os.makedirs('./results_topo', exist_ok=True)
    csv_file = './results_topo/optimization_history.csv'
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iter', 'Compliance', 'Volume', 'Total_Loss', 'Penalty_Weight', 'Time_Step_s'])

    history = {'obj': [], 'vol': [], 'loss': [], 'time': []}

    # === 新增：在内存中统一收集所有矩阵的空列表 ===
    phi_history_all = []
    rho_history_all = []

    print("\n" + "=" * 85)
    print(f"{'Iter':>5} | {'Compliance':>12} | {'Volume':>8} | {'Loss':>10} | {'Weight':>8} | {'Time(s)':>8}")
    print("=" * 85)

    c_initial = None  # 用于记录初始合规度基准

    lam = 0.0  # 拉格朗日乘子 (Lambda)
    gamma = 10.0  # 惩罚系数 (Gamma)
    gamma_max = 500.0  # 惩罚上限
    delta_gamma = 2.0  # 增长步长
    nR = 100

    for step in range(1, max_iter + 1):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        raw_phi_nodes = ls_net(node_inputs)
        raw_density = phi_to_rho_subgrid_area(raw_phi_nodes, nx, ny, device)
        rho_field = apply_darcy_bc_mask(raw_density, nx, ny)
        current_vol = rho_field.mean()

        pred_shape_features = piml_net(rho_field)
        assembled_global_shape = assembler.assemble_576(pred_shape_features)

        results = solve_darcy_fsi(
            eleVol_batch=rho_field,
            true_vec_batch=None,
            pred_vec_batch=assembled_global_shape,
            config=config,
            device=device
        )
        compliance = results['C_ANN']

        # === 新增：将合规度归一化至 1.0 量级 ===
        if c_initial is None:
            c_initial = compliance.item() if hasattr(compliance, 'item') else float(compliance)
        compliance_normalized = compliance / c_initial

        # 1. 计算约束违规量 (G)
        vol_constraint = current_vol - vol_target

        # 2. 计算 ALM 损失函数
        # Loss = Compliance_normalized + lam * G + 0.5 * gamma * G^2
        vol_penalty = lam * vol_constraint + 0.5 * gamma * (vol_constraint ** 2)
        loss_total = compliance_normalized + vol_penalty

        loss_total.backward()
        optimizer.step()

        # 3. 自适应更新 Lambda 和 Gamma (ALM 核心)
        if step % 20 == 0:  # 每 20 步更新一次乘子，保持稳定性
            if step <= nR:
                lam += gamma * vol_constraint.item()
            else:
                lam += gamma * vol_constraint.item()
                gamma = min(gamma + delta_gamma, gamma_max)

        t_iter = time.time() - t0
        history['obj'].append(compliance.item())
        history['vol'].append(current_vol.item())
        history['loss'].append(loss_total.item())
        history['time'].append(t_iter)

        print(f"{step:05d} | {compliance.item():12.4f} | {current_vol.item():8.4f} | {loss_total.item():10.4f} | Lam:{lam:8.2f} Gam:{gamma:5.1f} | {t_iter:8.3f}")
        with open(csv_file, mode='a', newline='') as f:
            # 记录 Lam 和 Gamma，确保每一行都有迹可循
            csv.writer(f).writerow([step, compliance.item(), current_vol.item(), loss_total.item(), lam, gamma, t_iter])
        # === 修改：不再单独落盘，追加至内存列表 ===
        save_interval = config['optimization']['save_interval']
        if save_interval > 0 and (step % save_interval == 0 or step == max_iter):
            phi_np = raw_phi_nodes.view(nx + 1, ny + 1).detach().cpu().numpy()
            rho_np = raw_density.view(nx, ny).detach().cpu().numpy()

            phi_history_all.append(phi_np)
            rho_history_all.append(rho_np)

        if step % 100 == 0 or step == max_iter:
            torch.save(ls_net.state_dict(), f"./results_topo/ls_net_iter_{step}.pth")

        # ---------------- 7. N 步滚动稳态收敛检查 ----------------
        if step >= N_conv:
            recent_objs = np.array(history['obj'][-N_conv:])
            c_base = recent_objs[0]
            numerator = np.abs(np.sum(recent_objs - c_base))
            denominator = np.sum(recent_objs) + 1e-30
            obj_error = numerator / denominator
            vol_err = abs(vol_constraint.item())

            if obj_error <= conv_tol and vol_err <= conv_tol:
                print("\n" + "*" * 55)
                print(f"[*] 满足高级稳态收敛条件，提前终止于第 {step} 步.")
                print(f"[*] 历史积分相对波动 (Error): {obj_error:.5e} | 体积误差: {vol_err:.5e}")
                print("*" * 55)
                break

    # === 新增：循环结束后，将所有矩阵堆叠为 3D 结构并一次性保存 ===
    if len(phi_history_all) > 0:
        phi_3d = np.stack(phi_history_all, axis=-1)
        rho_3d = np.stack(rho_history_all, axis=-1)

        sio.savemat("./results_topo/phi_history_all.mat", {"phi_all": phi_3d})
        np.save("./results_topo/rho_history_all.npy", rho_3d)
        print(f"\n[*] 矩阵合并完成：共收集 {phi_3d.shape[2]} 帧场数据，已保存为单文件 phi_history_all.mat")

    print("[*] 拓扑优化演化结束，所有数据均已落盘至 ./results_topo/")
    return history