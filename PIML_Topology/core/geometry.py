import os
import torch
import torch.nn as nn
import numpy as np


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class LevelSetNet(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, hidden_dim=64, num_hidden_layers=3,
                 activation='sine', pretrained_path=None, device='cpu', strict=True):
        super().__init__()

        act = Sine() if activation == 'sine' else nn.Tanh()
        layers = [nn.Linear(in_dim, hidden_dim), act]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained_weights(pretrained_path, device, strict)
        else:
            self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)

    def load_pretrained_weights(self, pretrained_path, device='cpu', strict=True):
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint
        else:
            raise ValueError("Unsupported checkpoint format.")
        self.load_state_dict(state_dict, strict=strict)
        print(f"[Geometry] 隐式图纸权重加载成功: {pretrained_path}")
        return self


def get_node_coords(nx, ny, device='cpu'):
    x_nodes = torch.linspace(-1, 1, nx + 1, dtype=torch.float64, device=device)
    y_nodes = torch.linspace(-1, 1, ny + 1, dtype=torch.float64, device=device)
    GridX, GridY = torch.meshgrid(x_nodes, y_nodes, indexing='ij')
    return torch.stack([GridX.flatten(), GridY.flatten()], dim=1)


@torch.no_grad()
def relative_l2_error(model, coords_t, phi_true_t):
    model.eval()
    phi_pred_t = model(coords_t)
    num = torch.norm(phi_pred_t - phi_true_t, p=2)
    den = torch.norm(phi_true_t, p=2) + 1e-30
    model.train()
    return (num / den).item()


def convergence_check(array, rel_tol=1e-5):
    num_check = 10
    if len(array) < 2 * num_check:
        return False
    mean1 = np.mean(array[-2 * num_check: -num_check])
    mean2 = np.mean(array[-num_check:])
    return (np.abs(mean1 - mean2) / (np.abs(mean2) + 1e-30)) < rel_tol


def generate_initial_weights(config_dict):
    save_path = config_dict['data']['init_ls_weights']
    if os.path.exists(save_path):
        return

    print("\n[Geometry] 正在执行多孔形态基准刻录...")
    nx, ny = config_dict['physics']['nx'], config_dict['physics']['ny']
    device = config_dict['optimization']['device']
    lr = config_dict['ls_network']['lr']
    radio = config_dict['ls_network']['ls_r']
    H_d = config_dict['ls_network']['H_d']
    tol = config_dict['ls_network']['ls_tol']
    epochs = config_dict['ls_network']['max_epoch']
    min_lr, min_epochs = 1e-5, 5000

    coords = get_node_coords(nx, ny, device)
    X_unscaled = (coords[:, 0] + 1.0) / 2.0 * nx
    Y_unscaled = (coords[:, 1] + 1.0) / 2.0 * ny

    r = ny * radio
    hX = nx * torch.tensor([1 / 6, 5 / 6, 1 / 6, 5 / 6, 1 / 6, 5 / 6, 0, 1 / 3, 2 / 3, 1, 0, 1 / 3, 2 / 3, 1, 1 / 2],
                           dtype=torch.float64, device=device)
    hY = ny * torch.tensor([0, 0, 1 / 2, 1 / 2, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 3 / 4, 3 / 4, 3 / 4, 3 / 4, 1 / 2],
                           dtype=torch.float64, device=device)

    dX = X_unscaled.unsqueeze(1) - hX.unsqueeze(0)
    dY = Y_unscaled.unsqueeze(1) - hY.unsqueeze(0)
    dist = torch.sqrt(dX ** 2 + dY ** 2)
    phi_target, _ = torch.min(dist - r, dim=1)
    phi_target = torch.clamp(phi_target, -H_d, H_d).unsqueeze(1)

    model = LevelSetNet(
        in_dim=config_dict['ls_network']['neu_in'],
        out_dim=config_dict['ls_network']['neu_out'],
        hidden_dim=config_dict['ls_network']['neu_hidden'][1],
        num_hidden_layers=config_dict['ls_network']['neu_hidden'][0],
        activation=config_dict['ls_network']['act_func'],
        device=device
    ).to(device=device, dtype=torch.float64)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    criterion = nn.MSELoss()

    model.train()
    rel_l2_history = []

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = model(coords)
        loss = criterion(pred, phi_target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:6d} | MSE Loss: {loss.item():.6e} | LR: {current_lr:.6e}")

        if (epoch + 1) > min_epochs and (epoch + 1) % 50 == 0:
            rel_l2 = relative_l2_error(model, coords, phi_target)
            rel_l2_history.append(rel_l2)

            if rel_l2 <= tol:
                print(f"  --> 达到目标容差 tol={tol:.3e}，提前终止于 epoch={epoch + 1}.")
                break
            if convergence_check(rel_l2_history, rel_tol=1e-5):
                print(f"  --> 相对 L2 误差趋于稳定，提前终止于 epoch={epoch + 1}.")
                break

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[Geometry] 刻录完成，初始化权重落盘至: {save_path}\n")