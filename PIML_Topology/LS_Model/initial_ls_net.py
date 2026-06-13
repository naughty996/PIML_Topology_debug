import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.io import savemat
from utils import get_torch_dtype


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class LevelSetInitialNet(nn.Module):
    def __init__(
        self,
        in_dim=2,
        out_dim=1,
        hidden_dim=64,
        num_hidden_layers=3,
        activation='sine',
    ):
        super().__init__()
        if activation == 'sine':
            act = Sine()
        elif activation == 'tanh':
            act = nn.Tanh()
        else:
            raise ValueError("activation must be 'sine' or 'tanh'")
        layers = [nn.Linear(in_dim, hidden_dim), act]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def relative_l2_error(model, coords_t, phi_true_t):
    model.eval()
    ref_param = next(model.parameters())
    model_device = ref_param.device
    model_dtype = ref_param.dtype

    if not isinstance(coords_t, torch.Tensor):
        coords_t = torch.as_tensor(coords_t, dtype=model_dtype, device=model_device)
    elif coords_t.device != model_device or coords_t.dtype != model_dtype:
        coords_t = coords_t.to(device=model_device, dtype=model_dtype)

    if not isinstance(phi_true_t, torch.Tensor):
        phi_true_t = torch.as_tensor(phi_true_t, dtype=model_dtype, device=model_device)
    elif phi_true_t.device != model_device or phi_true_t.dtype != model_dtype:
        phi_true_t = phi_true_t.to(device=model_device, dtype=model_dtype)

    if phi_true_t.ndim == 1:
        phi_true_t = phi_true_t.unsqueeze(1)

    phi_pred_t = model(coords_t)
    num = torch.norm(phi_pred_t - phi_true_t, p=2)
    den = torch.norm(phi_true_t, p=2) + 1e-30
    rel_l2 = (num / den).item()
    return rel_l2, phi_pred_t.detach().cpu().numpy()


def train_model(
    model,
    coords,
    phi,
    epochs=30000,
    lr=1e-3,
    min_lr=1e-5,
    device='cpu',
    dtype='float64',
    tol=1e-3,
    min_epochs=5000,
    save_dir='./initial_ls_results',
    model_name='initial_ls_model',
):
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)
    torch_dtype = get_torch_dtype(dtype)
    model = model.to(device=device, dtype=torch_dtype)

    coords = torch.as_tensor(coords, dtype=torch_dtype, device=device)
    phi = torch.as_tensor(phi, dtype=torch_dtype, device=device)
    if phi.ndim == 1:
        phi = phi.unsqueeze(1)

    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    criterion = nn.MSELoss()

    loss_history = []
    lr_history = []
    rel_l2_history = []
    best_model_path = os.path.join(save_dir, f'{model_name}.pt')

    model.train()
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        pred = model(coords)
        loss = criterion(pred, phi)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch:6d} | Loss = {loss.item():.6e} | LR = {current_lr:.6e}')

        if (epoch + 1) > min_epochs and (epoch + 1) % 50 == 0:
            rel_l2, _ = relative_l2_error(model, coords, phi)
            rel_l2_history.append((epoch + 1, rel_l2))
            print(f'          --> Relative L2 Error = {rel_l2:.6e}')
            if rel_l2 <= tol:
                print(f'\nReached target tolerance tol={tol:.3e} at epoch={epoch + 1}.')
                break
            if convergence_check(np.asarray(rel_l2_history)[:, 1], rel_tol=1e-5):
                print(f'\nRelative L2 error stabilized at epoch={epoch + 1}.')
                break

    final_rel_l2, _ = relative_l2_error(model, coords, phi)
    torch.save(model.state_dict(), best_model_path)
    return loss_history, lr_history, rel_l2_history, best_model_path, final_rel_l2


def convergence_check(array, rel_tol=1e-4):
    num_check = 10
    if len(array) < 2 * num_check:
        return False
    mean1 = np.mean(array[-2 * num_check: -num_check])
    mean2 = np.mean(array[-num_check:])
    return (np.abs(mean1 - mean2) / (np.abs(mean2) + 1e-30)) < rel_tol


@torch.no_grad()
def predict(model, coords, device='cpu', dtype='float64'):
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    torch_dtype = get_torch_dtype(dtype)
    device = torch.device(device)
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()
    coords = torch.as_tensor(coords, dtype=torch_dtype, device=device)
    pred = model(coords)
    return pred.cpu().numpy()


def plot_loss(loss_history, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_lr(lr_history, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.plot(lr_history, lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('CosineAnnealingLR Schedule')
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_rel_l2(rel_l2_history, save_path=None):
    if len(rel_l2_history) == 0:
        return
    epochs = [x[0] for x in rel_l2_history]
    errs = [x[1] for x in rel_l2_history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, errs, marker='o', lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 Error')
    plt.title('Relative L2 Error History')
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_comparison(phi_true, phi_pred, save_path=None):
    phi_true = np.asarray(phi_true).reshape(-1)
    phi_pred = np.asarray(phi_pred).reshape(-1)
    plt.figure(figsize=(5, 5))
    plt.scatter(phi_true, phi_pred, s=8, alpha=0.6)
    vmin = min(phi_true.min(), phi_pred.min())
    vmax = max(phi_true.max(), phi_pred.max())
    plt.plot([vmin, vmax], [vmin, vmax], 'r--', lw=2)
    plt.xlabel('True phi')
    plt.ylabel('Predicted phi')
    plt.title('Prediction vs Ground Truth')
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_levelset_field(coords, phi_true, phi_pred, nx, ny, save_path=None):
    x = coords[:, 0].reshape((ny, nx), order='F')
    y = coords[:, 1].reshape((ny, nx), order='F')
    phi_true = np.asarray(phi_true).reshape((ny, nx), order='F')
    phi_pred = np.asarray(phi_pred).reshape((ny, nx), order='F')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    c1 = axes[0].contourf(x, y, phi_true, levels=50, cmap='jet')
    axes[0].contour(x, y, phi_true, levels=[0.0], colors='k', linewidths=1.5)
    axes[0].set_title('True Level Set')
    fig.colorbar(c1, ax=axes[0])
    c2 = axes[1].contourf(x, y, phi_pred, levels=50, cmap='jet')
    axes[1].contour(x, y, phi_pred, levels=[0.0], colors='k', linewidths=1.5)
    axes[1].set_title('Predicted Level Set')
    fig.colorbar(c2, ax=axes[1])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_absolute_error_field(coords, phi_true, phi_pred, nx, ny, save_path=None):
    x = coords[:, 0].reshape((ny, nx), order='F')
    y = coords[:, 1].reshape((ny, nx), order='F')
    phi_true = np.asarray(phi_true).reshape((ny, nx), order='F')
    phi_pred = np.asarray(phi_pred).reshape((ny, nx), order='F')
    err = np.abs(phi_pred - phi_true)
    plt.figure(figsize=(6, 5))
    cf = plt.contourf(x, y, err, levels=50, cmap='jet')
    plt.colorbar(cf, label='|phi_pred - phi_true|')
    plt.contour(x, y, phi_true, levels=[0.0], colors='white', linewidths=1.2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Absolute Error Field')
    plt.axis('equal')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_initial_levelset_prediction(
    coords,
    phi,
    ndx,
    ndy,
    in_dim=2,
    out_dim=1,
    activation='sine',
    tol=1e-3,
    save_dir='./initial_ls_results',
    model_name='initial_ls_model',
    hidden_dim=64,
    num_hidden_layers=3,
    epochs=30000,
    min_epochs=5000,
    lr=1e-3,
    min_lr=1e-5,
    device=None,
    dtype='float64',
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(save_dir, exist_ok=True)
    model = LevelSetInitialNet(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        activation=activation,
    )
    loss_history, lr_history, rel_l2_history, best_model_path, best_rel_l2 = train_model(
        model=model,
        coords=coords,
        phi=phi,
        epochs=epochs,
        lr=lr,
        min_lr=min_lr,
        device=device,
        dtype=dtype,
        tol=tol,
        min_epochs=min_epochs,
        save_dir=save_dir,
        model_name=model_name,
    )
    phi_pred = predict(model, coords, device=device, dtype=dtype)
    phi_pred_mat_path = os.path.join(save_dir, f'{model_name}_phi_pred.mat')
    savemat(phi_pred_mat_path, {'phi_pred': np.asarray(phi_pred)})
    plot_loss(loss_history, save_path=os.path.join(save_dir, f'{model_name}_loss.png'))
    plot_lr(lr_history, save_path=os.path.join(save_dir, f'{model_name}_lr.png'))
    plot_rel_l2(rel_l2_history, save_path=os.path.join(save_dir, f'{model_name}_rel_l2.png'))
    plot_prediction_comparison(
        phi_true=np.asarray(phi).reshape(-1),
        phi_pred=np.asarray(phi_pred).reshape(-1),
        save_path=os.path.join(save_dir, f'{model_name}_pred_vs_true.png'),
    )
    plot_levelset_field(
        coords=np.asarray(coords),
        phi_true=np.asarray(phi).reshape(-1),
        phi_pred=np.asarray(phi_pred).reshape(-1),
        nx=ndx,
        ny=ndy,
        save_path=os.path.join(save_dir, f'{model_name}_levelset.png'),
    )
    plot_absolute_error_field(
        coords=np.asarray(coords),
        phi_true=np.asarray(phi).reshape(-1),
        phi_pred=np.asarray(phi_pred).reshape(-1),
        nx=ndx,
        ny=ndy,
        save_path=os.path.join(save_dir, f'{model_name}_abs_error.png'),
    )
    return {
        'best_model_path': best_model_path,
        'phi_pred_mat_path': phi_pred_mat_path,
        'best_rel_l2': best_rel_l2,
        'phi_pred': phi_pred,
    }
