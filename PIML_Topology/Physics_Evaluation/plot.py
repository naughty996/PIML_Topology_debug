# =========================================================================
# 功能: 生成宏观位移场云图、微观刚度误差矩阵等
# =========================================================================
import numpy as np
import matplotlib.pyplot as plt

def plot_multiscale_validation(res_dict, use_samples, save_path):
    """根据求解器返回的结果字典，绘制物理合规性对比三联图"""
    U_true_np = res_dict['U_true'].cpu().numpy()
    U_pred_np = res_dict['U_pred'].cpu().numpy()
    nelx_c, nely_c = res_dict['nelx_c'], res_dict['nely_c']

    # 计算位移模量并重塑网格以匹配物理空间
    disp_true_mag = np.sqrt(U_true_np[0::2] ** 2 + U_true_np[1::2] ** 2).reshape((nelx_c + 1, nely_c + 1)).T
    disp_pred_mag = np.sqrt(U_pred_np[0::2] ** 2 + U_pred_np[1::2] ** 2).reshape((nelx_c + 1, nely_c + 1)).T

    # 计算宏观单元级刚度矩阵相对误差
    sK_true_np = res_dict['sK_global_true'].cpu().numpy().reshape(use_samples, 64)
    sK_pred_np = res_dict['sK_global_pred'].cpu().numpy().reshape(use_samples, 64)
    element_err = np.linalg.norm(sK_pred_np - sK_true_np, axis=1) / (np.linalg.norm(sK_true_np, axis=1) + 1e-9)
    Grid_Err = element_err.reshape((nelx_c, nely_c)).T

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f"Darcy Pressure Load Validation ({nelx_c}x{nely_c} Macro Elements)", fontsize=20, fontweight='bold')

    ax1 = axes[0]
    im1 = ax1.imshow(disp_true_mag, cmap='jet', aspect='equal', origin='upper')
    ax1.set_title(f"True Displacement Field under Darcy Pressure (C_EMs: {res_dict['C_EMs']:.4f})", fontsize=14)
    fig.colorbar(im1, ax=ax1, fraction=0.02, pad=0.02, label="Displacement")

    ax2 = axes[1]
    im2 = ax2.imshow(disp_pred_mag, cmap='jet', aspect='equal', origin='upper', vmin=np.min(disp_true_mag), vmax=np.max(disp_true_mag))
    ax2.set_title(f"PIML Predicted Displacement Field (C_ANN: {res_dict['C_ANN']:.4f})", fontsize=14)
    fig.colorbar(im2, ax=ax2, fraction=0.02, pad=0.02, label="Displacement")

    ax3 = axes[2]
    im3 = ax3.imshow(Grid_Err, cmap='Reds', aspect='equal', origin='upper')
    ax3.set_title("Element-wise Stiffness Matrix Relative Error", fontsize=14)
    fig.colorbar(im3, ax=ax3, fraction=0.02, pad=0.02, label="Error")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)