# =========================================================================
# 功能: 纯数值层面的误差评估、日志曲线绘制及物理场抽样可视化组件
# =========================================================================
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat


def get_l2_relative_error(pred, target):
    """
    计算批次预测张量相对于真值的平均 L2 相对误差。
    数学公式: Error = mean( ||pred - target||_2 / ||target||_2 )
    注意：分母添加 1e-9 防止全零向量导致的除零异常。
    """
    error = torch.norm(pred - target, p=2, dim=1)
    norm = torch.norm(target, p=2, dim=1)
    return torch.mean(error / (norm + 1e-9)).item()


def plot_training_log(history_train_loss, history_test_l2, history_lr, fig_save):
    """
    绘制模型训练过程的动态对数日志图，包含：
    1. 训练集相对损失 (主Y轴，蓝色)
    2. 测试集全场 L2 相对误差 (副Y轴，红色)
    3. 学习率衰减曲线 (底部子图，绿色)
    """
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    # 训练损失主轴
    color = 'tab:blue'
    ax1.set_ylabel('Train Rel Loss (192d)', color=color)
    ax1.plot(range(len(history_train_loss)), history_train_loss, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_title('PIML Training Dynamics', fontsize=14)

    # 测试集误差副轴
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Test Relative L2 Error (576d)', color=color)
    ax2.plot(range(len(history_test_l2)), history_test_l2, color=color, linewidth=2, linestyle='-', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color)

    # 学习率衰减曲线
    color = 'tab:green'
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Learning Rate', color=color)
    ax3.plot(range(len(history_lr)), history_lr, color=color, linewidth=2)
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_yscale('log')
    ax3.grid(True, linestyle=':', alpha=0.6)

    fig.tight_layout()
    plt.savefig(fig_save, dpi=300)
    plt.close(fig)


def plot_contours(arr, fig_save):
    """
    将展平的一维张量重塑为二维空间矩阵，并绘制有限元形函数的云图。
    处理逻辑严格匹配 MATLAB 的 order='F' (Fortran 列优先排布)。
    """
    nx, ny = 5, 5
    x = np.arange(nx + 1)
    y = np.arange(ny + 1)
    X, Y = np.meshgrid(x, y)

    for i in range(8):
        N = arr[:, i]
        odd = N[0::2]
        even = N[1::2]

        Z_left = odd.reshape(ny + 1, nx + 1, order='F')
        Z_right = even.reshape(ny + 1, nx + 1, order='F')

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))

        cf_left = ax_left.contourf(X, Y, Z_left, cmap='viridis')
        ax_left.set_aspect('equal')
        plt.colorbar(cf_left, ax=ax_left)

        cf_right = ax_right.contourf(X, Y, Z_right, cmap='viridis')
        ax_right.set_aspect('equal')
        plt.colorbar(cf_right, ax=ax_right)

        plt.savefig(f"{fig_save}_{i + 1}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def evaluate_and_plot_sample(model, assembler, X_test_t, Y_576_test_t, sample_num=0, save_dir='./result_fig'):
    """
    高级接口：抽取指定测试样本，执行前向推断，计算残差，并无缝导出 MAT 数据与图像。
    模块解耦核心：将数据重塑与 I/O 脏活彻底从 train.py 中剥离。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. 提取真实标签并转换为 NumPy
    true_576 = Y_576_test_t[sample_num].cpu().numpy()

    # 2. 网络前向推断与物理场还原
    model.eval()
    with torch.no_grad():
        X_sample = X_test_t[sample_num].unsqueeze(0)
        # 模型输出 192维 -> 组装器还原为 576维
        pred_576_sample = assembler.assemble_576(model(X_sample)).squeeze(0).cpu().numpy()

    # 3. 维度重塑为有限元标准矩阵 (72个节点自由度 x 8个基底)
    SN_true = true_576.reshape((-1, 8), order='F')
    SN_pre = pred_576_sample.reshape((-1, 8), order='F')
    SN_error = SN_pre - SN_true

    # 4. 序列化为 MATLAB 数据格式供后续排查
    savemat(f'{save_dir}/SN_data.mat', {'sn_t': SN_true, 'sn_p': SN_pre, 'sn_e': SN_error})

    # 5. 调用底层画图函数输出可视化云图
    plot_contours(SN_pre, f'{save_dir}/sn_p')
    plot_contours(SN_true, f'{save_dir}/sn_t')
    plot_contours(SN_error, f'{save_dir}/error')

    print(f"[*] 单样本(ID:{sample_num}) 评估完成，图像与 MAT 数据已生成至: {save_dir}")