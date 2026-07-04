import os
import matplotlib.pyplot as plt
import numpy as np

from core.config import config
from core.geometry import generate_initial_weights
from opt import run_optimization
from opt_alm import run_optimization


def plot_convergence_history(history, save_dir='./results_topo'):
    os.makedirs(save_dir, exist_ok=True)
    iters = np.arange(1, len(history['obj']) + 1)

    fig, axs = plt.subplots(4, 1, figsize=(10, 14))

    axs[0].plot(iters, history['obj'], 'b-', linewidth=2)
    axs[0].set_title('Darcy Compliance Evolution', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('Compliance (Obj)')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    vol_target = config['optimization']['vf_target']
    axs[1].plot(iters, history['vol'], 'g-', linewidth=2, label='Current Vol')
    axs[1].axhline(y=vol_target, color='r', linestyle='--', linewidth=2, label='Target Vol')
    axs[1].set_title('Volume Fraction Constraint', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('Volume Fraction')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    axs[2].plot(iters, history['loss'], 'k-', linewidth=2)
    axs[2].set_title('Total Loss (Obj + Adaptive Penalty)', fontsize=12, fontweight='bold')
    axs[2].set_ylabel('Total Loss')
    axs[2].grid(True, linestyle='--', alpha=0.6)

    avg_time = np.mean(history['time'])
    axs[3].plot(iters, history['time'], 'm-', linewidth=1.5, alpha=0.8)
    axs[3].axhline(y=avg_time, color='r', linestyle=':', linewidth=2, label=f'Avg Time: {avg_time:.3f} s')
    axs[3].set_title('Computational Time per Step', fontsize=12, fontweight='bold')
    axs[3].set_xlabel('Iteration Step')
    axs[3].set_ylabel('Time (Seconds)')
    axs[3].legend(loc='upper right')
    axs[3].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    report_path = os.path.join(save_dir, 'convergence_report.png')
    plt.savefig(report_path, dpi=300)
    plt.close()

    print(f"\n[*] 演化历史图表已生成: {report_path}")
    print(f"[*] 平均单步计算耗时: {avg_time:.4f} s")


def main():
    print("=" * 80)
    print(">>> 启动 PIML-Darcy 拓扑优化流控中心 <<<")
    print("=" * 80)

    generate_initial_weights(config)
    history_data = run_optimization()

    if len(history_data['obj']) > 0:
        plot_convergence_history(history_data)


if __name__ == '__main__':
    main()