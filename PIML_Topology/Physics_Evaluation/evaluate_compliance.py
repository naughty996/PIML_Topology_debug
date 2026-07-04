# =========================================================================
# 功能: 调用主程序
# =========================================================================
import sys
import os
from config import config
import h5py
import numpy as np
import torch
import time
from pathlib import Path
from sklearn.model_selection import train_test_split

# [关键操作] 动态将 01_Surrogate_Model 跨文件夹挂载到系统路径，解决导入痛点
sys.path.append(str(Path(__file__).resolve().parent.parent / "Model"))
from network import PIMLNet, FieldAssembler

# 引入本地解耦的物理验证组件
from fem_darcy_solver import solve_darcy_fsi
from plot import plot_multiscale_validation


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 启动端到端设计相关压力物理合规考核 | 算力节点: {device}")

    # ================= 1. 读取全局物理/工程配置 =================
    data_size = config['data']['data_size']
    use_samples = config['data']['use_samples']

    data_size = config['data']['data_size']
    use_samples = config['data']['use_samples']

    # ================= 2. 严格按同种子切分出纯净测试集 =================
    print(f"[*] 正在解析数据集并抽取纯净测试单元 (Data: {data_size})...")
    with h5py.File(config['data']['dataset_path'], 'r') as f:
        X_all = np.array(f['/Dataset_Density'])
        if X_all.shape[0] != data_size: X_all = X_all.T

    with h5py.File(config['data']['labels_path'], 'r') as f:
        Y_all = np.array(f['/Labels_576'])
        if Y_all.shape[0] != data_size: Y_all = Y_all.T

    # 严格保持与训练阶段相同的随机种子 (2026)，严防数据泄露 (Data Leakage)
    indices = np.arange(data_size)
    _, test_idx = train_test_split(indices, test_size=0.1, random_state=config['data']['random_seed'])

    # 截取所需构建悬臂梁的微观单元数量
    use_samples = min(use_samples, len(test_idx))
    test_idx_subset = test_idx[:use_samples]

    eleVol_batch = X_all[test_idx_subset]  # X: (1000, 25)
    true_vec_batch = torch.tensor(Y_all[test_idx_subset], dtype=torch.float32).to(device)

    print(f'>>> 锁定 {use_samples} 个微观单元，即将组装 {use_samples // 20}x20 宏观悬臂梁')

    # ================= 3. PIML 代理模型挂载与显存内推断 =================
    print(f"[*] 加载神经代理权重文件: {config['data']['model_path']}")
    model = PIMLNet().to(device)
    model.load_state_dict(torch.load(config['data']['model_path'], map_location=device, weights_only=True))
    model.eval()
    assembler = FieldAssembler(config['data']['labels_path'], device)

    X_test_t = torch.tensor(eleVol_batch, dtype=torch.float32).to(device)

    t0 = time.time()
    with torch.no_grad():
        # 【端到端飞跃】直接在显存内：预测192维 -> 组装器推导还原576维
        pred_vec_batch = assembler.assemble_576(model(X_test_t))
    print(f"[*] ✅ GPU 原生物理场组装推断完成，耗时: {time.time() - t0:.4f} s")

    # ================= 4. 移交物理引擎执行合规性计算 =================
    print(">>> 启动 Darcy 流体力学及结构静力学方程组装与求逆...")
    t1 = time.time()
    results = solve_darcy_fsi(
        eleVol_batch=eleVol_batch,
        true_vec_batch=true_vec_batch,
        pred_vec_batch=pred_vec_batch,  # 直接喂入刚才算出的 GPU Tensor！
        config=config,
        device=device
    )
    print(f"[*] ✅ FEM 流固耦合分析完成，耗时: {time.time() - t1:.4f} s")

    C_EMs = results['C_EMs']
    C_ANN = results['C_ANN']
    rel_error = abs(C_ANN - C_EMs) / C_EMs * 100

    print('\n================ 终极流固耦合考核结果 ================')
    print(f'   [基准] 传统多尺度有限元合规性 (C_EMs) : {C_EMs:.6f}')
    print(f'   [代理] 神经网络预测合规性 (C_ANN)     : {C_ANN:.6f}')
    print(f'   >>> 宏观结构装配系统相对误差          : {rel_error:.4f} %')
    print('======================================================\n')

    # ================= 5. 渲染检验报告图表 =================
    os.makedirs('./fig', exist_ok=True)
    fig_path = f"./fig/Darcy_Validation_{data_size}.png"
    plot_multiscale_validation(results, use_samples, fig_path)
    print(f"[*] 评估报告已渲染并存储至: {fig_path}")


if __name__ == '__main__':
    main()