# =========================================================================
# 功能: 负责磁盘 I/O、张量维度转置、训练/测试集切分及批次封装
# =========================================================================
import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def get_piml_dataloaders(data_size, batch_size, test_size=0.1, random_state=2026, device='cpu'):
    """
    加载 HDF5 数据并返回适用于 PyTorch 的 DataLoader 及测试集张量。

    :param data_size: 目标读取的样本总数
    :param batch_size: 训练批次大小
    :param test_size: 测试集所占比例
    :param random_state: 随机种子，保证每次切分的数据集一致，利于复现
    :param device: 目标设备 (CPU/CUDA)
    """
    dataset_path = f'./dataset/PIML_Dataset_{data_size}_5x5.h5'
    labels_path = f'./dataset/PIML_Labels_{data_size}_5x5.h5'

    # 1. 加载输入特征 (密度场)
    with h5py.File(dataset_path, 'r') as f:
        X = np.array(f['/Dataset_Density'])
        # 兼容性检查：确保样本数位于第 0 维度 (N, Features)
        if X.shape[0] != data_size:
            X = X.T

    # 2. 加载目标标签 (192维核心特征 & 576维全场特征)
    with h5py.File(labels_path, 'r') as f:
        Y_192 = np.array(f['/Labels_192'])
        if Y_192.shape[0] != data_size:
            Y_192 = Y_192.T

        Y_576 = np.array(f['/Labels_576'])
        if Y_576.shape[0] != data_size:
            Y_576 = Y_576.T

    # 3. 数据集划分 (锁定随机种子)
    indices = np.arange(data_size)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

    # 4. 构建训练集张量并迁移至计算设备
    # 训练过程仅需监督 192 维的核心物理变量，可大幅节约显存与计算力
    X_train_t = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    Y_192_train_t = torch.tensor(Y_192[train_idx], dtype=torch.float32).to(device)

    # 5. 构建测试集张量
    # 测试集除 192 维外，必须保留 576 维真值，用于最终的物理场绝对误差评估
    X_test_t = torch.tensor(X[test_idx], dtype=torch.float32).to(device)
    Y_192_test_t = torch.tensor(Y_192[test_idx], dtype=torch.float32).to(device)
    Y_576_test_t = torch.tensor(Y_576[test_idx], dtype=torch.float32).to(device)

    # 6. 封装为 PyTorch 标准的数据加载器
    train_loader = DataLoader(
        TensorDataset(X_train_t, Y_192_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, X_test_t, Y_192_test_t, Y_576_test_t