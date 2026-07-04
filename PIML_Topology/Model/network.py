# =========================================================================
# 功能: 定义 PIMLNet (物理信息代理模型) 与 FieldAssembler (基于物理规则的降维还原器)
# =========================================================================
import torch
import torch.nn as nn
import h5py
import numpy as np


class PIMLNet(nn.Module):
    """
    基于全连接层与层归一化 (Layer Normalization) 的物理信息代理模型。

    输入维度: 25 维 (5x5 宏观单元内部的微观密度场排布)
    输出维度: 192 维 (剔除物理冗余后的独立高阶形函数分量)

    网络特性:
    使用 LayerNorm 替代 BatchNorm，以适应拓扑优化中样本间极化密度差异过大导致的协变量偏移；
    使用 Tanh 激活函数保证输出场的平滑连续性，符合有限元位移场的数学假设。
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # 1. 渐进升维
            nn.Linear(25, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            # 2. 高维核心算力层 (双 512)
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),

            # 3. 过渡漏斗层 (平滑降维，避免信息突变)
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            # 4. 线性输出 192 维独立特征
            nn.Linear(256, 192)
        )

    def forward(self, x):
        return self.layers(x)


class FieldAssembler:
    """
    基于配分性质 (Partition of Unity) 的物理场组装器。

    作用:
    有限元形函数在空间任一点的总和必须满足特定物理守恒律(如平移刚体位移为0)。
    通过此组装器，网络仅需预测 192 个线性独立自由度，剩余的冗余自由度
    通过严格的物理解析方程补全，最终无损还原为 576 维的完整物理场。
    """

    def __init__(self, h5_labels_path, device):
        self.device = device

        # 加载静态掩码与索引规则 (从 HDF5 中读取 MATLAB 生成的 Fortran 序索引)
        with h5py.File(h5_labels_path, 'r') as f:
            Y_576_raw = np.array(f['/Labels_576'])
            if Y_576_raw.shape[0] == 576:
                Y_576_raw = Y_576_raw.T

            # baseline 用于提供固定的零边界等恒定数值
            self.baseline = torch.tensor(Y_576_raw[0, :], dtype=torch.float32).to(device)
            # core_idx: 网络直接预测的 192 维核心变量在 576 维全场中的位置
            self.core_idx = torch.tensor(np.array(f['/Core_Idx']).flatten(), dtype=torch.long).to(device)
            # p4_idx: 需通过物理方程推导补全的变量位置
            self.p4_idx = torch.tensor(np.array(f['/P4_Idx']).flatten(), dtype=torch.long).to(device)

    def assemble_576(self, pred_192):
        """
        将批量的 192 维预测张量组装为 576 维物理场张量。
        :param pred_192: 形状为 (Batch_Size, 192) 的网络输出
        :return: 形状为 (Batch_Size, 576) 的完整物理场预测
        """
        B = pred_192.shape[0]
        # 【核心修改】：克隆 baseline 之后，显式将精度同步提升至输入张量的精度 (float64)
        pred_576 = self.baseline.unsqueeze(0).repeat(B, 1).clone().to(dtype=pred_192.dtype)

        # 将 192 维张量重塑为便于计算的空间通道格式 (B, 12个节点, 16个基向量分量)
        pred_192_ch = pred_192.view(B, 12, 16)

        # 严格的有限元形函数守恒法则：第4角点的值由前3个角点线性推导
        ch13 = 1.0 - (pred_192_ch[:, 0, :] + pred_192_ch[:, 4, :] + pred_192_ch[:, 8, :])
        ch14 = 0.0 - (pred_192_ch[:, 1, :] + pred_192_ch[:, 5, :] + pred_192_ch[:, 9, :])
        ch15 = 0.0 - (pred_192_ch[:, 2, :] + pred_192_ch[:, 6, :] + pred_192_ch[:, 10, :])
        ch16 = 1.0 - (pred_192_ch[:, 3, :] + pred_192_ch[:, 7, :] + pred_192_ch[:, 11, :])

        # 拼接生成的 64 维冗余特征
        pred_64 = torch.cat([ch13, ch14, ch15, ch16], dim=1)

        # 此时所有张量均为 Double 精度，切片写回将完美通行
        pred_576[:, self.core_idx] = pred_192
        pred_576[:, self.p4_idx] = pred_64
        return pred_576