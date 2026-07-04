# =========================================================================
# 功能: PIML-Darcy 拓扑优化专用的坐标映射与物理边界掩码器
# =========================================================================
import torch


def get_macro_element_coords(nelx_c, nely_c, device='cpu'):
    """生成宏观网格的单元形心坐标 (Element Centroids)，维度严格对齐 PIMLNet"""
    x_centers = torch.linspace(-1, 1, nelx_c + 1)[:-1] + (1.0 / nelx_c)
    y_centers = torch.linspace(-1, 1, nely_c + 1)[:-1] + (1.0 / nely_c)

    Y, X = torch.meshgrid(y_centers, x_centers, indexing='ij')
    coords = torch.stack([X.flatten(order='F'), Y.flatten(order='F')], dim=1)

    return coords.to(device, dtype=torch.float64)


def apply_darcy_bc_mask(raw_density, nelx_c, nely_c):
    """施加 Darcy 流体力学约束，并扩维适配 PIMLNet 的 25 维微观输入"""
    rho = raw_density.clone()
    rho_grid = rho.view(nelx_c, nely_c)

    # 强制左侧端面为实体，确保结构锚固
    rho_grid[0, :] = 1.0

    # 强制顶部中央为孔洞，模拟水流注入点
    inlet_start = nelx_c // 2 - 2
    inlet_end = nelx_c // 2 + 2
    rho_grid[inlet_start:inlet_end, -1] = 0.0

    rho_flat = torch.clamp(rho_grid.flatten(), min=1e-4, max=1.0)

    # 【张量映射】：1个宏观密度广播为 25 个微观单元密度
    rho_25d = rho_flat.unsqueeze(1).repeat(1, 25)

    return rho_25d