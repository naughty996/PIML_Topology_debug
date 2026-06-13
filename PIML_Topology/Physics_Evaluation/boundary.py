# =========================================================================
# 功能: 提取结构网格的物理边界节点，以及计算自由度 (DOF) 映射
# =========================================================================
import numpy as np
import torch

def get_dofs(nelx, nely):
    """
    获取全局自由度映射矩阵。
    返回:
      Pdofs: 压力自由度 (每单元4角点，每点1个标量，共4维)
      Udofs: 位移自由度 (每单元4角点，每点XY双矢量，共8维)
    """
    nel = nelx * nely
    Pdofs = np.zeros((nel, 4), dtype=np.int64)
    Udofs = np.zeros((nel, 8), dtype=np.int64)
    el = 0
    for ex in range(nelx):
        for ey in range(nely):
            n1 = ex * (nely + 1) + ey
            n2 = (ex + 1) * (nely + 1) + ey
            n3 = n2 + 1
            n4 = n1 + 1
            Pdofs[el, :] = [n1, n2, n3, n4]
            Udofs[el, :] = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1]
            el += 1
    return Pdofs, Udofs

def get_darcy_boundaries(nno_c, nely_c, device):
    """提取 Darcy 流体渗透分析的四个几何边界节点索引"""
    Tnode = torch.arange(0, nno_c, nely_c + 1, device=device)
    Bnode = torch.arange(nely_c, nno_c, nely_c + 1, device=device)
    Lnode = torch.arange(0, nely_c + 1, device=device)
    Rnode = torch.arange(nno_c - nely_c - 1, nno_c, device=device)
    return Lnode, Rnode, Bnode, Tnode

def get_structural_bc(Lnode, nno_c, device):
    """获取宏观结构的力学边界条件 (悬臂梁：左侧全固定)"""
    fixed_dofs = torch.unique(torch.cat([2 * Lnode, 2 * Lnode + 1]))
    mask_u = torch.ones(2 * nno_c, dtype=torch.bool, device=device)
    mask_u[fixed_dofs] = False
    freedofs = torch.arange(2 * nno_c, device=device)[mask_u]
    return freedofs, fixed_dofs