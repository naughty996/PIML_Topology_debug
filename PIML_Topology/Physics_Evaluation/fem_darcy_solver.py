# =========================================================================
# 功能: 张量化宏观刚度组装 + Darcy 压力求解
# =========================================================================
import numpy as np
import torch
from boundary import get_dofs, get_darcy_boundaries, get_structural_bc


def IFprj(xv, etaf=0.5, betaf=5.0):
    """消除灰度的流体渗透投影函数 (TOPress 原理)"""
    term_eta = np.tanh(betaf * etaf)
    term_1_minus_eta = np.tanh(betaf * (1.0 - etaf))
    num = term_eta + torch.tanh(betaf * (xv - etaf))
    den = term_eta + term_1_minus_eta
    return num / den


def reconstruct_SN_batch(vec_batch, device):
    """将一维预测向量 (576) 还原为微观高阶形函数矩阵 (72x8)"""
    B = vec_batch.shape[0]
    SN = np.zeros((B, 72, 8), dtype=np.float64)
    for i in range(B):
        # 兼容 MATLAB 的 Fortran 内存顺序
        temp_full = vec_batch[i].cpu().numpy().reshape((6, 6, 16), order='F')
        for col in range(8):
            vec_72 = np.zeros(72)
            vec_72[0::2] = temp_full[:, :, 2 * col].flatten(order='F')
            vec_72[1::2] = temp_full[:, :, 2 * col + 1].flatten(order='F')
            SN[i, :, col] = vec_72
    return torch.tensor(SN, dtype=torch.float64, device=device)


def solve_darcy_fsi(eleVol_batch, true_vec_batch, pred_vec_batch, config, device):
    """
    核心流固耦合有限元求解器
    输入: 密度场, 真值形函数, 预测形函数, config 字典
    输出: 包含柔度、位移场、宏观刚度的结果字典
    """
    use_samples = eleVol_batch.shape[0]
    nelx_c = use_samples // 20
    nely_c = 20
    nel_c = nelx_c * nely_c
    nno_c = (nelx_c + 1) * (nely_c + 1)

    # ================= 1. 计算宏观结构刚度矩阵 K (多尺度组装) =================
    nu, E0, Emin = config['physics']['nu'], config['physics']['E0'], config['physics']['Emin']

    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])
    KE_tensor = torch.tensor((1 / (1 - nu ** 2) / 24 * (
                np.block([[A11, A12], [A12.T, A11]]) + nu * np.block([[B11, B12], [B12.T, B11]]))).flatten(order='F'),
                             dtype=torch.float64, device=device)

    eleVol_t = torch.tensor(eleVol_batch, dtype=torch.float64, device=device)
    SN_true_t = reconstruct_SN_batch(true_vec_batch, device)
    SN_pred_t = reconstruct_SN_batch(pred_vec_batch, device)

    E_mod = Emin + eleVol_t * (E0 - Emin)
    sK_f_batch = torch.einsum('i, bj -> bji', KE_tensor, E_mod).reshape(use_samples, 1600)

    _, Udofs_f = get_dofs(config['physics']['nelx_f'], config['physics']['nely_f'])
    iK_f_t = torch.tensor(np.repeat(Udofs_f, 8, axis=1).flatten(), dtype=torch.long, device=device)
    jK_f_t = torch.tensor(np.tile(Udofs_f, (1, 8)).flatten(), dtype=torch.long, device=device)

    SK_i = torch.zeros((use_samples, 72, 72), dtype=torch.float64, device=device)
    batch_idx = torch.arange(use_samples, device=device).view(-1, 1)
    SK_i.index_put_((batch_idx, iK_f_t.unsqueeze(0), jK_f_t.unsqueeze(0)), sK_f_batch, accumulate=True)
    SK_i = (SK_i + SK_i.transpose(1, 2)) / 2.0

    # 核心映射：K_macro = N^T * K_fine * N
    K_macro_true = torch.bmm(torch.bmm(SN_true_t.transpose(1, 2), SK_i), SN_true_t)
    K_macro_pred = torch.bmm(torch.bmm(SN_pred_t.transpose(1, 2), SK_i), SN_pred_t)

    sK_global_true = K_macro_true.reshape(use_samples, 64).flatten()
    sK_global_pred = K_macro_pred.reshape(use_samples, 64).flatten()

    Pdofs_c, Udofs_c = get_dofs(nelx_c, nely_c)
    indices_c = torch.tensor(np.vstack((np.repeat(Udofs_c, 8, axis=1).flatten(), np.tile(Udofs_c, (1, 8)).flatten())),
                             dtype=torch.long, device=device)

    K_TRUE_global = torch.sparse_coo_tensor(indices_c, sK_global_true, size=(2 * nno_c, 2 * nno_c)).to_dense()
    K_PRED_global = torch.sparse_coo_tensor(indices_c, sK_global_pred, size=(2 * nno_c, 2 * nno_c)).to_dense()
    K_TRUE_global = (K_TRUE_global + K_TRUE_global.T) / 2.0
    K_PRED_global = (K_PRED_global + K_PRED_global.T) / 2.0

    # ================= 2. Darcy 流体力学求解 (计算压力场 PF) =================
    Kv, epsf, r_param, Dels = config['darcy']['Kv'], config['darcy']['epsf'], config['darcy']['r_param'], \
    config['darcy']['Dels']
    Ds = (np.log(r_param) / Dels) ** 2 * epsf

    xphys = eleVol_t.mean(dim=1)
    IF_val = IFprj(xphys)
    Kc = Kv * (1 - (1 - epsf) * IF_val)
    Dc = Ds * IF_val

    Kp = torch.tensor([[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]], dtype=torch.float64,
                      device=device) / 6.0
    KDp = torch.tensor([[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]], dtype=torch.float64,
                       device=device) / 36.0
    Te = torch.tensor([[-2, 2, 1, -1], [-2, -1, 1, 2], [-2, 2, 1, -1], [-1, -2, 2, 1],
                       [-1, 1, 2, -2], [-1, -2, 2, 1], [-1, 1, 2, -2], [-2, -1, 1, 2]], dtype=torch.float64,
                      device=device) / 12.0

    Ae = torch.outer(Kc, Kp.flatten()) + torch.outer(Dc, KDp.flatten())
    iP = torch.tensor(np.repeat(Pdofs_c, 4, axis=1).flatten(), dtype=torch.long, device=device)
    jP = torch.tensor(np.tile(Pdofs_c, (1, 4)).flatten(), dtype=torch.long, device=device)

    AG = torch.sparse_coo_tensor(torch.stack([iP, jP]), Ae.flatten(), size=(nno_c, nno_c)).to_dense()
    AG = (AG + AG.T) / 2.0

    Lnode, Rnode, Bnode, Tnode = get_darcy_boundaries(nno_c, nely_c, device)

    # 施加流体压力边界条件
    PF = torch.full((nno_c, 1), 1e-5, dtype=torch.float64, device=device)
    PF[Lnode] = PF[Rnode] = PF[Bnode] = 0.0
    PF[Tnode] = config['darcy']['Pin']

    fixedPdofs = torch.unique(torch.cat([Lnode, Rnode, Bnode, Tnode]))
    mask = torch.ones(nno_c, dtype=torch.bool, device=device)
    mask[fixedPdofs] = False
    freePdofs = torch.arange(nno_c, device=device)[mask]

    idx_free_grid = torch.meshgrid(freePdofs, freePdofs, indexing='ij')
    idx_free_fixed = torch.meshgrid(freePdofs, fixedPdofs, indexing='ij')
    PF[freePdofs] = torch.linalg.solve(AG[idx_free_grid], -AG[idx_free_fixed] @ PF[fixedPdofs])

    # ================= 3. 结构位移场求解 =================
    iT = torch.tensor(np.repeat(Udofs_c, 4, axis=1).flatten(), dtype=torch.long, device=device)
    jT = torch.tensor(np.tile(Pdofs_c, (1, 8)).flatten(), dtype=torch.long, device=device)
    TG = torch.sparse_coo_tensor(torch.stack([iT, jT]), Te.flatten().repeat(nel_c), size=(2 * nno_c, nno_c)).to_dense()
    F_pressure = -TG @ PF

    freedofs_t, _ = get_structural_bc(Lnode, nno_c, device)
    idx_u_grid = torch.meshgrid(freedofs_t, freedofs_t, indexing='ij')

    U_true = torch.zeros(2 * nno_c, 1, dtype=torch.float64, device=device)
    U_pred = torch.zeros(2 * nno_c, 1, dtype=torch.float64, device=device)

    U_true[freedofs_t] = torch.linalg.solve(K_TRUE_global[idx_u_grid], F_pressure[freedofs_t])
    U_pred[freedofs_t] = torch.linalg.solve(K_PRED_global[idx_u_grid], F_pressure[freedofs_t])

    # 计算系统合规性 (Compliance)
    C_EMs = torch.matmul(F_pressure.T, U_true).item()
    C_ANN = torch.matmul(F_pressure.T, U_pred).item()

    return {
        'C_EMs': C_EMs,
        'C_ANN': C_ANN,
        'U_true': U_true,
        'U_pred': U_pred,
        'sK_global_true': sK_global_true,
        'sK_global_pred': sK_global_pred,
        'nelx_c': nelx_c,
        'nely_c': nely_c
    }