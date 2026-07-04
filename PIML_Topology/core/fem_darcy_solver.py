import numpy as np
import torch


def get_dofs(nelx, nely):
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
    Tnode = torch.arange(0, nno_c, nely_c + 1, device=device)
    Bnode = torch.arange(nely_c, nno_c, nely_c + 1, device=device)
    Lnode = torch.arange(0, nely_c + 1, device=device)
    Rnode = torch.arange(nno_c - nely_c - 1, nno_c, device=device)
    return Lnode, Rnode, Bnode, Tnode


def get_structural_bc(Lnode, nno_c, device):
    fixed_dofs = torch.unique(torch.cat([2 * Lnode, 2 * Lnode + 1]))
    mask_u = torch.ones(2 * nno_c, dtype=torch.bool, device=device)
    mask_u[fixed_dofs] = False
    freedofs = torch.arange(2 * nno_c, device=device)[mask_u]
    return freedofs, fixed_dofs
#条件修改


def IFprj(xv, etaf=0.5, betaf=5.0):
    term_eta = np.tanh(betaf * etaf)
    term_1_minus_eta = np.tanh(betaf * (1.0 - etaf))
    num = term_eta + torch.tanh(betaf * (xv - etaf))
    den = term_eta + term_1_minus_eta
    return num / den


def reconstruct_SN_batch(vec_batch, device):
    B = vec_batch.shape[0]
    SN = np.zeros((B, 72, 8), dtype=np.float64)
    for i in range(B):
        temp_full = vec_batch[i].detach().cpu().numpy().reshape((6, 6, 16), order='F')
        for col in range(8):
            vec_72 = np.zeros(72)
            vec_72[0::2] = temp_full[:, :, 2 * col].flatten(order='F')
            vec_72[1::2] = temp_full[:, :, 2 * col + 1].flatten(order='F')
            SN[i, :, col] = vec_72
    return torch.tensor(SN, dtype=torch.float64, device=device)


def solve_darcy_fsi(eleVol_batch, true_vec_batch, pred_vec_batch, config, device):
    use_samples = eleVol_batch.shape[0]
    nelx_c = use_samples // config['physics']['nely_c']
    nely_c = config['physics']['nely_c']
    nel_c = nelx_c * nely_c
    nno_c = (nelx_c + 1) * (nely_c + 1)

    nu, E0, Emin = config['physics']['nu'], config['physics']['E0'], config['physics']['Emin']
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])
    KE_tensor = torch.tensor((1 / (1 - nu ** 2) / 24 * (
            np.block([[A11, A12], [A12.T, A11]]) + nu * np.block([[B11, B12], [B12.T, B11]]))).flatten(order='F'),
                             dtype=torch.float64, device=device)

    eleVol_t = eleVol_batch.to(dtype=torch.float64, device=device)
    E_mod = Emin + eleVol_t * (E0 - Emin)
    sK_f_batch = torch.einsum('i, bj -> bji', KE_tensor, E_mod).reshape(use_samples, 1600)

    _, Udofs_f = get_dofs(config['physics']['nelx_f'], config['physics']['nely_f'])
    iK_f_t = torch.tensor(np.repeat(Udofs_f, 8, axis=1).flatten(), dtype=torch.long, device=device)
    jK_f_t = torch.tensor(np.tile(Udofs_f, (1, 8)).flatten(), dtype=torch.long, device=device)

    SK_i = torch.zeros((use_samples, 72, 72), dtype=torch.float64, device=device)
    batch_idx = torch.arange(use_samples, device=device).view(-1, 1)
    SK_i.index_put_((batch_idx, iK_f_t.unsqueeze(0), jK_f_t.unsqueeze(0)), sK_f_batch, accumulate=True)
    SK_i = (SK_i + SK_i.transpose(1, 2)) / 2.0

    Pdofs_c, Udofs_c = get_dofs(nelx_c, nely_c)
    indices_c = torch.tensor(np.vstack((np.repeat(Udofs_c, 8, axis=1).flatten(), np.tile(Udofs_c, (1, 8)).flatten())),
                             dtype=torch.long, device=device)

    SN_pred_t = reconstruct_SN_batch(pred_vec_batch, device)
    K_macro_pred = torch.bmm(torch.bmm(SN_pred_t.transpose(1, 2), SK_i), SN_pred_t)
    sK_global_pred = K_macro_pred.reshape(use_samples, 64).flatten()

    K_PRED_global = torch.sparse_coo_tensor(indices_c, sK_global_pred, size=(2 * nno_c, 2 * nno_c)).to_dense()
    K_PRED_global = (K_PRED_global + K_PRED_global.T) / 2.0
#
    Kv, epsf, r_param, Dels = config['darcy']['Kv'], config['darcy']['epsf'], config['darcy']['r_param'], \
    config['darcy']['Dels']
    Ds = (np.log(r_param) / Dels) ** 2 * epsf

    xphys = eleVol_t.mean(dim=1)
    IF_val = IFprj(xphys)
    #平滑过渡，可有可无
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

    PF = torch.full((nno_c, 1), 1e-5, dtype=torch.float64, device=device)
    PF[Lnode] = PF[Rnode] = PF[Bnode] = 0.0
    PF[Tnode] = config['darcy']['Pin']
    #边界条件的pin和pout

    fixedPdofs = torch.unique(torch.cat([Lnode, Rnode, Bnode, Tnode]))
    mask = torch.ones(nno_c, dtype=torch.bool, device=device)
    mask[fixedPdofs] = False
    freePdofs = torch.arange(nno_c, device=device)[mask]

    idx_free_grid = torch.meshgrid(freePdofs, freePdofs, indexing='ij')
    idx_free_fixed = torch.meshgrid(freePdofs, fixedPdofs, indexing='ij')
    PF[freePdofs] = torch.linalg.solve(AG[idx_free_grid], -AG[idx_free_fixed] @ PF[fixedPdofs])

    iT = torch.tensor(np.repeat(Udofs_c, 4, axis=1).flatten(), dtype=torch.long, device=device)
    jT = torch.tensor(np.tile(Pdofs_c, (1, 8)).flatten(), dtype=torch.long, device=device)
    TG = torch.sparse_coo_tensor(torch.stack([iT, jT]), Te.flatten().repeat(nel_c), size=(2 * nno_c, nno_c)).to_dense()
    F_pressure = -TG @ PF
    #断开
    freedofs_t, _ = get_structural_bc(Lnode, nno_c, device)
    idx_u_grid = torch.meshgrid(freedofs_t, freedofs_t, indexing='ij')

    U_pred = torch.zeros(2 * nno_c, 1, dtype=torch.float64, device=device)
    U_pred[freedofs_t] = torch.linalg.solve(K_PRED_global[idx_u_grid], F_pressure[freedofs_t])
    C_ANN = torch.matmul(F_pressure.T, U_pred).squeeze()

    C_EMs = None
    U_true = None
    sK_global_true = None

    if true_vec_batch is not None:
        SN_true_t = reconstruct_SN_batch(true_vec_batch, device)
        K_macro_true = torch.bmm(torch.bmm(SN_true_t.transpose(1, 2), SK_i), SN_true_t)
        sK_global_true = K_macro_true.reshape(use_samples, 64).flatten()

        K_TRUE_global = torch.sparse_coo_tensor(indices_c, sK_global_true, size=(2 * nno_c, 2 * nno_c)).to_dense()
        K_TRUE_global = (K_TRUE_global + K_TRUE_global.T) / 2.0

        U_true = torch.zeros(2 * nno_c, 1, dtype=torch.float64, device=device)
        U_true[freedofs_t] = torch.linalg.solve(K_TRUE_global[idx_u_grid], F_pressure[freedofs_t])
        C_EMs = torch.matmul(F_pressure.T, U_true).squeeze()

    return {
        'C_ANN': C_ANN,
        'C_EMs': C_EMs,
        'U_pred': U_pred,
        'U_true': U_true,
        'sK_global_pred': sK_global_pred,
        'sK_global_true': sK_global_true,
        'nelx_c': nelx_c,
        'nely_c': nely_c
    }