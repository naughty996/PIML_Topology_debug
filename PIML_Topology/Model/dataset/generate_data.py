# =========================================================================
# 功能: PIML 数据集全流程生成 (随机密度极化 -> 去重 -> FEM 求解 -> 分离导出 H5)
# =========================================================================
import numpy as np
import h5py
import time
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# =========================================================================
# 用户配置区 (User Configuration)
# =========================================================================
NUM_TARGET = 5000  # 最终目标样本数 (例如 5000 或 50000)
NELX, NELY = 5, 5  # 微观单元网格尺寸
BETA = 6  # 极化参数 (越大越趋近于 0/1)
METHOD_FLAG = 1  # 1: 向量化欧氏距离去重, 2: DBSCAN 聚类去重
LIMIT = 1.0  # 去重距离阈值

# 独立的文件命名
DATASET_FILE = f'PIML_Dataset_{NUM_TARGET}_{NELX}x{NELY}.h5'
LABELS_FILE = f'PIML_Labels_{NUM_TARGET}_{NELX}x{NELY}.h5'


# =========================================================================
# 模块 1: 生成极化密度数据与去重
# =========================================================================
def generate_and_filter_data(num_target, nelx, nely, beta, method_flag, limit):
    print('>>> [1/3] 正在生成原始极化随机样本并去重...')
    unique_data = np.empty((0, nelx * nely))
    original_data = np.empty((0, nelx * nely))
    num_generate = num_target

    t0 = time.time()
    while unique_data.shape[0] < num_target:
        # 1. 生成原始随机分布并应用 SIMP 极化映射
        eta = np.random.rand(num_generate, nely, nelx)
        rho = (np.tanh(beta * (eta - 0.5)) + np.tanh(beta / 2)) / (2 * np.tanh(beta / 2))
        rho_vec = rho.reshape(num_generate, -1)  # 铺平为 (N, 25)

        original_data = np.vstack((original_data, rho_vec))
        combined = np.vstack((unique_data, rho_vec)) if unique_data.size else rho_vec

        # 2. 去重过滤
        if method_flag == 1:
            keep_idx = np.ones(combined.shape[0], dtype=bool)
            for i in range(combined.shape[0] - 1):
                if not keep_idx[i]: continue
                diffs = np.sqrt(np.sum((combined[i + 1:] - combined[i]) ** 2, axis=1))
                dup_idx = np.where(diffs < limit)[0]
                keep_idx[i + 1 + dup_idx] = False
            unique_data = combined[keep_idx]

        elif method_flag == 2:
            db = DBSCAN(eps=limit, min_samples=1).fit(combined)
            _, unique_indices = np.unique(db.labels_, return_index=True)
            unique_data = combined[np.sort(unique_indices)]
        else:
            raise ValueError("method_flag 必须为 1 或 2")

        num_generate = max(1, num_target - unique_data.shape[0])

    print(f'>>> 数据生成耗时: {time.time() - t0:.2f} s')

    unique_data = unique_data[:num_target, :]
    original_data = original_data[:num_target, :]
    return unique_data, original_data


# =========================================================================
# 模块 2: 有限元求解器 (施加 Dirichlet 边界条件)
# =========================================================================
def apply_linear_bc_dirichlet(K, nodes):
    n_nodes = nodes.shape[0]

    X, Y = nodes[:, 0], nodes[:, 1]
    X_n = (X - np.min(X)) / (np.max(X) - np.min(X))
    Y_n = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

    tol = 1e-6
    is_bnd = (X_n < tol) | (X_n > 1 - tol) | (Y_n < tol) | (Y_n > 1 - tol)
    bc_idx = np.where(is_bnd)[0]
    inter_idx = np.where(~is_bnd)[0]

    bc_dofs = np.sort(np.concatenate((2 * bc_idx, 2 * bc_idx + 1)))
    inter_dofs = np.sort(np.concatenate((2 * inter_idx, 2 * inter_idx + 1)))

    N1 = (1 - X_n[bc_idx]) * (1 - Y_n[bc_idx])
    N2 = X_n[bc_idx] * (1 - Y_n[bc_idx])
    N3 = X_n[bc_idx] * Y_n[bc_idx]
    N4 = (1 - X_n[bc_idx]) * Y_n[bc_idx]

    Ub_mat = np.zeros((len(bc_dofs), 8))
    Ub_mat[0::2, 0] = N1;
    Ub_mat[1::2, 1] = N1
    Ub_mat[0::2, 2] = N2;
    Ub_mat[1::2, 3] = N2
    Ub_mat[0::2, 4] = N3;
    Ub_mat[1::2, 5] = N3
    Ub_mat[0::2, 6] = N4;
    Ub_mat[1::2, 7] = N4

    K_ii = K[np.ix_(inter_dofs, inter_dofs)].tocsc()
    K_ib = K[np.ix_(inter_dofs, bc_dofs)].tocsc()

    RHS = -K_ib.dot(Ub_mat)
    U_ii = spsolve(K_ii, RHS)

    U_all = np.zeros((2 * n_nodes, 8))
    U_all[bc_dofs, :] = Ub_mat
    U_all[inter_dofs, :] = U_ii

    return U_all


# =========================================================================
# 主函数 (Pipeline)
# =========================================================================
def main():
    # ---------------------------------------------------------------------
    # Step 1: 数据生成
    # ---------------------------------------------------------------------
    dataset_density, original_data = generate_and_filter_data(NUM_TARGET, NELX, NELY, BETA, METHOD_FLAG, LIMIT)

    print('>>> 绘制去重前后样本可视化 (关闭图像窗口以继续运行计算)...')
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(np.mean(original_data, axis=1), 'o', markersize=2)
    plt.title('Raw Samples Mean')
    plt.subplot(2, 2, 2)
    plt.plot(np.mean(dataset_density, axis=1), 'o', markersize=2)
    plt.title('Filtered Samples Mean')
    plt.subplot(2, 2, 3)
    plt.hist(original_data.flatten(), bins=50)
    plt.title('Raw Histogram')
    plt.subplot(2, 2, 4)
    plt.hist(dataset_density.flatten(), bins=50)
    plt.title('Filtered Histogram')
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------
    # Step 2: FEM 初始化与单元刚度矩阵配置
    # ---------------------------------------------------------------------
    print(f'>>> [2/3] 开启 FEM 并行组装物理标签 (规模: {NUM_TARGET})...')
    nu, E0, Emin = 0.3, 1.0, 1e-9
    noden = (NELX + 1) * (NELY + 1)

    S_Nodes = np.zeros((noden, 2))
    for ex in range(NELX + 1):
        for ey in range(NELY + 1):
            S_Nodes[ex * (NELY + 1) + ey, :] = [ex, ey]

    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])
    KE = 1 / (1 - nu ** 2) / 24 * (np.block([[A11, A12], [A12.T, A11]]) + nu * np.block([[B11, B12], [B12.T, B11]]))

    edofMat = np.zeros((NELX * NELY, 8), dtype=int)
    el = 0
    for ex in range(NELX):
        for ey in range(NELY):
            n1 = ex * (NELY + 1) + ey
            n2 = (ex + 1) * (NELY + 1) + ey
            n3 = n2 + 1
            n4 = n1 + 1
            edofMat[el, :] = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1]
            el += 1

    iK = np.repeat(edofMat, 8, axis=1).flatten()
    jK = np.tile(edofMat, (1, 8)).flatten()

    Labels_192_all = np.zeros((NUM_TARGET, 192), dtype=np.float64)
    Labels_576_all = np.zeros((NUM_TARGET, 576), dtype=np.float64)

    # ---------------------------------------------------------------------
    # Step 3: 主循环批量求解
    # ---------------------------------------------------------------------
    t1 = time.time()
    for i in range(NUM_TARGET):
        eleVol = dataset_density[i, :]
        sK = (KE.flatten('F')[:, None] * (Emin + eleVol * (E0 - Emin))).flatten('F')
        SK = coo_matrix((sK, (iK, jK)), shape=(2 * noden, 2 * noden)).tocsr()
        SK = (SK + SK.T) / 2.0

        SN = apply_linear_bc_dirichlet(SK, S_Nodes)

        temp_full = np.zeros((NELY + 1, NELX + 1, 16), order='F')
        for col in range(8):
            vec = SN[:, col]
            temp_full[:, :, 2 * col] = vec[0::2].reshape((NELY + 1, NELX + 1), order='F')
            temp_full[:, :, 2 * col + 1] = vec[1::2].reshape((NELY + 1, NELX + 1), order='F')

        Labels_576_all[i, :] = temp_full.flatten(order='F')
        internal_data = temp_full[1:5, 1:5, 0:12]
        Labels_192_all[i, :] = internal_data.flatten(order='F')

        if (i + 1) % 500 == 0:
            print(f"  ...已完成 {i + 1}/{NUM_TARGET} 个样本求逆解析")

    print(f'>>> FEM 求解耗时: {time.time() - t1:.2f} s')

    # ---------------------------------------------------------------------
    # Step 4: Python 端组装索引用法生成与独立导出
    # ---------------------------------------------------------------------
    print(f'>>> [3/3] 正在序列化并导出至独立的 HDF5 文件...')
    All_Indices = np.arange(576).reshape((6, 6, 16), order='F')
    Core_Indices = All_Indices[1:5, 1:5, 0:12]
    P4_Indices = All_Indices[1:5, 1:5, 12:16]
    Core_Idx_Py = Core_Indices.flatten(order='F')
    P4_Idx_Py = P4_Indices.flatten(order='F')

    # 清理旧文件（如果存在）
    if os.path.exists(DATASET_FILE): os.remove(DATASET_FILE)
    if os.path.exists(LABELS_FILE): os.remove(LABELS_FILE)

    # 1. 导出输入特征 (Dataset)
    with h5py.File(DATASET_FILE, 'w') as f:
        f.create_dataset('/Dataset_Density', data=dataset_density, dtype='float64')
    print(f'>>> ✅ 密度数据集已保存至: {DATASET_FILE}')

    # 2. 导出目标标签与组装索引 (Labels)
    with h5py.File(LABELS_FILE, 'w') as f:
        f.create_dataset('/Labels_192', data=Labels_192_all, dtype='float64')
        f.create_dataset('/Labels_576', data=Labels_576_all, dtype='float64')
        f.create_dataset('/Core_Idx', data=Core_Idx_Py, dtype='int32')
        f.create_dataset('/P4_Idx', data=P4_Idx_Py, dtype='int32')
    print(f'>>> ✅ 物理标签集已保存至: {LABELS_FILE}')


if __name__ == '__main__':
    main()