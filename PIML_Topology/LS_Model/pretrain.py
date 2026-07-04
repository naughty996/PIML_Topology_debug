import torch
import numpy as np
from initial_ls_net import run_initial_levelset_prediction
from set_bc import initial_ls_func

# 定义网格参数
nx, ny = 200, 100
domain_type = 'multi_holes'
H_d = 1.0  # 平滑带宽
ls_r = 0.15 # 孔洞半径
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 生成坐标网格 (替代 build_precompute)
x = np.linspace(0, 8.0, nx + 1)
y = np.linspace(0, 4.0, ny + 1)
X, Y = np.meshgrid(x, y, indexing='ij')
node_coordinate = np.stack([X.flatten(order='F'), Y.flatten(order='F')], axis=1)

# 2. 生成初始多孔 phi
phi = initial_ls_func(nx, ny, domain_type, H_d, ls_r).reshape(-1, 1, order='F')

# 3. 运行预训练
print(">>> 正在生成初始几何图纸...")
results = run_initial_levelset_prediction(
    coords=node_coordinate,
    phi=phi,
    ndx=nx + 1,
    ndy=ny + 1,
    device=device,
    save_dir='./model_data',
    model_name='init_topo_weights'
)

print(f">>> 初始权重已生成: {results['best_model_path']}")