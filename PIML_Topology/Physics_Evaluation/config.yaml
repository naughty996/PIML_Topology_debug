data:
  # ---------------- 数据与模型溯源 ----------------
  data_size: 50000               # 原始数据集总样本数
  use_samples: 1000              # 抽取用于拼接宏观悬臂梁的微观单元数量
  random_seed: 2026              # 随机种子，必须与训练时严格一致，以防测试集数据泄露

  # ---------------- 相对路径配置 ----------------
  # 自动回退到父级目录并进入 Model 文件夹取数据
  dataset_path: "../Model/dataset/PIML_Dataset_50000_5x5.h5"
  labels_path: "../Model/dataset/PIML_Labels_50000_5x5.h5"

  # ⚠️ [必改项] 请将这里替换为你刚才训练生成的真实的 .pth 文件名
  model_path: "../Model/train_models/PIML_50k_bs1024_lr1e-3_V2Funnel_PolarB6_0524_1838.pth"


physics:
  # ---------------- 结构力学参数 ----------------
  nu: 0.3                        # 材料泊松比 (Poisson's ratio)
  E0: 1.0                        # 实体材料杨氏模量 (Young's modulus for solid)
  Emin: 1.0e-9                   # 空洞区域的惩罚杨氏模量 (防止矩阵求逆时产生奇异)
  nelx_f: 5                      # 微观晶胞 X 方向网格数
  nely_f: 5                      # 微观晶胞 Y 方向网格数


darcy:
  # ---------------- Darcy 流体渗透模型参数 ----------------
  Kv: 1.0                        # 基础流体渗透率 (Base Permeability)
  epsf: 1.0e-7                   # 固体区域的极小渗透率参数
  r_param: 0.1                   # 内部排水控制参数 (Drainage penalty parameter)
  Dels: 2.0                      # Darcy 模型几何尺度系数
  Pin: 1.0                       # 施加于悬臂梁顶部的等效流体注入压力 (Input Pressure)