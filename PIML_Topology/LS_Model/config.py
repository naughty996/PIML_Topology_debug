# Neural level-set network parameters.
network_paras = {
    # Activation function for the implicit level-set network. Typical: "sine".
    "act_func": "sine",
    # Input coordinate dimension. For 2D topology optimization this should be 2.
    "neu_in": 2,
    # Output dimension. The network predicts scalar nodal phi, so this should be 1.
    "neu_out": 1,
    # Hidden network layout [num_hidden_layers, neurons_per_layer].
    # Reference range for this problem: [3, 64] ~ [5, 100].
    "neu_hidden": [3, 64],
    # Learning rate for initial level-set fitting. Reference range: 1e-4 ~ 1e-3.
    "lr": 1e-3,
    # Initial level-set fitting tolerance. Reference range: 1e-3 ~ 1e-2.
    "ls_tol": 1e-2,
    # Max epochs for initial level-set fitting. Used only if the pretrained model file is unavailable.
    "max_epoch": [30000],
    # Main network/device. Network inference, phi->density mapping and network fitting
    # run here. The full FEM objective/sensitivity and ZPR update run on CPU.
    "device": "cuda",
    # Floating-point precision. Use float64 for dynamic topology optimization.
    "dtype": "float64",
    # Global random seed used for optional initial level-set pretraining and optimizer initialization.
    # Reference: any non-negative integer. Keep fixed for reproducible runs.
    "seed": 2026,
}

optimize_paras = {
    # Target global volume fraction. Common range: 0.3 ~ 0.6.
    "vf": 0.5,
    # Maximum outer topology optimization iterations.
    "max_iter": 400,#开始时调大一些，得出大概次数
    # Objective type. Supported: "Stress", "Compliance", "Volume"
    "obj_type": "Compliance",
    # Optional observed global DOF indices for U_DOF. If None, nonzero loaded DOFs are used.
    # "u_dof_indices": None,
    # Constraint tolerance for optional stopping. Reference range: 1e-4 ~ 1e-3.
    "con_tol": 1e-3,
    # Save phi every N iterations. 0 disables intermediate phi saving; final phi is always saved.
    # Reference: 0 ~ 50. Use 0 for light output, 10 for checking topology evolution.
    "save_interval": 1,

    # FEM / material parameters.
    "E_min": 1e-6,                        # Ersatz lower bound. Reference: 1e-9 ~ 1e-4.
}

cantilever = {
    "length": 8.0,        # Domain length in x direction.
    "height": 4.0,        # Domain height in y direction.
    "thickness": 0.01,    # Plate thickness.
    "nx": 200,            # Number of elements in x direction.
    "ny": 100,            # Number of elements in y direction.
    "E": 200e9,           # Solid material Young's modulus.
    "nu": 0.3,            # Poisson's ratio. Typical isotropic range: 0.2 ~ 0.35.
    "load_size": 1000.0,  # Load magnitude used by set_bc.py.
    "domain": "multi_holes",  # Initial level-set topology type used by initial_ls_func.
    "model_path": "./model_data/",    # Directory storing initial level-set network model.
    "ls_r": 0.15,          # Radius/scale parameter used by initial_ls_func for initial holes.
    # 明确一个粗网格包含的细网格节点数，输出参数设置信息
}
