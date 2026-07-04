config = {
    "physics": {
        "length": 8.0,
        "height": 4.0,
        "nx": 40,
        "ny": 20,
        "nely_c": 20,
        "nelx_f": 5,
        "nely_f": 5,
        "nu": 0.3,
        "E0": 1.0,
        "Emin": 1e-6,
    },
    "darcy": {
        "Kv": 1.0,
        "epsf": 1.0e-7,
        "r_param": 0.1,
        "Dels": 2.0,
        "Pin": 1.0
    },
    "ls_network": {
        "act_func": "sine",
        "neu_in": 2,
        "neu_out": 1,
        "neu_hidden": [3, 64],
        "lr": 1e-3,
        "ls_tol": 1e-2,
        "max_epoch": 30000,
        "initial_domain": "multi_holes",
        "ls_r": 0.15,
        "H_d": 3.0,
    },
    "data": {
        "use_samples": 1000,
        "labels_path": r"D:\Study\prp\PIML_Topology_debug\PIML_Topology\Model\dataset\PIML_Labels_50000_5x5.h5",
        "piml_model_path": r"D:\Study\prp\PIML_Topology_debug\PIML_Topology\Model\train_models\PIML_50k_bs1024_lr1e-3_V2Funnel_PolarB6_0524_1838.pth",
        "init_ls_weights": "./model_data/init_topo_weights.pt"
    },
    "optimization": {
        "vf_target": 0.5,
        "max_iter": 2000,
        "obj_type": "Compliance",
        "con_tol": 1e-3,
        "opt_lr": 0.01,
        "penalty_weight": 500.0,
        "device": "cuda",
        "dtype": "float64",
        "seed": 2026,
        "save_interval": 1,
        "N_conv": 5
    }
}