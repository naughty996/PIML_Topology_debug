config = {
    "data": {
        "data_size": 50000,
        "use_samples": 1000,
        "random_seed": 2026,
        "dataset_path": r"D:\Study\prp\PIML_Topology_debug\PIML_Topology\Model\dataset\PIML_Dataset_50000_5x5.h5",
        "labels_path": r"D:\Study\prp\PIML_Topology_debug\PIML_Topology\Model\dataset\PIML_Labels_50000_5x5.h5",
        "model_path": r"D:\Study\prp\PIML_Topology_debug\PIML_Topology\Model\train_models\PIML_50k_bs1024_lr1e-3_V2Funnel_PolarB6_0524_1838.pth"
    },

    "physics": {
        "nu": 0.3,
        "E0": 1.0,
        "Emin": 1.0e-9,
        "nelx_f": 5,
        "nely_f": 5
    },

    "darcy": {
        "Kv": 1.0,
        "epsf": 1.0e-7,
        "r_param": 0.1,
        "Dels": 2.0,
        "Pin": 1.0
    }
}