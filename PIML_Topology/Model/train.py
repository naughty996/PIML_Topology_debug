# =========================================================================
# 文件名: train.py
# 功能: 顶级调度脚本。负责参数统筹、计算图迭代以及 MLOps 日志序列化。
# 模块边界: 不包含任何特定的 I/O 读取细节与可视化底层逻辑。
# =========================================================================
import os
import time
import json
from datetime import datetime
import torch
import torch.optim as optim
import numpy as np

# 导入跨文件解耦的核心模块
from network import PIMLNet, FieldAssembler
from dataloader import get_piml_dataloaders
from metrics_and_plots import get_l2_relative_error, plot_training_log, evaluate_and_plot_sample


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training initialization | Target device: {device}")

    # ================= 1. 实验全局超参数配置 =================
    # 使用字典统一管理，便于后续直接落盘为 JSON 溯源档案
    config = {
        "data_size": 50000,
        "epochs": 2000,
        "batch_size": 64,
        "initial_lr": 0.002,
        "random_seed": 2026
    }
    labels_path = f'./dataset/PIML_Labels_{config["data_size"]}_5x5.h5'

    # ================= 2. 模块初始化 =================
    # 2.1 委托 dataloader.py 准备就绪数据流
    train_loader, X_test_t, Y_192_test_t, Y_576_test_t = get_piml_dataloaders(
        data_size=config["data_size"],
        batch_size=config["batch_size"],
        test_size=0.1,
        random_state=config["random_seed"],
        device=device
    )

    # 2.2 实例化神经网络与物理规则组装器
    model = PIMLNet().to(device)
    assembler = FieldAssembler(labels_path, device)

    # 2.3 配置优化器与学习率衰减策略
    optimizer = optim.Adam(model.parameters(), lr=config["initial_lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=20, min_lr=1e-6
    )

    # 训练状态追踪容器
    history_train_loss, history_test_l2, history_lr = [], [], []

    # ================= 3. 深度学习主循环 (Training Loop) =================
    t1 = time.time()
    for epoch in range(config["epochs"]):
        model.train()
        batch_losses = []
        for bx, by_192 in train_loader:
            optimizer.zero_grad()

            # 前向传播：在 192 维降维空间内进行计算
            pred_192 = model(bx)

            # 计算批次平均相对误差并触发反向传播
            epsilon = 1e-9
            relative_error = torch.norm(pred_192 - by_192, p=2, dim=1) / (torch.norm(by_192, p=2, dim=1) + epsilon)
            loss = torch.mean(relative_error)

            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        # 记录当前 Epoch 状态并调度学习率
        avg_train_loss = np.mean(batch_losses)
        history_train_loss.append(avg_train_loss)
        scheduler.step(avg_train_loss)
        history_lr.append(optimizer.param_groups[0]['lr'])

        # 测试集验证：评估全局 576 维物理场还原后的绝对保真度
        model.eval()
        with torch.no_grad():
            preds_192 = model(X_test_t)
            preds_576 = assembler.assemble_576(preds_192)
            l2_err = get_l2_relative_error(preds_576, Y_576_test_t)
            history_test_l2.append(l2_err)

        # 终端进度监控
        if epoch % 20 == 0 or epoch == config["epochs"] - 1:
            print(
                f"Epoch {epoch:04d}/{config['epochs']} | Train Rel Loss (192d): {avg_train_loss:.6f} | Test L2 (576d): {l2_err:.5f} | LR: {history_lr[-1]:.6f}")

    train_time = time.time() - t1
    print(f'[*] Training completed in {train_time:.2f} s')

    # ================= 4. 验证与可视化委托 =================
    os.makedirs('./result_fig', exist_ok=True)
    os.makedirs('./train_models', exist_ok=True)

    # 4.1 委托 metrics_and_plots 模块绘制训练趋势图
    plot_training_log(history_train_loss, history_test_l2, history_lr, './result_fig/training_log.png')

    # 4.2 委托 metrics_and_plots 模块执行单样本一键物理考核
    evaluate_and_plot_sample(model, assembler, X_test_t, Y_576_test_t, sample_num=0, save_dir='./result_fig')

    # ================= 5. MLOps 参数与模型持久化 =================
    # 生成防冲突动态时间戳文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    lr_str = str(config["initial_lr"]).replace('.', '')
    model_name = f'PIMLNet_{int(config["data_size"] / 1000)}k_bs{config["batch_size"]}_lr{lr_str}_{timestamp}'

    # 导出核心网络权重 (.pth)
    model_save_path = f'./train_models/{model_name}.pth'
    torch.save(model.state_dict(), model_save_path)

    # 丰富元数据并导出 JSON 日志溯源档案
    config["final_train_loss"] = float(avg_train_loss)
    config["final_test_l2"] = float(l2_err)
    config["training_time_sec"] = float(train_time)

    log_save_path = f'./train_models/{model_name}_config.json'
    with open(log_save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[*] ✅ 模型权重及实验配置档案已成功入库至: ./train_models")


if __name__ == '__main__':
    main()