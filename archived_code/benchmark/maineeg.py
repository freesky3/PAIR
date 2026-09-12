import os
import json
import random
import torch
import torch.nn as nn
import numpy as np
import re
import copy
# 假设这些是你自定义的模块
import model
import dataloader
import Config
import argparse # 必须引入这个

# 初始化 Config
Config = Config.benchmark_config()

# --- 1. 配置区域 ---
train_config = {
    "model_name": "GCN_LSTM", 
    "task": "20-c",
    "feature": "raw",
    "data": "recall",
    
    "seed": 42,
    "epochs": 100,      # 单电极任务简单且数量多，可以适当减少 epoch
    "patience": 20,     # 早停也可以适当缩短
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "save_path": "topomap_results.json", # 保存路径改为热力图专用
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}

# 辅助函数：根据字符串获取模型类
def get_model_class(name):
    if hasattr(model, name):
        return getattr(model, name)
    else:
        raise ValueError(f"Model {name} not found in model module.")

# --- 2. 设置随机种子 ---
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# --- 4. 训练与评估函数 (保持不变) ---
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)
    return train_loss / len(train_loader), 100. * correct / total

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            val_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
    return val_loss / len(val_loader), 100. * correct / total

# --- 5. JSON 结果保存逻辑 (微调以支持热力图数据) ---
def update_json_results(filepath, config, results_array):
    """
    保存结构:
    {
        "watch": {
            "20-c": {
                "GCN_LSTM": {
                    "raw": [
                        [sub1_elec1_acc, sub1_elec2_acc, ...],  <-- Subject 1 all electrodes
                        [sub2_elec1_acc, sub2_elec2_acc, ...],  <-- Subject 2 all electrodes
                        ...
                    ]
                }
            }
        }
    }
    """
    data_type = config["data"]
    task_type = config["task"]
    model_name = config["model_name"]
    feature = config["feature"]

    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                data_store = json.load(f)
            except json.JSONDecodeError:
                data_store = {}
    else:
        data_store = {}

    if data_type not in data_store: data_store[data_type] = {}
    if task_type not in data_store[data_type]: data_store[data_type][task_type] = {}
    if model_name not in data_store[data_type][task_type]: data_store[data_type][task_type][model_name] = {}
    
    if isinstance(results_array, np.ndarray):
        results_array = results_array.tolist()
        
    data_store[data_type][task_type][model_name][feature] = results_array

    json_text = json.dumps(data_store, indent=4) 
    # 压缩格式，让每一行是一个 Subject 的所有电极数据
    compact_json_text = re.sub(
        r'\[\s+([\d. ,e+-]+(?:,\s+[\d. ,e+-]+)*)\s+\]', 
        lambda m: '[' + re.sub(r'\s+', ' ', m.group(1)).strip() + ']', 
        json_text
    )

    with open(filepath, 'w') as f:
        f.write(compact_json_text)
    
    print(f"Results saved to {filepath}")


def train_one_epoch_parallel(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    # 注意：这里计算的是所有电极混合后的平均准确率，仅供参考
    correct = 0
    total = 0
    
    for x, y in train_loader:
        # x shape: [Batch, 62, T] -> [Batch * 62, 1, T]
        B, C, T = x.shape
        x = x.view(B * C, 1, T).to(device)
        
        # y shape: [Batch] -> [Batch * 62] (标签复制，因为同一个样本的所有电极标签相同)
        y = y.to(device)
        y_expanded = y.repeat_interleave(C)
        
        optimizer.zero_grad()
        output = model(x) # output: [B*62, Num_Classes]
        
        loss = criterion(output, y_expanded)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(y_expanded).sum().item()
        total += y_expanded.size(0)
        
    return train_loss / len(train_loader), 100. * correct / total

def evaluate_parallel(model, val_loader, criterion, device):
    """
    关键修改：这里不仅返回Loss，还要返回每个电极的独立准确率列表
    """
    model.eval()
    val_loss = 0
    total_samples = 0
    
    # 用于记录每个电极做对了多少个样本
    # 假设通道数是 62，我们需要动态获取 C
    correct_per_electrode = None 
    
    with torch.no_grad():
        for x, y in val_loader:
            B, C, T = x.shape
            
            # 初始化计数器 (只在第一次循环执行)
            if correct_per_electrode is None:
                correct_per_electrode = np.zeros(C)
            
            # 1. 准备输入
            # [Batch, 62, T] -> [Batch * 62, 1, T]
            x_reshaped = x.view(B * C, 1, T).to(device)
            y = y.to(device) # [Batch]
            y_expanded = y.repeat_interleave(C) # [Batch * 62]
            
            # 2. 前向传播
            output = model(x_reshaped) # [B*62, Num_Classes]
            loss = criterion(output, y_expanded)
            val_loss += loss.item()
            
            # 3. 计算准确率 (核心步骤：还原维度)
            _, predicted = output.max(1) # [B*62]
            
            # 将预测结果变回 [Batch, 62]
            predicted_reshaped = predicted.view(B, C)
            
            # y 变回 [Batch, 1] 以便广播对比
            y_reshaped = y.view(B, 1)
            
            # 比较：得到一个 [Batch, 62] 的 True/False 矩阵
            matches = predicted_reshaped.eq(y_reshaped).cpu().numpy()
            
            # 按列求和（即求每个电极在当前 Batch 中预测对了多少个）
            correct_per_electrode += matches.sum(axis=0)
            
            total_samples += B

    # 计算每个电极的最终准确率
    acc_per_electrode = (correct_per_electrode / total_samples) * 100.0
    mean_val_loss = val_loss / len(val_loader)
    
    return mean_val_loss, acc_per_electrode


def run_single_electrode_analysis_parallel():
    set_seed(train_config["seed"])
    
    # ... (参数设置 T, out_dim 等代码保持不变) ...
    # 假设已经获取了 T 和 out_dim
    if train_config["feature"] == "raw" and train_config["data"] == "watch": T = 400 
    elif train_config["feature"] == "raw" and train_config["data"] == "recall": T = 600
    else: T = 5 # PSD/DE
    
    task_dim_map = {"20-c": 20, "4-c": 4, "fast_slow": 2, "color": 5, "number": 3, "face": 2, "human": 2}
    out_dim = task_dim_map.get(train_config["task"])

    # 路径配置 (保持不变)
    if train_config["feature"] == "raw": 
        if train_config["data"] == "watch": base_dir = Config.watch_dir
        elif train_config["data"] == "recall": base_dir = Config.recall_dir
    else: 
        base_dir = Config.watch_PSD_DE # 简写，根据你的逻辑补全
        
    data_path_list = sorted(os.listdir(base_dir))
    all_subjects_results = [] 
    
    print(f"Starting PARALLEL Single Electrode Analysis on {len(data_path_list)} subjects...")

    for i, data_path in enumerate(data_path_list):
        print(f"\n=== Processing Subject {i+1}/{len(data_path_list)}: {data_path} ===")
        
        full_path = os.path.join(base_dir, data_path)
        
        # 获取原始 Full Dataset (Batch, 62, T)
        train_loader_full, val_loader_full = dataloader.get_dataloaders(full_path, train_config["feature"], train_config["task"])
        
        # --- 模型初始化 ---
        ModelClass = get_model_class(train_config["model_name"])
        
        # 强制 input_dim 适配单通道 (C=1)
        # 即使数据是 62 通道进来，我们通过 view 变成了 1 通道给模型看
        current_C = 1 
        current_input_dim = 1 * T
        
        if train_config["model_name"] == "mlpnet":
            model_instance = ModelClass(out_dim=out_dim, input_dim=current_input_dim)
        elif train_config["model_name"] == "glfnet_mlp":
            model_instance = ModelClass(out_dim=out_dim, emb_dim=128, input_dim=current_input_dim)
        else:
            # 这里的 C=1 非常重要，告诉模型它只接受单通道
            model_instance = ModelClass(out_dim=out_dim, C=current_C, T=T)
        
        model_instance.to(train_config["device"])
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # 注意学习率可能需要调整，因为 Batch Size 变大了 62 倍
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=train_config["learning_rate"], weight_decay=train_config["weight_decay"])
        
        best_mean_acc = 0.0
        best_electrode_accuracies = None # 用来存这一轮最好的那个由62个数值组成的数组
        patience_counter = 0
        
        # --- 训练循环 (只跑一次，而不是 62 次) ---
        print(f"  Training shared model on all electrodes simultaneously...")
        for epoch in range(train_config["epochs"]):
            t_loss, t_acc = train_one_epoch_parallel(model_instance, train_loader_full, criterion, optimizer, train_config["device"])
            v_loss, v_acc_array = evaluate_parallel(model_instance, val_loader_full, criterion, train_config["device"])
            
            # 使用所有电极的平均准确率来决定 Early Stopping
            mean_val_acc = np.mean(v_acc_array)
            
            if mean_val_acc > best_mean_acc:
                best_mean_acc = mean_val_acc
                best_electrode_accuracies = v_acc_array # 保存当前的 62 个准确率
                patience_counter = 0
                # print(f"    Epoch {epoch}: New Best Mean Acc: {mean_val_acc:.2f}%")
            else:
                patience_counter += 1
                
            if patience_counter >= train_config["patience"]:
                print(f"    Early stopping at epoch {epoch}")
                break
        
        # 训练结束，保存这个 Subject 的 62 个电极数据
        print(f"  [Sub {i+1}] Finished. Best Mean Acc: {best_mean_acc:.2f}%")
        all_subjects_results.append(best_electrode_accuracies)

        del model_instance, optimizer
        torch.cuda.empty_cache()
        
        # 实时保存
        update_json_results(train_config["save_path"], train_config, np.array(all_subjects_results))


if __name__ == "__main__":
    # 1. 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Task name (e.g., 20-c)')
    parser.add_argument('--data', type=str, default='watch', help='Data type (watch/recall)')
    parser.add_argument('--feature', type=str, default='raw', help='Feature type')
    parser.add_argument('--model', type=str, default='Conformer', help='Model name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (e.g., cuda:0, cuda:1)')
    
    args = parser.parse_args()

    # 2. 将参数覆盖到 Config 中
    train_config["task"] = args.task
    train_config["data"] = args.data
    train_config["feature"] = args.feature
    train_config["model_name"] = args.model
    train_config["device"] = args.device
    
    # 3. 【关键】修改保存路径，防止多进程写入冲突
    # 每个任务存一个单独的文件，例如: result_20-c_watch_raw.json
    filename = f"result_{args.task}_{args.data}_{args.feature}.json"
    train_config["save_path"] = os.path.join("results_buffer", filename) # 建议存到一个子文件夹
    
    # 确保文件夹存在
    os.makedirs("results_buffer", exist_ok=True)

    print(f"--- Process Start: {args.task} | {args.data} | GPU: {args.device} ---")
    
    # 4. 运行主逻辑 (带有显存保护)
    try:
        run_single_electrode_analysis_parallel()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"[OOM Error] Task {args.task} failed on {args.device}. Try reducing batch size.")
        else:
            raise e