import os
import json
import torch
import numpy as np
import re
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 导入你的自定义模块
import dataloader
import Config

# 初始化 Config
Config = Config.benchmark_config()

# --- 1. 配置区域 ---
train_config = {
    "model_name": "svm", 
    # 可以选择 'raw', 'psd', 'de'
    # 注意：raw 数据维度很高 (62*400=24800)，SVM 跑起来会很慢，建议先跑 psd 或 de
    "feature": "psd", 
    
    # SVM 超参数
    "kernel": "rbf",   # 'linear', 'rbf', 'poly'
    "C": 1.0,          # 正则化系数
    "gamma": "scale",  # 核函数系数
    
    "seed": 42,
    "save_path": "benchmark_results_alpha_removed.json",
    "num_workers": 4
}

# --- 2. 辅助函数 ---

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# 将 DataLoader 中的数据转换为 sklearn 需要的 (N_samples, N_features) 格式
def extract_data_from_loader(loader):
    all_x = []
    all_y = []
    
    # 不需要计算梯度
    with torch.no_grad():
        for x, y in loader:
            # x shape: [Batch, 62, Time] or [Batch, 62, Freq]
            # 我们需要将其打平为 [Batch, 62 * Time]
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1).numpy() # 转为 numpy
            y_np = y.numpy()
            
            all_x.append(x_flat)
            all_y.append(y_np)
            
    # 拼接所有 batch
    X = np.concatenate(all_x, axis=0)
    Y = np.concatenate(all_y, axis=0)
    return X, Y

def update_json_results(filepath, config, results_array):
    """
    保存结果到 JSON，格式与 main.py 保持一致，方便对比
    """
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

    if feature not in data_store: data_store[feature] = {}
    if model_name not in data_store[feature]: data_store[feature][model_name] = []
    
    data_store[feature][model_name].append(results_array)

    json_text = json.dumps(data_store, indent=4) 
    # 压缩数组显示格式
    compact_json_text = re.sub(
        r'\[(?:\s*[\d. ,e+-]+\s*)+\]', 
        lambda m: '[' + re.sub(r'\s+', ' ', m.group(0)[1:-1]).strip() + ']', 
        json_text
    )

    with open(filepath, 'w') as f:
        f.write(compact_json_text)
    print(f"Results saved to {filepath}")

# --- 3. 主循环 ---
def main():
    set_seed(train_config["seed"])
    
    results = []

    # 路径选择逻辑
    if train_config["feature"] == "raw":
        base_dir = Config.watch_dir
    elif train_config["feature"] in ["psd", "de"]:
        base_dir = Config.watch_PSD_DE
    else:
        raise ValueError(f"Invalid feature: {train_config['feature']}")
        
    data_path_list = os.listdir(base_dir)
    data_path_list.sort()

    print(f"Starting SVM training on {len(data_path_list)} subjects...")
    print(f"Feature: {train_config['feature']}, Kernel: {train_config['kernel']}")

    for i, data_path in enumerate(data_path_list):
        print(f"\nSubject {i+1}/{len(data_path_list)}: {data_path}")
        
        full_path = os.path.join(base_dir, data_path)
        
        # 1. 获取 DataLoader (复用现有的 dataloader.py)
        train_loader, val_loader = dataloader.get_dataloaders(train_config["feature"], full_path)
        
        # 2. 转换数据格式 (Tensor -> Numpy Flattened)
        print("  Extracting data...")
        X_train, y_train = extract_data_from_loader(train_loader)
        X_val, y_val = extract_data_from_loader(val_loader)
        
        print(f"  Train shape: {X_train.shape}, Val shape: {X_val.shape}")

        # 3. 定义 SVM 模型
        clf = SVC(
            kernel=train_config["kernel"], 
            C=train_config["C"], 
            gamma=train_config["gamma"],
            cache_size=1000, # 增加缓存以加快速度
            random_state=train_config["seed"]
        )
        
        # 4. 训练
        print("  Training SVM...")
        clf.fit(X_train, y_train)
        
        # 5. 预测与评估
        print("  Evaluating...")
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred) * 100
        
        print(f"  Subject {i+1} Accuracy: {acc:.2f}%")
        results.append(acc)

    # 6. 保存结果
    update_json_results(train_config["save_path"], train_config, results)
    print(f"\nMean Accuracy: {np.mean(results):.2f}%")

if __name__ == "__main__":
    # 你可以在这里修改 feature 类型进行批量测试
    # 警告：raw 数据的特征维度极大，SVM 训练可能非常慢，建议优先测试 psd 或 de
    for feature in ["raw"]: 
        train_config["feature"] = feature
        main()