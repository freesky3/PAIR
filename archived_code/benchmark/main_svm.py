import os
import json
import numpy as np
import torch
import re
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# 引入现有的 dataloader 和 Config
import dataloader
import Config

# 初始化 Config
Config = Config.benchmark_config()

# --- 1. 配置区域 ---
'''
支持的 task: '20-c', '4-c', 'color', 'fast_slow', 'number', 'face', 'human'
支持的 feature: 'raw', 'psd', 'de'
'''
train_config = {
    "model_name": "svm", # 模型名称，用于保存结果
    "task": "20-c",          # 任务类型
    "feature": "raw",        # 特征类型
    "data": "recall",        # watch 或 recall
    
    "seed": 42,
    "pca_variance": None, # 0.95,    # PCA 保留的方差比例 (0.95 表示保留95%的信息)
    "svm_C": 1.0,            # SVM 正则化系数
    "svm_kernel": "rbf",     # 核函数: 'linear', 'rbf', 'poly'
    "save_path": "benchmark_results.json", 
}

# --- 2. 设置随机种子 ---
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # sklearn 的随机性通常由 random_state 控制，在模型初始化时传入

set_seed(train_config["seed"])

# --- 3. 辅助函数：将 DataLoader 转换为 sklearn 可用的 numpy 数组 ---
def extract_features_from_loader(loader):
    """
    从 DataLoader 中提取所有数据并展平。
    深度学习 DataLoader 输出: (Batch, Channel, Time)
    SVM 需要: (Total_Samples, Channel * Time)
    """
    all_x = []
    all_y = []
    
    # 只需要遍历一次 loader 即可获取所有数据
    for x, y in loader:
        # x shape: [batch_size, 62, T] or [batch_size, 62, 5] (psd/de)
        # 展平为 [batch_size, features]
        x_flat = x.view(x.size(0), -1).numpy()
        y_np = y.numpy()
        
        all_x.append(x_flat)
        all_y.append(y_np)
        
    # 拼接所有 batch
    X = np.concatenate(all_x, axis=0)
    Y = np.concatenate(all_y, axis=0)
    
    return X, Y

# --- 4. JSON 结果保存逻辑 (复用 main.py 逻辑) ---
def update_json_results(filepath, config, results_array):
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

    # 紧凑格式写入
    json_text = json.dumps(data_store, indent=4) 
    compact_json_text = re.sub(
        r'\[(?:\s*[\d. ,e+-]+\s*)+\]', 
        lambda m: '[' + re.sub(r'\s+', ' ', m.group(0)[1:-1]).strip() + ']', 
        json_text
    )

    with open(filepath, 'w') as f:
        f.write(compact_json_text)
    
    print(f"Results saved to {filepath} under [{data_type}][{task_type}][{model_name}][{feature}]")

# --- 5. 主循环 ---
def main():
    results = []
    
    # 确定数据目录
    if train_config["data"] == "watch":
        base_dir = Config.watch_dir
    elif train_config["data"] == "recall":
        base_dir = Config.recall_dir
    else:
        raise ValueError(f"Invalid data type: {train_config['data']}")
        
    data_path_list = os.listdir(base_dir)
    data_path_list.sort() 

    print(f"Starting SVM training on {len(data_path_list)} subjects...")
    print(f"Config: Task={train_config['task']}, Feature={train_config['feature']}, "
          f"PCA={train_config['pca_variance']}, Kernel={train_config['svm_kernel']}")

    for i, data_path in enumerate(data_path_list):
        # 1. 加载数据 (使用 dataloader 保证划分一致)
        full_path = os.path.join(base_dir, data_path)
        
        # 获取 PyTorch DataLoader
        train_loader, val_loader = dataloader.get_dataloaders(
            full_path, 
            train_config["feature"], 
            train_config["task"]
        )
        
        # 2. 转换为 Numpy 格式
        X_train, y_train = extract_features_from_loader(train_loader)
        X_test, y_test = extract_features_from_loader(val_loader)
        
        # 3. 构建 SVM Pipeline
        # Raw Data 维度极高 (62*400=24800)，必须使用 PCA 降维，否则维度灾难且速度极慢
        # PSD/DE Data 维度较低 (62*5=310)，PCA 作用较小但加上也无妨
        
        pipeline = make_pipeline(
            PCA(n_components=train_config["pca_variance"], random_state=train_config["seed"]), # 降维
            SVC(
                kernel=train_config["svm_kernel"], 
                C=train_config["svm_C"], 
                gamma='scale', 
                random_state=train_config["seed"],
                class_weight='balanced' # 解决样本不平衡问题
            )
        )
        
        # 4. 训练与预测
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100.0
        except Exception as e:
            print(f"Error on Subject {i+1}: {e}")
            acc = 0.0
            
        print(f"Subject {i+1}/{len(data_path_list)} ({data_path}): Acc = {acc:.2f}%")
        results.append(acc)

    # 5. 保存结果
    avg_acc = np.mean(results)
    print(f"\nAverage Accuracy: {avg_acc:.2f}%")
    update_json_results(train_config["save_path"], train_config, results)

if __name__ == "__main__":
    # 可以在这里写循环批量跑实验
    # 示例：跑 recall 数据下的 4分类任务
    for data in ["watch", "recall"]:
        for task in ["20-c", "4-c", "fast_slow", "color", "number", "face", "human"]:
            for feature in ["raw", "psd", "de"]:
                train_config["data"] = data
                train_config["task"] = task
                train_config["feature"] = feature
                print(f"Running SVM training on {data} {task} {feature}...")
                main()