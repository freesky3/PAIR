import os
import json
import random
import torch
import torch.nn as nn
import numpy as np
import re
# 假设这些是你自定义的模块
import model
import dataloader
import Config

# 初始化 Config
Config = Config.benchmark_config()

# --- 1. 配置区域 ---
'''
model_name you can choose from: 
for raw data: shallownet, deepnet, eegnet, conformer, tsconv, glfnet
for psd, de data: mlpnet, glfnet_mlp
'''
train_config = {
    "model_name": "conformer", 
    "feature": "raw",
    
    "seed": 42,
    "epochs": 400,
    "patience": 40,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "save_path": "benchmark_results_alpha_removed.json", # 改为 json 后缀
    "device": "cuda:2" if torch.cuda.is_available() else "cpu",
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

set_seed(train_config["seed"])



# 任务输出维度
out_dim = 2


# --- 4. 训练与评估函数 ---
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = x.float()
        optimizer.zero_grad() # 别忘了清空梯度
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
    with torch.no_grad(): # 验证时不需要梯度
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            x = x.float()
            output = model(x)
            loss = criterion(output, y)
            val_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
    return val_loss / len(val_loader), 100. * correct / total

# --- 5. JSON 结果保存逻辑 ---
def update_json_results(filepath, config, results_array):
    """
    结构设计:
    {
        "feature": {
            "model_name": [acc_sub1, acc_sub2, ...]
        }
    }
    """
    model_name = config["model_name"]
    feature = config["feature"]

    # 1. 读取现有数据
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                data_store = json.load(f)
            except json.JSONDecodeError:
                data_store = {}
    else:
        data_store = {}

    # 2. 构建层级 (如果不存在则创建)
    if feature not in data_store: data_store[feature] = {}
    if model_name not in data_store[feature]: data_store[feature][model_name] = []
    
    # 3. 存入数据 (以 feature 为最后一级 key)
    data_store[feature][model_name].append(results_array)

    # 4. 写入文件
    # 修复：先将字典转为字符串
    json_text = json.dumps(data_store, indent=4) 

    compact_json_text = re.sub(
        r'\[(?:\s*[\d. ,e+-]+\s*)+\]', 
        lambda m: '[' + re.sub(r'\s+', ' ', m.group(0)[1:-1]).strip() + ']', 
        json_text
    )

    with open(filepath, 'w') as f:
        f.write(compact_json_text)
    
    print(f"Results saved to {filepath} under [{feature}][{model_name}]")

# --- 6. 主循环 ---
def main():
    results = []

    # --- 3. 动态参数设置 ---
    if train_config["feature"] == "raw":
        T = 400 
    elif train_config["feature"] in ["psd", "de"]:
        T = 4
    else:
        raise ValueError(f"Invalid feature: {train_config['feature']}")
    
    # 修复路径逻辑
    if train_config["feature"] == "raw":
        base_dir = Config.watch_dir
    elif train_config["feature"] in ["psd", "de"]:
        base_dir = Config.watch_PSD_DE
    else:
        raise ValueError(f"Invalid feature: {train_config['feature']}")
        
    data_path_list = os.listdir(base_dir)
    # 排序以保证结果顺序一致
    data_path_list.sort() 

    print(f"Starting training on {len(data_path_list)} subjects...")

    for i, data_path in enumerate(data_path_list):
        print(f"Subject {i+1}/{len(data_path_list)}: {data_path}")
        
        # 加载数据 (使用 base_dir)
        full_path = os.path.join(base_dir, data_path)
        train_loader, val_loader = dataloader.get_dataloaders(train_config["feature"], full_path)
        
        # 实例化模型
        ModelClass = get_model_class(train_config["model_name"])
        input_dim = 62*T
        if train_config["model_name"] == "mlpnet":
            model_instance = ModelClass(out_dim=out_dim, input_dim=input_dim)
        elif train_config["model_name"] == "glfnet_mlp": # glfnet_mlp
            # glfnet_mlp 需要 (out_dim, emb_dim, input_dim)
            model_instance = ModelClass(out_dim=out_dim, emb_dim=128, input_dim=input_dim)
        else:
            model_instance = ModelClass(out_dim=out_dim, C=62, T=T)
        model_instance.to(train_config["device"])
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=train_config["learning_rate"], weight_decay=train_config["weight_decay"])
        
        # 早停相关变量
        best_val_acc = 0.0 # 通常我们保存最佳准确率
        patience_counter = 0
        best_epoch = 0

        for epoch in range(train_config["epochs"]):
            train_loss, train_acc = train_one_epoch(model_instance, train_loader, criterion, optimizer, train_config["device"])
            val_loss, val_acc = evaluate(model_instance, val_loader, criterion, train_config["device"])
            
            # 监控验证集准确率 (也可以改为监控 loss)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                # 可以在这里保存模型权重 save_state_dict
            else:
                patience_counter += 1
                
            if patience_counter >= train_config["patience"]:
                print(f"  Early stopping at epoch {epoch}. Best Acc: {best_val_acc:.2f}%")
                break
        
        results.append(best_val_acc)

    # 训练结束后，保存结果到 JSON
    update_json_results(train_config["save_path"], train_config, results)

if __name__ == "__main__":
    for model_name in ["glfnet_mlp"]:
        train_config["model_name"] = model_name
        for feature in ["psd"]:
            train_config["feature"] = feature
            main()