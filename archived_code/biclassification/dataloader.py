# dataloader.py
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, Subset
from Config import benchmark_config
from scipy.signal import butter, filtfilt

def remove_alpha_raw(data, fs=200):
    # 设计一个 8-13 Hz 的带阻滤波器
    low = 8
    high = 13
    nyq = 0.5 * fs
    low_cut = low / nyq
    high_cut = high / nyq
    b, a = butter(4, [low_cut, high_cut], btype='bandstop')
    return filtfilt(b, a, data, axis=-1)

config = benchmark_config()


def raw_process(path: str, watch_or_recall: str):
    data = np.load(path)
    data = torch.from_numpy(data).float()
    data = data.reshape(250, 62, -1)

    if watch_or_recall == 'recall':
        data = data[..., 100:500].clone()
        
    return data

def psd_de_process(path: str, feature_type: str):
    data = np.load(path)
    data = torch.from_numpy(data).float()
    data = data.permute(1, 2, 3, 0, 4).reshape(250, 62, 10)
    
    # === 修改处：加上 .clone() ===
    if feature_type == 'psd':
        data = data[..., :5].clone()
    else:
        data = data[..., 5:].clone()
        
    return data

class UnifiedDataset(Dataset):
    def __init__(self, feature_type, data_path: str, remove_alpha=True):
        """
        feature_type: 'raw', 'psd', 'de'
        data_shape:
        watch_cleaned: [5, 50, 62, 400]
        recall_cleaned: [5, 50, 62, 600]
        watch_PSD_DE: [2, 5, 50, 62, 5]
        recall_PSD_DE: [2, 5, 50, 62, 5]
        """
        self.feature_type = feature_type
        self.data_path = data_path
        self.remove_alpha = remove_alpha
        if feature_type == 'raw':
            self.watch_folder = self.data_path
            self.recall_folder = self.watch_folder.replace('watch_cleaned', 'recall_cleaned')
            self.watch_data = raw_process(self.watch_folder, 'watch')
            self.recall_data = raw_process(self.recall_folder, 'recall')
            if self.remove_alpha:
                self.watch_data = remove_alpha_raw(self.watch_data)
                self.recall_data = remove_alpha_raw(self.recall_data)
            self.len_watch = len(self.watch_data)
            self.len_recall = len(self.recall_data)

        elif feature_type == 'psd' or feature_type == 'de':
            self.watch_folder = self.data_path
            self.recall_folder = self.watch_folder.replace('watch_PSD_DE', 'recall_PSD_DE')
            self.watch_data = psd_de_process(self.watch_folder, feature_type)
            self.recall_data = psd_de_process(self.recall_folder, feature_type)
            if self.remove_alpha:
                self.watch_data = np.delete(self.watch_data, 2, axis=-1)
                self.recall_data = np.delete(self.recall_data, 2, axis=-1)
            self.len_watch = len(self.watch_data)
            self.len_recall = len(self.recall_data)
        

    def __len__(self):
        return self.len_watch + self.len_recall

    def __getitem__(self, item):
        if item < self.len_watch:
            x = self.watch_data[item]
            mean = x.mean()
            std = x.std()
            x = (x - mean) / (std + 1e-6)
            y = 0 # watch
        else:
            x = self.recall_data[item - self.len_watch]
            mean = x.mean()
            std = x.std()
            x = (x - mean) / (std + 1e-6)
            y = 1 # recall
        return x, y



def get_dataloaders(feature_type: str, data_path: str):
    dataset = UnifiedDataset(feature_type, data_path)
    
    # 1. 获取两类数据的边界
    len_watch = dataset.len_watch
    len_recall = dataset.len_recall
    
    # 2. 分别生成两部分的索引
    watch_indices = list(range(len_watch))
    recall_indices = list(range(len_watch, len_watch + len_recall))
    
    # 3. 计算每一部分各自的切分点
    watch_split = int(config.train_ratio * len_watch)
    recall_split = int(config.train_ratio * len_recall)
    
    # 4. 组合训练集索引和验证集索引
    # 训练集：watch的前80% + recall的前80%
    train_indices = watch_indices[:watch_split] + recall_indices[:recall_split]
    # 验证集：watch的后20% + recall的后20%
    valid_indices = watch_indices[watch_split:] + recall_indices[recall_split:]
    
    # 5. 如果需要，可以把训练集索引打乱（防止训练时先全是watch后全是recall）
    random.shuffle(train_indices)

    # 6. 创建 Subset
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    return train_loader, valid_loader