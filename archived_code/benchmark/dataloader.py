# dataloader.py
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from Config import benchmark_config

config = benchmark_config()

class UnifiedDataset(Dataset):
    def __init__(self, data_path, feature_type=None, task_type=None, train_or_test=None):
        """
        feature_type: 'raw', 'psd', 'de'
        task_type: '20-c', '4-c', 'color', 'fast_slow', 'number', 'face', 'human'

        data_shape:
        watch_cleaned: [5, 50, 62, 400]
        recall_cleaned: [5, 50, 62, 600]
        watch_PSD_DE: [2, 5, 50, 62, 5]
        recall_PSD_DE: [2, 5, 50, 62, 5]
        """
        self.feature_type = feature_type
        self.task_type = task_type
        self.train_or_test = train_or_test
        # 1. 加载数据
        raw_data = np.load(data_path)
        self.data = torch.from_numpy(raw_data).float()
        
        # 2. 处理特征类型 (对应表格的行)
        if 'PSD_DE' in data_path:
            # 原始是 [2, 5, 50, 62, 5]
            if feature_type == 'psd':
                self.data = self.data[0].reshape(250, 62, 5) # 取前5个特征
            elif feature_type == 'de':
                self.data = self.data[1].reshape(250, 62, 5) # 取后5个特征
        else:
            self.data = self.data.reshape(250, 62, -1)

        # 3. 处理标签 (对应表格的列)
        self.GT_label = torch.from_numpy(np.load(config.GT_label)).long().reshape(-1)
        self.optical_flow = torch.from_numpy(np.load(config.optical_flow_score)).float()
        self.video_features = np.load(config.video_features, allow_pickle=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        x = self.data[item]
        mean = x.mean()
        std = x.std()
        x = (x - mean) / (std + 1e-6)
        if self.train_or_test == 'train': # 仅针对难任务
            noise = torch.randn_like(x) * 0.05 # 强度 0.05
            x = x + noise
        # 根据任务类型返回 y
        if self.task_type == '20-c':
            y = self.GT_label[item]-1
        elif self.task_type == '4-c':
            original = self.GT_label[item]-1
            # 0-19 -> 0-3
            y = original // 5
        elif self.task_type == 'fast_slow':
            score = self.optical_flow[item]
            y = 1 if score > config.optical_flow_threshold else 0
        elif self.task_type == 'color':
            original_color = self.video_features[item]["main_color"]
            new_color = config.map_color[original_color]
            y = config.color2num[new_color]
        elif self.task_type == 'number':
            count = self.video_features[item]["object_count"]
            y = (count > 1) + (count > 4)
        elif self.task_type == 'face':
            y = self.video_features[item]["has_face"]
        elif self.task_type == 'human':
            y = self.video_features[item]["has_human"]

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        return x, y.long()



def get_dataloaders(data_path: str, feature_type: str, task_type: str):
    """Create and return train, validation data loader"""
    train_dataset = UnifiedDataset(data_path, feature_type, task_type, 'train')
    valid_dataset = UnifiedDataset(data_path, feature_type, task_type, 'test')
    total_len = len(train_dataset)
    train_size = int(config.train_ratio * len(train_dataset))  # 250 * 0.8 = 200

    indices = list(range(total_len))
    random.shuffle(indices)

    # indices = list(range(total_len))
    train_dataset = Subset(train_dataset, indices[:train_size])
    valid_dataset = Subset(valid_dataset, indices[train_size:])

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False,
        num_workers=config.num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers
    )

    # print(f"Created {len(train_loader)} training batches, {len(valid_loader)} validation batches.")
    return train_loader, valid_loader

