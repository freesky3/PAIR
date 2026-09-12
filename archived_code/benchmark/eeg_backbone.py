import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
import os
import logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append('/userhome2/zhoutianyi/BrainDecoding/ATS/base')
from distangle import *
from einops import rearrange

class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return  x + self.f(x)
# UBP    
class EEGProjectLayer(nn.Module):
    def __init__(self,  z_dim, c_num, timesteps, drop_proj=0.3):
        super(EEGProjectLayer, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1]-self.timesteps[0])
        proj_dim = 1024

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim))
        self.logit_scale_fg = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus_fg = nn.Softplus()
        
    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x
    
class BaseModel(nn.Module):
    def __init__(self,  z_dim, c_num, timesteps, embedding_dim = 1440):
        super(BaseModel, self).__init__()

        self.backbone = None
        self.project = nn.Sequential(
            FlattenHead(),
            nn.Linear(embedding_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(0.5))),
            nn.LayerNorm(z_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self,x):
        ### batch_size, c_num, timesteps (200, 8, 250)
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.project(x)
        return x

class Shallownet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.Dropout(0.5),
            )
    
class Deepnet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps,embedding_dim = 1400)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 25, (1, 10), (1, 1)),
                nn.Conv2d(25, 25, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(25),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(25, 50, (1, 10), (1, 1)),
                nn.BatchNorm2d(50),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(50, 100, (1, 10), (1, 1)),
                nn.BatchNorm2d(100),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(100, 200, (1, 10), (1, 1)),
                nn.BatchNorm2d(200),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
            )
        
class EEGnet(BaseModel):
    def __init__(self,  z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim = 1248)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 8, (1, 64), (1, 1)),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
                nn.Conv2d(16, 16, (1, 16), (1, 1)),
                nn.BatchNorm2d(16), 
                nn.ELU(),
                # nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout2d(0.5)
            )
        
class TSconv(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Dropout(0.5),
            )




# Our shared temporal attention encoder (STAE)
class Shared_Temporal_Attention_Encoder(nn.Module):

    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.3):
        super().__init__()

        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.input_dim = self.c_num * (self.timesteps[1]-self.timesteps[0])
        self.time_len = timesteps[1] - timesteps[0]
        proj_dim = 1024
        self.temporal_attention = nn.Parameter(torch.ones(self.time_len))

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim)
        )
        
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.softplus = nn.Softplus()
        self.logit_scale_fg = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.softplus_fg = nn.Softplus()
        self.logit_scale_bg = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.softplus_bg = nn.Softplus()
        
    def forward(self, x):

        batch_size = x.shape[0]
        time_len = self.timesteps[1] - self.timesteps[0]
        
        x_reshaped = x.view(batch_size, self.c_num, time_len)

        temporal_weights = torch.softmax(self.temporal_attention, dim=0)
        x_weighted = x_reshaped * temporal_weights.unsqueeze(0).unsqueeze(0)
        
        x_flat = x_weighted.view(batch_size, self.input_dim)

        return self.model(x_flat)


class SpatioTemporalAttentionEEGEncoder(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.3):
        super().__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.time_len = timesteps[1] - timesteps[0]
        self.input_dim = self.c_num * self.time_len
    
        self.channel_attention = nn.Parameter(torch.ones(self.c_num))
        
        self.temporal_attention = nn.Parameter(torch.ones(self.time_len))
    
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(z_dim)
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        batch_size = x.shape[0]
        

        x_reshaped = x.view(batch_size, self.c_num, self.time_len)
        
  
        channel_weights = torch.softmax(self.channel_attention, dim=0)
        x_channel_weighted = x_reshaped * channel_weights.unsqueeze(0).unsqueeze(-1)
        

        temporal_weights = torch.softmax(self.temporal_attention, dim=0)
        x_full_weighted = x_channel_weighted * temporal_weights.unsqueeze(0).unsqueeze(0)

        x_flat = x_full_weighted.view(batch_size, self.input_dim)
        return self.model(x_flat)

class TemporalSpatioAttentionEEGEncoder(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.3):
        super().__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.time_len = timesteps[1] - timesteps[0]
        self.input_dim = self.c_num * self.time_len
        

        self.channel_attention = nn.Parameter(torch.ones(self.c_num))
    
        self.temporal_attention = nn.Parameter(torch.ones(self.time_len))
        

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(z_dim)
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        batch_size = x.shape[0]
        

        x_reshaped = x.view(batch_size, self.c_num, self.time_len)

        temporal_weights = torch.softmax(self.temporal_attention, dim=0)
        x_time_weighted = x_reshaped * temporal_weights.unsqueeze(0).unsqueeze(0)
        

        channel_weights = torch.softmax(self.channel_attention, dim=0)
        x_full_weighted = x_time_weighted * channel_weights.unsqueeze(0).unsqueeze(-1)
        
        x_flat = x_full_weighted.view(batch_size, self.input_dim)
        return self.model(x_flat)




class SpatioTemporalAttention2DEEGEncoder(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.3):
        super().__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.time_len = timesteps[1] - timesteps[0]
        self.input_dim = self.c_num * self.time_len
        

        self.spatiotemporal_attention = nn.Parameter(torch.ones(self.c_num, self.time_len))
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(z_dim)
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
        
    def forward(self, x):

        batch_size = x.shape[0]
        
        x_reshaped = x.view(batch_size, self.c_num, self.time_len)
        
        attention_weights = torch.softmax(self.spatiotemporal_attention.view(-1), dim=0).view(self.c_num, self.time_len)
        x_weighted = x_reshaped * attention_weights.unsqueeze(0)
        
        x_flat = x_weighted.view(batch_size, self.input_dim)
        return self.model(x_flat)

class STAE_Dual_Template(nn.Module):
    def __init__(self, z_dim=768, c_num=17, timesteps=[0,250], drop_proj=0.3):
        super().__init__()

        self.z_dim = z_dim
        self.c_num = c_num
        self.time_len = timesteps[1] - timesteps[0]
        self.input_dim = self.c_num * self.time_len
        
        # === 核心：两个独立的时间模板 (Templates) ===
        # nn.Parameter 意味着它们是固定的“知识”，而不是随输入变化的“猜测”
        # 初始化：
        # 我们用 randn 初始化，模型会自己在训练中把重要的时间点“顶”上去
                # 修改 __init__ 中的初始化逻辑
        # 使用 uniform 或 normal，保持均值在 0 附近，方差小一点
        self.fg_time_template = nn.Parameter(torch.empty(self.time_len))
        nn.init.normal_(self.fg_time_template, mean=0, std=0.01) # Sigmoid 结果都在 0.5 附近徘徊

        self.bg_time_template = nn.Parameter(torch.empty(self.time_len))
        nn.init.normal_(self.bg_time_template, mean=0, std=0.01)
        # === 双流处理 (Dual Stream) ===
        proj_dim = z_dim
        
        # FG 专用通道 (只看物体时间点)
        self.fg_net = nn.Sequential(
            nn.Linear(self.input_dim, proj_dim),
            self._make_res_block(proj_dim, drop_proj),
            nn.LayerNorm(proj_dim)
        )
        
        # BG 专用通道 (只看环境时间点)
        self.bg_net = nn.Sequential(
            nn.Linear(self.input_dim, proj_dim),
            self._make_res_block(proj_dim, drop_proj),
            nn.LayerNorm(proj_dim)
        )
        
        self.logit_scale_fg = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_bg = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus_fg = nn.Softplus()
        self.softplus_bg = nn.Softplus()

    def _make_res_block(self, dim, drop):
        return ResidualAdd(nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(drop),
        ))

    def forward(self, x):
        # x: (B, C*T) or (B, C, T)
        batch_size = x.shape[0]
        
        # 1. 准备数据
        if x.dim() == 2:
            x_reshaped = x.view(batch_size, self.c_num, self.time_len)
        else:
            x_reshaped = x

        # 2. 激活模板 (Template Activation)
        # Sigmoid: 每个时间点独立打分 (0~1)
        # 如果 sigmoid(t) 接近 1，说明这个时间点对该任务重要
        raw_scores = torch.stack([self.fg_time_template, self.bg_time_template], dim=0)
        
        # 2. 关键步骤：使用 Softmax 在维度 0 (类别维度) 上进行竞争
        # 这样保证了对于每个时间点 t: fg_mask[t] + bg_mask[t] = 1
        # 如果模型认为某时刻背景特征很强，bg_mask 变大，fg_mask 自然就被压制了
        attention_weights = F.softmax(raw_scores, dim=0) # shape: [2, Time]
        
        fg_mask = attention_weights[0] # [Time]
        bg_mask = attention_weights[1] # [Time]
        
        # 3. 应用模板 (Filtering)
        # 就像用模具扣饼干一样，只保留模具形状内的部分
        # (B, C, T) * (1, 1, T)
        x_fg = x_reshaped * fg_mask.view(1, 1, -1)
        x_bg = x_reshaped * bg_mask.view(1, 1, -1)
        
        # 4. 展平并映射
        # 此时的 x_fg 已经剔除了背景时间点的信息
        # 此时的 x_bg 已经剔除了物体时间点的信息
        fg_clip = self.fg_net(x_fg.view(batch_size, -1))
        bg_clip = self.bg_net(x_bg.view(batch_size, -1))
        
        # 返回 mask 是为了可视化验证！
        return fg_clip, bg_clip, fg_mask, bg_mask


########## CBAM  + TSConv###########
class TSConv_attention(nn.Module):
    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=17):
        super().__init__()
        
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (channels_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        
        emb_size = 40
        self.projection = nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1))
        
        self.embedding_dim = (math.ceil((((eeg_sample_points - 25) + 1) - 51) / 5.) + 1) * 40
    
    def forward(self, x:Tensor):
        x = x.unsqueeze(dim=1)
        x = self.tsconv(x)
        x = self.projection(x)
        x = x.flatten(2).transpose(1,2)
        return x

class TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, in_chans=1, out_chans=8, config=None):
        '''
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x

class TConv_CBAM(nn.Module):
    def __init__(self,z_dim=768, c_num=17, timesteps=[0,250], drop_proj=0.3):
        super().__init__()
        self.eeg_encoder = TSConv_attention() #[b,17,256]
        self.cbam = CBAM(channel=c_num)
        embedding_dim = self.eeg_encoder.embedding_dim
        feature_dim = z_dim
        self.proj_bg = nn.Sequential(
                nn.Linear(embedding_dim, feature_dim),
                # ResidualAdd(nn.Sequential(
                #     nn.GELU(),
                #     nn.Linear(feature_dim, feature_dim),
                #     nn.Dropout(drop_proj),
                # )),
                nn.LayerNorm(feature_dim),
            )
        self.proj_fg = nn.Sequential(
                nn.Linear(embedding_dim, feature_dim),
                # ResidualAdd(nn.Sequential(
                #     nn.GELU(),
                #     nn.Linear(feature_dim, feature_dim),
                #     nn.Dropout(drop_proj),
                # )),
                nn.LayerNorm(feature_dim),
            )
        self.logit_scale_fg = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_bg = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus_fg = nn.Softplus()
        self.softplus_bg = nn.Softplus()

    def forward(self,x):
        x = self.eeg_encoder(x)
        # x = x.transpose(1,2)
        fg,bg = self.cbam(x)
        fg = fg.flatten(1)
        bg = bg.flatten(1)
        fg = self.proj_fg(fg)
        bg = self.proj_bg(bg)
        return fg,bg

class TSConv_CBAM(nn.Module):
    def __init__(self,z_dim=768, c_num=17, timesteps=[0,250], drop_proj=0.3):
        super().__init__()
        self.eeg_encoder = TSConv_attention() #[b,17,256]
        self.cbam = CBAM(channel=40)
        embedding_dim = self.eeg_encoder.embedding_dim
        feature_dim = z_dim
        self.proj_bg = nn.Sequential(
                nn.Linear(embedding_dim, feature_dim),
                # ResidualAdd(nn.Sequential(
                #     nn.GELU(),
                #     nn.Linear(feature_dim, feature_dim),
                #     nn.Dropout(drop_proj),
                # )),
                nn.LayerNorm(feature_dim),
            )
        self.proj_fg = nn.Sequential(
                nn.Linear(embedding_dim, feature_dim),
                # ResidualAdd(nn.Sequential(
                #     nn.GELU(),
                #     nn.Linear(feature_dim, feature_dim),
                #     nn.Dropout(drop_proj),
                # )),
                nn.LayerNorm(feature_dim),
            )
        self.logit_scale_fg = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.logit_scale_bg = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus_fg = nn.Softplus()
        # self.softplus_bg = nn.Softplus()

    def forward(self,x):
        x = self.eeg_encoder(x)
        # x = x.transpose(1,2)
        fg,bg = self.cbam(x)
        fg = fg.flatten(1)
        bg = bg.flatten(1)
        fg = self.proj_fg(fg)
        bg = self.proj_bg(bg)
        return fg,bg
    

import torch
import torch.nn as nn
import torch.nn.functional as F
eeg_chan_order = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
'O1', 'Oz', 'O2']
eeg17_chan_order = ['P7', 'P5', 'P3', 'P1','Pz', 'P2', 'P4', 'P6', 'P8', 
                    'PO7', 'PO3', 'POz', 'PO4', 'PO8','O1', 'Oz', 'O2']
class SeeEEGEncoder(nn.Module):
    def __init__(self, chan_order=eeg17_chan_order, fs=250, embed_dim=256, n_heads=8, n_layers=2,
                 z_dim=768, c_num=17, timesteps=[0,250], drop_proj=0.3):
        super().__init__()
        self.fs = fs
        self.num_electrodes = len(chan_order)
        self.num_regions = 6
        self.chan_order = chan_order
        self.total_nodes = self.num_electrodes + self.num_regions
        self.logit_scale_fg = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.logit_scale_bg = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus_fg = nn.Softplus()

        # 1. 区域映射定义 (基于神经科学分区) [cite: 118, 120]
        self.region_map = [
            # ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8'], # R1
            # ['F7', 'F5', 'F3'],                               # R2
            # ['F4', 'F6', 'F8'],                               # R3
            # ['FT9', 'FT7', 'FC5', 'FC3'],                     # R4
            # ['F1', 'F2', 'FC1', 'FCz', 'FC2'],                # R5
            # ['FC4', 'FC6', 'FT8', 'FT10'],                    # R6
            # ['T7', 'C5', 'C3'],                               # R7
            # ['C1', 'Cz', 'C2','CP1', 'CPz', 'CP2'],    # R8
            # ['C4', 'C6', 'T8'],                               # R9
            # ['TP9', 'TP7', 'CP5', 'CP3'],                     # R10
            # ['CP4', 'CP6', 'TP8', 'TP10'],                    # R11
            ['P7', 'P5', 'P3'],                               # R12
            ['P1', 'Pz', 'P2'],                               # R13
            ['P4', 'P6', 'P8'],                               # R14
            ['PO7', 'PO3'],                                   # R15
            ['PO4', 'PO8'],                                   # R16
            ['POz', 'O1', 'Oz', 'O2']                         # R17
        ]
        
        # 2. 多尺度时间动力学编码器 (Sec 3.2.1) [cite: 126, 127, 128]
        # 使用 Fs/2, Fs/32, Fs/64 作为卷积核大小 [cite: 127]
        scales = [fs // 2, fs // 32, fs // 64]
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(self.total_nodes, self.total_nodes, kernel_size=k, padding=k//2, groups=self.total_nodes) 
            for k in scales
        ])
        self.gating = nn.Sequential(
            nn.Linear(self.total_nodes, len(scales)),
            nn.Softmax(dim=-1)
        )

        # 3. 区域感知 Transformer 与 空间掩码 [cite: 132, 179, 181]
        self.input_proj = nn.Linear(250, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 预计算符合 Figure 2 的 Attention Mask [cite: 181]
        self.register_buffer('attn_mask', self._create_spatial_mask())

        # 4. 多级图聚合与投射头 (Sec 3.2.1) [cite: 185, 191, 193]
        self.elec_conv = nn.Conv2d(1, 1, kernel_size=(17, 1)) 
        # 公式 (4): 处理区域级 (Batch, 1, 17, Embed) -> (Batch, 1, 1, Embed)
        self.region_conv = nn.Conv2d(1, 1, kernel_size=(6, 1))
        # self.global_conv = nn.Linear(embed_dim * 2, embed_dim)
        
        self.projection_head_fg = nn.Sequential(
            nn.Linear(512, 768), # 对齐 CLIP 维度 [cite: 193, 196]
            nn.LayerNorm(768)
        )
        self.projection_head_bg = nn.Sequential(
            nn.Linear(512, 768), # 对齐 CLIP 维度 [cite: 193, 196]
            nn.LayerNorm(768)
        )

    def _create_spatial_mask(self):
        # True 表示屏蔽，False 表示允许关注 [cite: 181]
        mask = torch.ones((self.total_nodes, self.total_nodes), dtype=torch.bool)
        mask[0:self.num_electrodes, 0:self.num_electrodes] = False # 电极间全可见
        mask[self.num_electrodes:, self.num_electrodes:] = False   # 区域间全可见
        
        name_to_idx = {name: i for i, name in enumerate(self.chan_order)}
        for r_idx, channels in enumerate(self.region_map):
            r_node = self.num_electrodes + r_idx
            for chan in channels:
                e_idx = name_to_idx[chan]
                mask[e_idx, r_node] = mask[r_node, e_idx] = False # 仅可见所属区域节点
        return mask

    def forward(self, eeg):
        # eeg shape: (Batch, 63, 250)
        batch_size = eeg.shape[0]

        # A. 生成层次化表示 (Hierarchical Region Representation) 
        name_to_idx = {name: i for i, name in enumerate(self.chan_order)}
        region_list = []
        for chan_names in self.region_map:
            indices = [name_to_idx[name] for name in chan_names]
            region_list.append(eeg[:, indices, :].mean(dim=1, keepdim=True))
        r_i = torch.cat(region_list, dim=1)
        x = torch.cat([eeg, r_i], dim=1) # (B, 80, 250) [cite: 123]

        # B. 多尺度时间编码 [cite: 129, 131]
        feat_list = [conv(x) for conv in self.temporal_convs]
        weights = self.gating(x.mean(dim=-1))
        x = sum(f * weights[:, i].view(-1, 1, 1) for i, f in enumerate(feat_list))

        # C. 区域感知 Transformer [cite: 132, 183]
        x = self.input_proj(x)
        # 应用预设的空间掩码 [cite: 181]
        z = self.transformer(x, mask=self.attn_mask) 

        # D. 多级图聚合 [cite: 185, 192]
        z_elec = self.elec_conv(z[:, :self.num_electrodes, :].unsqueeze(1)).squeeze(1).squeeze(1)
        z_reg = self.region_conv(z[:, self.num_electrodes:, :].unsqueeze(1)).squeeze(1).squeeze(1)
        # print(z_elec.shape)
        # exit()
        # z_ml = self.global_conv(torch.cat([z_elec, z_reg], dim=1)) 
        z_ml = torch.cat([z_elec, z_reg], dim=1)

        # E. 最终全局嵌入 [cite: 193]
        # eeg_fg = self.projection_head_fg(z_ml)
        # eeg_bg = self.projection_head_bg(z_ml)
        return z_ml

class SEBlock1D(nn.Module):
    """
    对应论文图中的 SE: Squeeze-and-Excitation Block 
    用于计算通道注意力权重 'a'
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (Batch, Channel) or (Batch, Channel, Time)
        # 这里假设输入是特征向量 (Batch, Channel)
        b, c = x.size()
        y = self.fc(x) # 生成权重 a, 对应论文 Eq.5 [cite: 234]
        return y

class EEGFeatureDecouplerWithSCFR(nn.Module):
    def __init__(self, input_dim =40,feature_dim=1024, margin=1.0,z_dim=768, c_num=17, timesteps=[0,250], drop_proj=0.3):
        super(EEGFeatureDecouplerWithSCFR, self).__init__()
        self.margin = margin
        
        # 1. IN层: 去除风格/背景 (论文 Eq.2)
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.Dropout(0.5),
            )
        self.projector = nn.Sequential(
            nn.Conv1d(input_dim, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        self.in_layer = nn.InstanceNorm1d(feature_dim, affine=True)
        
        # 2. SE模块: 用于 SCFR 特征还原 (论文 Eq.5) [cite: 230]
        self.se_block = SEBlock1D(feature_dim, reduction=8) # reduction可调
        
        # 3. 编码器 (模拟论文中的 Block4)
        self.fg_proj = nn.Sequential(
            nn.Linear(feature_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim)
        )
        self.bg_proj = nn.Sequential(
            nn.Linear(feature_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim)
        )

    def forward(self, eeg_features):
        """
        :param eeg_features: 原始 EEG 特征 (Batch, Dim) -> 对应论文 F_3
        """
        # --- 阶段 1: 粗略解耦 (SMFD) ---
        # 适配 IN 维度 (N, C, L)
        eeg_features = eeg_features.unsqueeze(1)
        x = self.encoder(eeg_features).squeeze(2)
        
        x_reshaped =x
        
        x_reshaped = self.projector(x_reshaped)
        f_id_seq = self.in_layer(x_reshaped) # F_id (初步前景)
        # print(f_id_seq.shape)
        
        # 获取残差 (背景/风格) F_st = F_3 - F_id [cite: 214]
        f_st_seq = x_reshaped - f_id_seq 
        
        # --- 阶段 2: 特征还原 (SCFR) ---
        # 计算注意力权重 a 
        # 注意：这里对 f_st_raw 进行操作
        f_id_vec = f_id_seq.mean(dim=2)
        f_st_vec = f_st_seq.mean(dim=2)
        
        # 4. 还原 (SCFR): 从背景里捞回有用信息
        attention = self.se_block(f_st_vec) # (B, 512)
        f_stf = attention * f_st_vec        # 有用的背景成分
        f_res_vec = f_id_vec + f_stf        # 最终前景
        
        # 无用的背景成分 (用于 Loss 约束)
        f_stl_vec = (1 - attention) * f_st_vec
        
        # 5. 最终映射
        out_res = self.fg_proj(f_res_vec)
        out_stl = self.bg_proj(f_stl_vec)
        
        return out_res,out_stl
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class DAPT_EEG_Encoder(nn.Module):
    def __init__(self, in_channels=17, time_points=250, clip_dim=768, num_heads=8,**kwargs):
        """
        DAPT-EEG Encoder: Decouple before Align
        
        Args:
            in_channels: EEG 通道数 (63)
            clip_dim: 目标对齐的 CLIP 维度 (1024)
            num_heads: Attention 头数
        """
        super().__init__()
        
        # --- 1. Backbone (您指定的结构) ---
        # 作用：提取时空特征，但保留时间维度，不直接 Flatten
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),      # Temporal Conv
            nn.Conv2d(40, 40, (in_channels, 1), (1, 1)),  # Spatial Conv
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),          # Large Pooling
            nn.Dropout(0.5),
        )
        
        # 计算 Backbone 输出维度
        # 输入 (250) -> Conv1(25) -> 226
        # Pool(51, stride 5) -> floor((226-51)/5)+1 = 36
        self.seq_len = 36
        self.feature_dim = 40
        
        # --- 2. Projection Layer ---
        # 将 Backbone 的 40 维特征映射到 CLIP 的 1024 维
        # 依然保持序列形式 (B, Seq_Len, 1024)
        self.projector = nn.Linear(self.feature_dim, clip_dim)
        
        # --- 3. Decoupling Module (Query-based Active Purification) ---
        # 定义两个可学习的“探针”，分别负责抓取前景和背景
        self.fg_query = nn.Parameter(torch.randn(1, 1, clip_dim)) 
        self.bg_query = nn.Parameter(torch.randn(1, 1, clip_dim))
        
        # Cross-Attention: Query 去序列里“淘金”
        self.cross_attn = nn.MultiheadAttention(embed_dim=clip_dim, num_heads=num_heads, batch_first=True)
        
        # --- 4. Global Anchor Pooling ---
        # 用于生成 z_global (训练时的锚点)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.logit_scale_fg = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.logit_scale_bg = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus_fg = nn.Softplus()

    def forward(self, x):
        """
        Args:
            x: EEG Raw Data (B, 63, 250)
        Returns:
            z_global: 全局混合特征 (Anchor)
            z_fg:     纯净前景特征 (Positive)
            z_bg:     分离背景特征 (Negative)
        """
        B = x.shape[0]
        
        # 1. 维度适配: (B, 63, 250) -> (B, 1, 63, 250)
        x = x.unsqueeze(1)
        
        # 2. Backbone 提取
        # Out: (B, 40, 1, 36)
        feat = self.backbone(x)
        
        # 3. 序列化 (Squeeze & Permute)
        # 去掉空间维(1)，变为 (B, 40, 36) -> (B, 36, 40)
        feat = feat.squeeze(2).permute(0, 2, 1)
        
        # 4. 映射到 CLIP 空间
        # feat_seq: (B, 36, 1024) -> 这是含金沙和泥沙的混合流
        feat_seq = self.projector(feat)
        
        # --- DAPT 解耦阶段 ---
        
        # A. 生成 Global Anchor (简单平均)
        # z_global: (B, 1024)
        z_global = feat_seq.mean(dim=1)
        
        # B. 生成 Foreground (主动提纯)
        # 扩展 Query: (B, 1, 1024)
        fg_q = self.fg_query.expand(B, -1, -1)
        # z_fg: (B, 1, 1024)
        z_fg, _ = self.cross_attn(query=fg_q, key=feat_seq, value=feat_seq)
        z_fg = z_fg.squeeze(1)
        
        # C. 生成 Background (主动分离)
        bg_q = self.bg_query.expand(B, -1, -1)
        z_bg, _ = self.cross_attn(query=bg_q, key=feat_seq, value=feat_seq)
        z_bg = z_bg.squeeze(1)
        
        # --- 归一化 (CLIP 空间必备) ---
        # z_global = F.normalize(z_global, p=2, dim=1)
        z_fg = F.normalize(z_fg, p=2, dim=1)
        z_bg = F.normalize(z_bg, p=2, dim=1)
        
        return z_global,z_fg, z_bg

if __name__=="__main__":
    x =torch.zeros((32,17,250))
    model = EEGFeatureDecouplerWithSCFR()
    y,z = model(x)
    print(y.shape)