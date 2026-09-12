import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange
from einops.layers.torch import Reduce, Rearrange

class shallownet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(shallownet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, C, T)
            dummy_output = self.net(dummy_input)
            self.num_flat_features = dummy_output.numel()
        self.out = nn.Linear(self.num_flat_features, out_dim)
    
    def forward(self, x):               #input:(batch,C,T)
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

class deepnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(deepnet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (C, 1), (1, 1)),
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
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, C, T)
            dummy_output = self.net(dummy_input)
            self.num_flat_features = dummy_output.numel()
        self.out = nn.Linear(self.num_flat_features, out_dim)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class eegnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(eegnet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (C, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.5)
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, C, T)
            dummy_output = self.net(dummy_input)
            self.num_flat_features = dummy_output.numel()
        self.out = nn.Linear(self.num_flat_features, out_dim)
    
    def forward(self, x):               #input:(batch,1,C,T)
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

class PatchEmbedding(nn.Module):
    # 1. 增加 C 参数，默认值设为 62 以防万一
    def __init__(self, emb_size=40, C=62): 
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            # 2. 将 (63, 1) 修改为 (C, 1)
            nn.Conv2d(40, 40, (C, 1), (1, 1)), 
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        x = self.projection(x)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = scaling = (self.emb_size // self.num_heads) ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ClassificationHead(nn.Module): # 建议继承 nn.Module 而不是 Sequential，更灵活
    def __init__(self, emb_size, out_dim):
        super().__init__()
        
        # 使用全局平均池化 (GAP)
        # 无论输入序列长度是 7 还是 700，这里输出永远是 (Batch, emb_size)
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, out_dim)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Emb_Size)
        out = self.clshead(x)
        return out

class conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=3, out_dim=4, **kwargs):
        # 1. 从 kwargs 中提取 C，如果没有则默认为 62
        C = kwargs.get('C', 62) 
        
        super().__init__(
            # 2. 将 C 传递给 PatchEmbedding
            PatchEmbedding(emb_size, C=C), 
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, out_dim)
        )

class tsconv(nn.Sequential):
    def __init__(self, emb_size=40, depth=3, out_dim=4, C=62, T=400):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, C, T)
            dummy_output = self.net(dummy_input)
            self.num_flat_features = dummy_output.numel()
        self.out = nn.Linear(self.num_flat_features, out_dim)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

class glfnet(nn.Module):
    def __init__(self, out_dim, emb_dim=128, C=62, T=400):
        super(glfnet, self).__init__()
        
        self.globalnet = shallownet(emb_dim, C, T)
        
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = shallownet(emb_dim, 12, T)
        
        self.out = nn.Linear(emb_dim*2, out_dim)
        
    
    def forward(self, x):               #input:(batch,C,T)
        global_feature = self.globalnet(x)
        global_feature = global_feature.view(x.size(0), -1)
        # global_feature = self.out(global_feature)
        occipital_x = x[:, self.occipital_index, :]
        # print("occipital_x.shape = ", occipital_x.shape)
        occipital_feature = self.occipital_localnet(occipital_x)
        # print("occipital_feature.shape = ", occipital_feature.shape)
        out = self.out(torch.cat((global_feature, occipital_feature), 1))
        return out

class mlpnet(nn.Module):
    def __init__(self, out_dim, input_dim):
        super(mlpnet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, out_dim)
        )
        
    def forward(self, x):               #input:(batch,C,5)
        out = self.net(x)
        return out

class glfnet_mlp(nn.Module):
    def __init__(self, out_dim, emb_dim, input_dim):
        super(glfnet_mlp, self).__init__()
        
        self.globalnet = mlpnet(emb_dim, input_dim)
        
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = mlpnet(emb_dim, 12*5)
        
        self.out = nn.Linear(emb_dim*2, out_dim)
        
    
    def forward(self, x):               #input:(batch,C,5)
        global_feature = self.globalnet(x)
        # global_feature = global_feature.view(x.size(0), -1)
        # global_feature = self.out(global_feature)
        occipital_x = x[:, self.occipital_index, :]
        # print("occipital_x.shape = ", occipital_x.shape)
        occipital_feature = self.occipital_localnet(occipital_x)
        # print("occipital_feature.shape = ", occipital_feature.shape)
        out = self.out(torch.cat((global_feature, occipital_feature), 1))
        return out

# data/adj_matrix.npy
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    raise ImportError("Please install torch_geometric to use the GCN_LSTM model.")

class GCN_LSTM(nn.Module):
    """
    GCN-LSTM Model for EEG Analysis
    - Spatial: Graph Convolutional Network (GCN) using electrode adjacency matrix.
    - Temporal: Bi-directional LSTM with Attention Mechanism.
    """
    def __init__(self, out_dim, C=62, T=400, 
                 adj_path='data/adj_matrix.npy',
                 gcn_hidden=16, lstm_hidden=64, dropout=0.5):
        super(GCN_LSTM, self).__init__()
        
        self.C = C
        self.T = T
        self.gcn_hidden = gcn_hidden
        
        # --- 1. Graph Initialization ---
        # Load adjacency matrix and register as buffer (moves to GPU automatically with model)
        edge_index, edge_weight = self._prepare_graph(adj_path, threshold=0.5)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)


        self.temp_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 25), stride=(1, 10)), 
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # --- 2. Spatial Encoder (GCN) ---
        # We treat every timepoint as a graph signal. Input feature per node is 1 (amplitude).
        self.gcn1 = GCNConv(16, gcn_hidden)
        self.bn1 = nn.LayerNorm(gcn_hidden)
        
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.bn2 = nn.LayerNorm(gcn_hidden)
        
        self.dropout = nn.Dropout(dropout)

        # --- 3. Temporal Encoder (LSTM) ---
        # Input to LSTM: Flattened spatial features (Num_Nodes * GCN_Features)
        lstm_input_size = C * gcn_hidden
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # --- 4. Attention Mechanism ---
        # Projects LSTM output to a score for each time step
        self.attention_layer = nn.Linear(lstm_hidden * 2, 1)

        # --- 5. Prediction Head ---
        # Input: Context vector (LSTM hidden size * 2 because bidirectional)
        self.out = nn.Linear(lstm_hidden * 2, out_dim)

    def _prepare_graph(self, adj_path, threshold=0.5):
        """Loads .npy adjacency matrix and returns edge_index and edge_weight tensors."""
        try:
            adj_matrix = np.load(adj_path)
        except FileNotFoundError:
            print(f"Warning: {adj_path} not found. Initializing random graph for testing.")
            adj_matrix = np.random.rand(self.C, self.C)

        # Thresholding
        adj_matrix[adj_matrix < threshold] = 0
        
        # Convert to sparse format (COO)
        sources, targets = np.where(adj_matrix > 0)
        weights = adj_matrix[sources, targets]
        
        edge_index = torch.tensor(np.array([sources, targets]), dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        
        return edge_index, edge_weight

    def forward(self, x):
        # Input x shape: (Batch, C, T)
        x = x.unsqueeze(1)
        x = self.temp_conv(x)
        B, n_filters, C, T = x.shape
        
        # --- Spatial Feature Extraction (GCN) ---
        # Reshape to treat every timepoint in the batch as a separate graph sample
        # (B, C, T) -> (B, T, C) -> (B*T, C, 1)
        x_gcn = x.permute(0, 3, 2, 1).reshape(B * T, C, n_filters)
        
        # Pass static edge_index and edge_weight (automatically on correct device)
        x_gcn = self.gcn1(x_gcn, self.edge_index, self.edge_weight)
        x_gcn = F.relu(self.bn1(x_gcn))
        x_gcn = self.dropout(x_gcn)
        
        x_gcn = self.gcn2(x_gcn, self.edge_index, self.edge_weight)
        x_gcn = F.relu(self.bn2(x_gcn))
        x_gcn = self.dropout(x_gcn)
        
        # Output shape: (B*T, C, GCN_Hidden)
        
        # --- Temporal Modeling (LSTM) ---
        # Reshape back to sequence: (B, T, C * GCN_Hidden)
        x_seq = x_gcn.view(B, T, -1)
        
        lstm_out, _ = self.lstm(x_seq) 
        # lstm_out shape: (B, T, LSTM_Hidden * 2)

        # --- Attention Aggregation ---
        # Calculate attention scores
        # (B, T, Hidden*2) -> (B, T, 1)
        attn_logits = self.attention_layer(lstm_out)
        attn_weights = F.softmax(torch.tanh(attn_logits), dim=1)
        
        # Weighted sum of LSTM outputs
        # (B, T, Hidden*2) * (B, T, 1) -> Sum over T -> (B, Hidden*2)
        context_vector = torch.sum(lstm_out * attn_weights, dim=1)
        
        # --- Final Prediction ---
        out = self.out(context_vector)
        return out
