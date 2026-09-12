"""GCN-LSTM baseline; graph construction and layer order follow the archived code."""

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GCNLSTM(nn.Module):
    """GCNLSTM with the archived initialization and forward operations.

    Args:
        out_dim: Number of classification outputs.
        C: Number of EEG channels.
        T: Number of time samples.
        adj_path: Path to the electrode adjacency array.
        gcn_hidden: Graph-convolution feature width.
        lstm_hidden: LSTM hidden width per direction.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        out_dim: int,
        C: int = 62,
        T: int = 400,
        adj_path="data/adj_matrix.npy",
        gcn_hidden: int = 16,
        lstm_hidden: int = 64,
        dropout: float = 0.5,
    ) -> None:
        from torch_geometric.nn import GCNConv

        super(GCNLSTM, self).__init__()
        self.C = C
        self.T = T
        self.gcn_hidden = gcn_hidden
        edge_index, edge_weight = self._prepare_graph(adj_path, threshold=0.5)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.temp_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 25), stride=(1, 10)), nn.BatchNorm2d(16), nn.ReLU()
        )
        self.gcn1 = GCNConv(16, gcn_hidden)
        self.bn1 = nn.LayerNorm(gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.bn2 = nn.LayerNorm(gcn_hidden)
        self.dropout = nn.Dropout(dropout)
        lstm_input_size = C * gcn_hidden
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attention_layer = nn.Linear(lstm_hidden * 2, 1)
        self.out = nn.Linear(lstm_hidden * 2, out_dim)

    def _prepare_graph(self, adj_path, threshold=0.5):
        """Loads .npy adjacency matrix and returns edge_index and edge_weight tensors."""
        try:
            adj_matrix = np.load(adj_path)
        except FileNotFoundError:
            print(f"Warning: {adj_path} not found. Initializing random graph for testing.")
            adj_matrix = np.random.rand(self.C, self.C)
        adj_matrix[adj_matrix < threshold] = 0
        sources, targets = np.where(adj_matrix > 0)
        weights = adj_matrix[sources, targets]
        edge_index = torch.tensor(np.array([sources, targets]), dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        return (edge_index, edge_weight)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the archived forward computation.

        Args:
            x: Input batch in the shape expected by this module.

        Returns:
            Output tensor with the original model semantics.
        """
        x = x.unsqueeze(1)
        x = self.temp_conv(x)
        B, n_filters, C, T = x.shape
        x_gcn = x.permute(0, 3, 2, 1).reshape(B * T, C, n_filters)
        x_gcn = self.gcn1(x_gcn, self.edge_index, self.edge_weight)
        x_gcn = F.relu(self.bn1(x_gcn))
        x_gcn = self.dropout(x_gcn)
        x_gcn = self.gcn2(x_gcn, self.edge_index, self.edge_weight)
        x_gcn = F.relu(self.bn2(x_gcn))
        x_gcn = self.dropout(x_gcn)
        x_seq = x_gcn.view(B, T, -1)
        lstm_out, _ = self.lstm(x_seq)
        attn_logits = self.attention_layer(lstm_out)
        attn_weights = F.softmax(torch.tanh(attn_logits), dim=1)
        context_vector = torch.sum(lstm_out * attn_weights, dim=1)
        out = self.out(context_vector)
        return out
