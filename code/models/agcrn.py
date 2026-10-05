"""AGCRN as a BasicTS 1.0 model.

Paper : Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting (NeurIPS 2020)
Code  : https://github.com/LeiBAI/AGCRN  (model/AGCN.py, AGCRNCell.py, AGCRN.py)

This is a faithful port of the official architecture with ONE purely computational change:
the adaptive supports [I, A, ...] and the node-specific weights/bias are computed ONCE per forward
pass instead of once per time step (they only depend on the node embeddings, which do not change
inside a forward pass), so the maths is identical but it is ~10x cheaper.
Parameter names are kept identical to the official code, so an official state_dict loads directly
(this is what code/scripts/check_agcrn_equivalence.py relies on).

BasicTS interface (see docs/model_design.md of BasicTS):
  forward(inputs: [batch, input_len, num_features]) -> prediction [batch, output_len, num_features]
  where num_features = number of sensors (nodes) and every node has ONE channel (traffic flow).
"""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from basicts.configs import BasicTSModelConfig


@dataclass
class AGCRNConfig(BasicTSModelConfig):
    """Config of AGCRN. Defaults = official PEMSD4 setting (embed_dim=2 for PEMSD8)."""

    input_len: int = field(default=12, metadata={"help": "Input sequence length."})
    output_len: int = field(default=12, metadata={"help": "Output sequence length (horizon)."})
    num_features: int = field(default=None, metadata={"help": "Number of nodes (sensors)."})
    embed_dim: int = field(default=10, metadata={"help": "Node embedding dimension (10 for PEMSD4, 2 for PEMSD8)."})
    rnn_units: int = field(default=64, metadata={"help": "Hidden size of the recurrent cells."})
    num_layers: int = field(default=2, metadata={"help": "Number of stacked AGCRN layers."})
    cheb_k: int = field(default=2, metadata={"help": "Order of the Chebyshev-style graph convolution (>= 2)."})
    input_dim: int = field(default=1, metadata={"help": "Channels per node in the input."})
    output_dim: int = field(default=1, metadata={"help": "Channels per node in the output."})


class AVWGCN(nn.Module):
    """Node-adaptive graph convolution (official AGCN.py), split into prepare() + forward()."""

    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, dim_out))

    def prepare(self, node_embeddings):
        """Node-specific weights [N, K, dim_in, dim_out] and bias [N, dim_out] (computed once per forward)."""
        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)
        return weights, bias

    def forward(self, x, supports, weights, bias):
        # x: [B, N, C]; supports: [K, N, N]
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      # [B, K, N, C]
        x_g = x_g.permute(0, 2, 1, 3)                         # [B, N, K, C]
        return torch.einsum("bnki,nkio->bno", x_g, weights) + bias   # [B, N, dim_out]


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + dim_out, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + dim_out, dim_out, cheb_k, embed_dim)

    def prepare(self, node_embeddings):
        return self.gate.prepare(node_embeddings), self.update.prepare(node_embeddings)

    def forward(self, x, state, supports, prepared):
        (gate_w, gate_b), (upd_w, upd_b) = prepared
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports, gate_w, gate_b))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports, upd_w, upd_b))
        return r * state + (1 - r) * hc


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super().__init__()
        assert num_layers >= 1, "At least one DCRNN layer in the Encoder."
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, supports, node_embeddings):
        # x: [B, T, N, D]; init_state: [num_layers, B, N, hidden]
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        for i in range(self.num_layers):
            cell = self.dcrnn_cells[i]
            prepared = cell.prepare(node_embeddings)          # once per layer, not per time step
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = cell(current_inputs[:, t, :, :], state, supports, prepared)
                inner_states.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs                                  # outputs of the last layer: [B, T, N, hidden]


class AGCRN(nn.Module):
    """Paper: Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting.
    Official Code: https://github.com/LeiBAI/AGCRN. Venue: NeurIPS 2020. Task: Spatial-Temporal Forecasting."""

    def __init__(self, config: AGCRNConfig):
        super().__init__()
        assert config.num_features is not None, "AGCRNConfig.num_features (number of nodes) must be set"
        assert config.cheb_k >= 2, "cheb_k must be >= 2 (as in the official code)"
        assert config.output_dim == 1, "this port supports output_dim == 1 only"
        self.num_node = config.num_features
        self.input_dim = config.input_dim
        self.hidden_dim = config.rnn_units
        self.output_dim = config.output_dim
        self.horizon = config.output_len
        self.num_layers = config.num_layers
        self.cheb_k = config.cheb_k

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, config.embed_dim), requires_grad=True)
        self.encoder = AVWDCRNN(self.num_node, self.input_dim, self.hidden_dim, config.cheb_k,
                                config.embed_dim, config.num_layers)
        # predictor
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        # official initialisation (Run.py): xavier for matrices, uniform(0, 1) for 1-d parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def _supports(self):
        """[I, A, 2A*A_prev - A_prevprev, ...] with A = softmax(relu(E E^T)); shape [cheb_k, N, N]."""
        e = self.node_embeddings
        a = F.softmax(F.relu(torch.mm(e, e.transpose(0, 1))), dim=1)
        support_set = [torch.eye(self.num_node, device=e.device, dtype=e.dtype), a]
        for _ in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * a, support_set[-1]) - support_set[-2])
        return torch.stack(support_set, dim=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Args: inputs [batch, input_len, num_features]. Returns: prediction [batch, output_len, num_features]."""
        x = inputs.unsqueeze(-1)                                              # [B, T, N, 1]
        init_state = x.new_zeros(self.num_layers, x.shape[0], self.num_node, self.hidden_dim)
        output = self.encoder(x, init_state, self._supports(), self.node_embeddings)   # [B, T, N, hidden]
        output = output[:, -1:, :, :]                                         # [B, 1, N, hidden]
        output = self.end_conv(output)                                        # [B, horizon*C, N, 1]
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                                   # [B, horizon, N, C]
        return output.squeeze(-1)                                             # [B, horizon, N]
