"""Dynamic causal direction-biased graph attention used by CDGAT-PUR."""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class DynamicCausalGATConv(MessagePassing):
    """Edge-aware GATv2 layer with a learnable ANM direction bias."""

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        edge_dim=10,
        dropout=0.2,
        negative_slope=0.2,
        lambda_init=-4.0,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.lambda_init = float(lambda_init)

        # The reported experiments used a shared layer implementation that
        # registered both static and dynamic attention parameters. The static
        # parameters below are inactive in this GATv2-only forward pass, but
        # retaining them preserves the initialization sequence of those runs.
        self.static_projection = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.static_attention_source = nn.Parameter(torch.empty(1, heads, out_channels))
        self.static_attention_target = nn.Parameter(torch.empty(1, heads, out_channels))
        self.lin_source = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_target = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.attention = nn.Parameter(torch.empty(1, heads, out_channels))
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.static_attention_edge = nn.Parameter(torch.empty(1, heads, out_channels))
        output_dim = heads * out_channels if concat else out_channels
        self.bias = nn.Parameter(torch.empty(output_dim))
        self.raw_lambda = nn.Parameter(torch.tensor(self.lambda_init))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.static_projection.weight)
        nn.init.xavier_uniform_(self.static_attention_source)
        nn.init.xavier_uniform_(self.static_attention_target)
        nn.init.xavier_uniform_(self.lin_source.weight)
        nn.init.xavier_uniform_(self.lin_target.weight)
        nn.init.xavier_uniform_(self.attention)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.static_attention_edge)
        nn.init.zeros_(self.bias)
        with torch.no_grad():
            self.raw_lambda.fill_(self.lambda_init)

    def forward(self, x, edge_index, edge_attr, causal_confidence):
        source = self.lin_source(x).view(-1, self.heads, self.out_channels)
        target = self.lin_target(x).view(-1, self.heads, self.out_channels)
        output = self.propagate(
            edge_index,
            x=(source, target),
            edge_attr=edge_attr,
            causal_confidence=causal_confidence,
            size=None,
        )
        if self.concat:
            output = output.reshape(-1, self.heads * self.out_channels)
        else:
            output = output.mean(dim=1)
        return output + self.bias

    def message(self, x_i, x_j, edge_attr, causal_confidence, index, ptr, size_i):
        edge_repr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        dynamic_repr = F.leaky_relu(x_i + x_j + edge_repr, self.negative_slope)
        attention_logit = (dynamic_repr * self.attention).sum(dim=-1)
        lambda_positive = F.softplus(self.raw_lambda)
        attention_logit = attention_logit + lambda_positive * causal_confidence.view(-1, 1)
        coefficients = softmax(attention_logit, index, ptr, size_i)
        coefficients = F.dropout(coefficients, p=self.dropout, training=self.training)
        return x_j * coefficients.unsqueeze(-1)


class CDGATPUR(nn.Module):
    """Two-layer dynamic CDGAT encoder with a gene-level prediction head."""

    def __init__(
        self,
        in_features,
        edge_features=10,
        hidden_features=16,
        heads=8,
        dropout=0.2,
        negative_slope=0.2,
        lambda_init=-4.0,
    ):
        super().__init__()
        hidden_dim = hidden_features * heads
        self.dropout = nn.Dropout(dropout)
        self.gat1 = DynamicCausalGATConv(
            in_features,
            hidden_features,
            heads=heads,
            concat=True,
            edge_dim=edge_features,
            dropout=dropout,
            negative_slope=negative_slope,
            lambda_init=lambda_init,
        )
        self.gat2 = DynamicCausalGATConv(
            hidden_dim,
            hidden_features,
            heads=1,
            concat=False,
            edge_dim=edge_features,
            dropout=dropout,
            negative_slope=negative_slope,
            lambda_init=lambda_init,
        )
        self.residual1 = (
            nn.Linear(in_features, hidden_dim) if in_features != hidden_dim else nn.Identity()
        )
        self.residual2 = (
            nn.Linear(hidden_dim, hidden_features) if hidden_dim != hidden_features else nn.Identity()
        )
        self.classifier = nn.Linear(hidden_features, 1)

    def forward(self, x, edge_index, edge_attr, causal_confidence):
        hidden = self.gat1(x, edge_index, edge_attr, causal_confidence)
        hidden = self.dropout(F.relu(hidden + self.residual1(x)))
        embedding = self.gat2(hidden, edge_index, edge_attr, causal_confidence)
        embedding = self.dropout(F.relu(embedding + self.residual2(hidden)))
        return self.classifier(embedding).squeeze(-1)
