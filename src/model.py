"""
https://arxiv.org/abs/1703.06103
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class RGCNLayer(nn.Module):
    def __init__(self, num_relations, num_nodes, hidden_dim=64):
        super().__init__()
        self.num_relations = num_relations
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        self.relation_weights = nn.Parameter(torch.randn(num_relations, hidden_dim, hidden_dim))
        self.selfloop_weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.randn(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.relation_weights)
        nn.init.xavier_uniform_(self.selfloop_weights)
        nn.init.zeros_(self.bias)

    def forward(self, in_node_feature, edge_index, edge_type):
        """
            in_node_feature: (num_nodes, hidden_dim)
            edge_index: [2, num_edge], edge_index[0, i] = src_node, edge_index[1, i] = dst_node
            edge_type: [num_edges], the relation type ID for the edge

            h_(l+1) = activate_func(sum_relation(sum_neighbors(node_features)) + h_l@selfloop_weights + bias)
        """
        out_node_feature = torch.zeros(self.num_nodes, self.hidden_dim) # h_(l+1)

        for rel in range(self.num_relations): # --> sum_relation(*)
            mask = edge_type == rel
            edges = edge_index[:, mask]

            if edges.size(1) == 0: continue

            src, dst = edges
            messages = in_node_feature[src] @ self.relation_weights[rel] # --> sum_neighbors(node_features)

            temp_out = torch.zeros(self.num_nodes, self.hidden_dim)
            temp_out.index_add_(0, dst, messages)
            out_node_feature += temp_out

        out_node_feature += in_node_feature @ self.selfloop_weights
        out_node_feature += self.bias

        return F.relu(out_node_feature)