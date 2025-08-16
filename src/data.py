import torch
from collections import defaultdict
import random

class KnowledgeGraphDataBuilder:
    def __init__(self, triples: dict):
        self.triples = triples

        self.idx2node = {}
        self.node2idx = {}

        self.idx2edge = {}
        self.edge2idx = {}

        self.graph = [[[], []], []]

        self.build()

    def get_node_size(self):
        return len(self.idx2node)
    
    def get_edge_size(self):
        return len(self.idx2edge)

    def get_graph_size(self):
        return len(self.graph[1])

    def get_graph(self):
        return self.graph

    def get_nodes(self):
        return self.idx2node, self.node2idx
    
    def get_edges(self):
        return self.idx2edge, self.edge2idx

    def build(self):
        self.build_edges_nodes()
        self.build_graph()

    def build_edges_nodes(self):
        node_set = set()
        edge_set = set()
    
        for triple in self.triples:
            node_set.add(triple[0])
            edge_set.add(triple[1])
            node_set.add(triple[2])
        
        for idx, node in enumerate(node_set):
            self.idx2node[idx] = node
            self.node2idx[node] = idx

        for idx, edge in enumerate(edge_set):
            self.idx2edge[idx] = edge
            self.edge2idx[edge] = idx

    def build_graph(self):
        for (s, p, o) in self.triples:
            s_idx = self.node2idx[s]
            p_idx = self.edge2idx[p]
            o_idx = self.node2idx[o]

            self.graph[0][0].append(s_idx)
            self.graph[0][1].append(o_idx)
            self.graph[1].append(p_idx)
        
        assert len(self.graph[0][0]) == len(self.graph[0][1])
        assert len(self.graph[0][0]) == len(self.graph[1])
