from __future__ import annotations

from typing import Sequence

import numpy as np

from pose_module.interfaces import IMUGPT_22_PARENT_INDICES, MOTIONBERT_17_PARENT_INDICES


def _build_edges_from_parents(parents: Sequence[int]) -> tuple[int, list[tuple[int, int]], int]:
    num_nodes = int(len(parents))
    self_links = [(node_index, node_index) for node_index in range(num_nodes)]
    neighbor_links = [(node_index, int(parent_index)) for node_index, parent_index in enumerate(parents) if int(parent_index) >= 0]
    return num_nodes, self_links + neighbor_links, 0

class Graph:
    """Graph definition adapted from the ST-GCN reference implementation."""

    def __init__(self, layout: str = "imugpt22", strategy: str = "spatial", max_hop: int = 1, dilation: int = 1) -> None:
        self.max_hop = int(max_hop)
        self.dilation = int(dilation)
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=self.max_hop)
        self.get_adjacency(strategy)

    def __str__(self) -> str:
        return str(self.A)

    def get_edge(self, layout: str) -> None:
        normalized_layout = str(layout).strip().lower()
        if normalized_layout == "imugpt22":
            self.num_node, self.edge, self.center = _build_edges_from_parents(IMUGPT_22_PARENT_INDICES)
            self.center = 0
            return
        if normalized_layout == "motionbert17":
            self.num_node, self.edge, self.center = _build_edges_from_parents(MOTIONBERT_17_PARENT_INDICES)
            self.center = 0
            return
        if normalized_layout == "openpose":
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (4, 3),
                (3, 2),
                (7, 6),
                (6, 5),
                (13, 12),
                (12, 11),
                (10, 9),
                (9, 8),
                (11, 5),
                (8, 2),
                (5, 1),
                (2, 1),
                (0, 1),
                (15, 0),
                (14, 0),
                (17, 15),
                (16, 14)
            ]
            self.edge = self_link + neighbor_link
            self.center = 1
            return
        raise ValueError(f"Unsupported graph layout: {layout}")

    def get_adjacency(self, strategy: str) -> None:
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalized_adjacency = normalize_digraph(adjacency)

        if strategy == "uniform":
            graph = np.zeros((1, self.num_node, self.num_node))
            graph[0] = normalized_adjacency
            self.A = graph
            return

        if strategy == "distance":
            graph = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for graph_index, hop in enumerate(valid_hop):
                graph[graph_index][self.hop_dis == hop] = normalized_adjacency[self.hop_dis == hop]
            self.A = graph
            return

        if strategy == "spatial":
            graph_partitions = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for target_index in range(self.num_node):
                    for source_index in range(self.num_node):
                        if self.hop_dis[source_index, target_index] != hop:
                            continue
                        if self.hop_dis[source_index, self.center] == self.hop_dis[target_index, self.center]:
                            a_root[source_index, target_index] = normalized_adjacency[source_index, target_index]
                        elif self.hop_dis[source_index, self.center] > self.hop_dis[target_index, self.center]:
                            a_close[source_index, target_index] = normalized_adjacency[source_index, target_index]
                        else:
                            a_further[source_index, target_index] = normalized_adjacency[source_index, target_index]
                if hop == 0:
                    graph_partitions.append(a_root)
                else:
                    graph_partitions.append(a_root + a_close)
                    graph_partitions.append(a_further)
            self.A = np.stack(graph_partitions)
            return

        raise ValueError(f"Unsupported graph strategy: {strategy}")

def get_hop_distance(num_node: int, edge: Sequence[tuple[int, int]], max_hop: int = 1) -> np.ndarray:
    adjacency = np.zeros((num_node, num_node))
    for source_index, target_index in edge:
        adjacency[target_index, source_index] = 1
        adjacency[source_index, target_index] = 1

    hop_distance = np.zeros((num_node, num_node)) + np.inf
    transfer_matrices = [np.linalg.matrix_power(adjacency, depth) for depth in range(max_hop + 1)]
    arrival_matrix = np.stack(transfer_matrices) > 0
    for depth in range(max_hop, -1, -1):
        hop_distance[arrival_matrix[depth]] = depth
    return hop_distance

def normalize_digraph(adjacency: np.ndarray) -> np.ndarray:
    degree = np.sum(adjacency, axis=0)
    num_node = adjacency.shape[0]
    degree_matrix = np.zeros((num_node, num_node))
    for node_index in range(num_node):
        if degree[node_index] > 0:
            degree_matrix[node_index, node_index] = degree[node_index] ** (-1)
    return np.dot(adjacency, degree_matrix)