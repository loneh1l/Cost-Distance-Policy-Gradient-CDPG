import os
from typing import Tuple

import numpy as np
import torch


class DataLoader:
    def __init__(self, filename, default_device):
        self.filename = filename
        print(f"Loading data from {filename} to DataLoader……")
        self.data = torch.load(filename)
        self.default_device = default_device
        print("Loaded.")

    def load(self, device=None):
        """
        OUTPUT:
        tensor_adj,   : Tensor (? x 2 x N x N)
        tensor_vector,   : Tensor (? x 2 x N)
        """
        if device is None:
            device = self.default_device
        adj = self.data['adj'].to(device)
        vector = self.data['vector'].to(device)
        return adj, vector


def _load_csv_graph(nodes_path: str, edges_path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes_arr = np.loadtxt(nodes_path, delimiter=",", skiprows=1, dtype=np.float64)
    if nodes_arr.size == 0:
        raise ValueError(f"No nodes found in {nodes_path}")
    if nodes_arr.ndim == 1:
        nodes_arr = nodes_arr[None, :]

    node_ids = nodes_arr[:, 0].astype(np.int64)
    weights_np = nodes_arr[:, 1].astype(np.float32)
    types_np = nodes_arr[:, 2].astype(np.float32)

    max_node = int(node_ids.max())
    n = max_node + 1
    weights_np_full = np.zeros((n,), dtype=np.float32)
    types_np_full = np.zeros((n,), dtype=np.float32)
    weights_np_full[node_ids] = weights_np
    types_np_full[node_ids] = types_np

    try:
        edges_arr = np.loadtxt(edges_path, delimiter=",", skiprows=1, dtype=np.float64)
    except ValueError:
        edges_arr = np.empty((0, 4), dtype=np.float64)
    if edges_arr.size == 0:
        cons_np = np.zeros((n, n), dtype=np.float32)
        trans_np = np.zeros((n, n), dtype=np.float32)
    else:
        if edges_arr.ndim == 1:
            edges_arr = edges_arr[None, :]
        u = edges_arr[:, 0].astype(np.int64)
        v = edges_arr[:, 1].astype(np.int64)
        cons = edges_arr[:, 2].astype(np.float32)
        trans = edges_arr[:, 3].astype(np.float32)
        cons_np = np.zeros((n, n), dtype=np.float32)
        trans_np = np.zeros((n, n), dtype=np.float32)
        cons_np[u, v] = cons
        trans_np[u, v] = trans
        cons_np[v, u] = cons
        trans_np[v, u] = trans

    adj = torch.from_numpy(np.stack([cons_np, trans_np])).to(device)
    vector = torch.from_numpy(np.stack([weights_np_full, types_np_full])).to(device)
    return adj, vector


class GbaCDNetLoader:
    def __init__(self, root_dir: str, subgraph: int, device: torch.device):
        self.root_dir = root_dir
        self.subgraph = subgraph
        self.device = device

    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sub_dir = os.path.join(self.root_dir, f"GBA-CDNet-{self.subgraph}")
        nodes_path = os.path.join(sub_dir, "nodes.csv")
        edges_path = os.path.join(sub_dir, "edges.csv")
        if not os.path.isfile(nodes_path):
            raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
        if not os.path.isfile(edges_path):
            raise FileNotFoundError(f"Edges file not found: {edges_path}")
        return _load_csv_graph(nodes_path, edges_path, self.device)
