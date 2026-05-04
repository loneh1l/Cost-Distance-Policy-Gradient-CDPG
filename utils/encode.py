import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def _base_edge_mask_np(adj: torch.Tensor) -> np.ndarray:
    cons_adj = adj[0].detach().cpu().numpy()
    trans_adj = adj[1].detach().cpu().numpy()
    return (cons_adj > 0) & (trans_adj > 0)

def dijkstra_distances(
    adj: torch.Tensor,
    center_node: int,
    dij_type: str = "trans",
    vector: Optional[torch.Tensor] = None,
    directed: bool = True,
) -> torch.Tensor:
    base_mask = _base_edge_mask_np(adj)
    rows, cols = np.nonzero(base_mask)
    cons_adj = adj[0].detach().cpu().numpy()
    trans_adj = adj[1].detach().cpu().numpy()
    n = cons_adj.shape[0]

    if dij_type == "weighted":
        if vector is None:
            raise ValueError("vector is required for weighted dijkstra.")
        node_weights = vector[0].detach().cpu().numpy()
        avg_nw = np.sum(node_weights) / (2 * n)
        weights = trans_adj[rows, cols] * avg_nw + cons_adj[rows, cols]
    else:
        weights = trans_adj[rows, cols]

    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
    dist = dijkstra(graph, directed=directed, indices=center_node)
    return torch.tensor(dist, dtype=torch.float32, device=adj.device)

def acyclic_mask_from_adj(
    adj: torch.Tensor,
    center_node: int,
    dij_type: str = "trans",
    vector: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dist = dijkstra_distances(adj, center_node, dij_type=dij_type, vector=vector, directed=True)
    dist_np = dist.detach().cpu().numpy()
    order = np.argsort(dist_np)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(dist_np))

    base_mask = _base_edge_mask_np(adj)
    rows, cols = np.nonzero(base_mask)
    allowed = rank[rows] <= rank[cols]
    mask = np.zeros_like(base_mask, dtype=bool)
    mask[rows[allowed], cols[allowed]] = True
    mask[:, center_node] = False
    return torch.tensor(mask, device=adj.device, dtype=torch.bool)

def acyclic_mask_from_prob(
    prob_adj: torch.Tensor,
    base_mask: torch.Tensor,
    center_node: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    mask_np = base_mask.detach().cpu().numpy().astype(bool)
    prob_np = prob_adj.detach().cpu().numpy()
    rows, cols = np.nonzero(mask_np)
    n = prob_np.shape[0]

    weights = -np.log(np.clip(prob_np[rows, cols], eps, 1.0))
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
    dist = dijkstra(graph, directed=True, indices=center_node)
    order = np.argsort(dist)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(dist))

    allowed = rank[rows] <= rank[cols]
    mask = np.zeros_like(mask_np, dtype=bool)
    mask[rows[allowed], cols[allowed]] = True
    mask[:, center_node] = False
    return torch.tensor(mask, device=prob_adj.device, dtype=torch.bool)

def get_mask(adj: torch.Tensor, vector: torch.Tensor, mask_method: str, args):
    """
    INPUT:
    adj(2 x num_nodes x num_nodes) : Tensor
    vector(2 x num_nodes) : Tensor

    OUTPUT:
    mask(num_nodes x num_nodes) : Tensor
    """
    center_node = (vector[1] == 0).nonzero().squeeze().item()

    if mask_method == "acyclic":
        mask_tensor = acyclic_mask_from_adj(
            adj,
            center_node=center_node,
            dij_type=args.dij_type,
            vector=vector,
        )
    elif mask_method == "edge":
        base_mask = _base_edge_mask_np(adj)
        base_mask[:, center_node] = False
        mask_tensor = torch.tensor(base_mask, device=adj.device, dtype=torch.bool)
    else:
        raise SystemExit("No such mask method")
    return mask_tensor


def masked_softmax(prob_adj, mask_adj=None):
    """
    Apply softmax only to elements where mask is True; set others to 0.
    """
    if mask_adj is None:
        mask_adj = torch.ones_like(prob_adj)
    # Ensure mask_adj is boolean
    mask_adj = mask_adj.bool() if not mask_adj.dtype == torch.bool else mask_adj

    # Fill masked entries with -inf
    non_mask_elements = prob_adj.masked_fill(~mask_adj, -torch.inf)

    # Compute softmax
    softmax_result = F.softmax(non_mask_elements, dim=0)
    # Handle fully-masked rows (avoid NaN)
    softmax_result = torch.where(
        ~mask_adj.any(dim=0),  # Check fully-masked rows
        torch.zeros_like(softmax_result),  # Return zeros if fully masked
        softmax_result
    )

    return softmax_result

def differentiable_argmax(pred_adj, mask=None):
    pred_adj = pred_adj.T
    _, max_indices = pred_adj.max(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(pred_adj).scatter_(-1, max_indices, 1.0)
    pred_adj = (y_hard - pred_adj).detach() + pred_adj
    pred_adj = pred_adj.T
    if mask is not None:
        pred_adj = pred_adj * mask
    return pred_adj

class DynamicMaskBuilder:
    def __init__(self, adj: torch.Tensor, center_node: int, device: torch.device, eps: float = 1e-6):
        cons_adj = adj[0].detach().cpu()
        trans_adj = adj[1].detach().cpu()
        base_mask = (cons_adj > 0) & (trans_adj > 0)
        row_idx, col_idx = torch.nonzero(base_mask, as_tuple=True)
        self.rows_np = row_idx.numpy()
        self.cols_np = col_idx.numpy()
        self.rows_t = row_idx.to(device)
        self.cols_t = col_idx.to(device)
        self.cons_edges = cons_adj[row_idx, col_idx].numpy()
        self.trans_edges = trans_adj[row_idx, col_idx].numpy()
        self.n = cons_adj.shape[0]
        self.center_node = int(center_node)
        self.device = device
        self.eps = eps

    def compute_mask(self, pred_adj: torch.Tensor, node_w_sum: torch.Tensor) -> torch.Tensor:
        aw_edges = (node_w_sum[self.rows_t] * pred_adj[self.rows_t, self.cols_t]).detach().cpu().numpy()
        weights = self.cons_edges / (aw_edges + self.eps) + self.trans_edges
        graph = csr_matrix((weights, (self.rows_np, self.cols_np)), shape=(self.n, self.n))
        dist = dijkstra(graph, directed=True, indices=self.center_node)
        order = np.argsort(dist)
        rank = np.empty_like(order)
        rank[order] = np.arange(self.n)
        allowed = rank[self.rows_np] <= rank[self.cols_np]
        mask = np.zeros((self.n, self.n), dtype=bool)
        mask[self.rows_np[allowed], self.cols_np[allowed]] = True
        mask[:, self.center_node] = False
        return torch.tensor(mask, device=self.device, dtype=torch.bool)
