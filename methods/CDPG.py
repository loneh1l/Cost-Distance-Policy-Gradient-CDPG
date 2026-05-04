import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from numba import njit
from utils.loss import ValueSolver
from utils.exp_tracker import LocalTracker
from utils.encode import (
    acyclic_mask_from_adj,
    dijkstra_distances,
    DynamicMaskBuilder,
    masked_softmax,
)


@njit(cache=True)
def _round_topo_greedy_numba(
    indptr,
    parents,
    probs,
    cons,
    trans,
    topo_order,
    weights,
):
    n = weights.shape[0]
    expected = np.zeros(n, dtype=np.float32)

    for rev_idx in range(topo_order.shape[0] - 1, -1, -1):
        node = topo_order[rev_idx]
        start = indptr[node]
        end = indptr[node + 1]
        if start == end:
            continue
        total = 0.0
        for edge_idx in range(start, end):
            prob = probs[edge_idx]
            if prob <= 0.0:
                continue
            parent = parents[edge_idx]
            total += prob * (trans[edge_idx] + expected[parent])
        expected[node] = total

    work_weights = weights.copy()
    chosen_parent = np.full(n, -1, dtype=np.int32)
    for order_idx in range(topo_order.shape[0]):
        node = topo_order[order_idx]
        k_i = work_weights[node]
        if k_i <= 0.0:
            continue
        start = indptr[node]
        end = indptr[node + 1]
        if start == end:
            continue
        best_parent = -1
        best_score = 0.0
        found = False
        for edge_idx in range(start, end):
            prob = probs[edge_idx]
            if prob <= 0.0:
                continue
            parent = parents[edge_idx]
            score = cons[edge_idx] + k_i * (trans[edge_idx] + expected[parent])
            if (not found) or (score < best_score):
                best_score = score
                best_parent = parent
                found = True
        if found:
            chosen_parent[node] = best_parent
            work_weights[best_parent] += k_i

    return chosen_parent


class CDPG:
    def __init__(self, adj:torch.Tensor, vector:torch.Tensor, mask:torch.Tensor, args, graph=None):
        lora_rank = args.lora_rank
        self.adj = adj
        self.vector = vector
        self.args = args
        self.mask = mask
        self.device = args.device
        self.d = torch.tensor(lora_rank).to(self.device)
        with torch.no_grad():
            if args.initial_point_type == 'random':
                edge_attr = torch.rand_like(self.adj[0])
            elif args.initial_point_type == 'cons':
                edge_attr = self.adj[0].clone()
            else:
                edge_attr = self.adj[1].clone()
            edge_attr = torch.exp(-edge_attr)
            edge_attr = edge_attr * mask
        self.edge_attr = edge_attr.to(self.device).requires_grad_()
        n = self.adj[0].shape[0]
        self.lora_Q = torch.randn(n, lora_rank).to(self.device).requires_grad_()
        self.lora_K = torch.zeros(lora_rank, n).to(self.device).requires_grad_()
        params = [self.edge_attr, self.lora_Q, self.lora_K]
        optimizer_name = getattr(args, "optimizer", "rmsprop").lower()
        if optimizer_name == "rmsprop":
            self.optimizer = optim.RMSprop(params=params, lr=args.lr)
        elif optimizer_name == "adagrad":
            self.optimizer = optim.Adagrad(params=params, lr=args.lr)
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(params=params, lr=args.lr)
        elif optimizer_name == "nadam":
            self.optimizer = optim.NAdam(params=params, lr=args.lr)
        else:
            raise ValueError(f"Unsupported optimizer for CDPG: {optimizer_name}")
        self.center_node = (vector[1] == 0).nonzero().squeeze().item()
        self.criterion = ValueSolver(adj, vector, graph)
        self.dynamic_mask_builder = None
        if args.dynamic_mask and args.mask_method == "acyclic":
            self.dynamic_mask_builder = DynamicMaskBuilder(
                adj=adj,
                center_node=self.center_node,
                device=self.device,
            )
        log_dir = args.log_dir
        self.tracker = LocalTracker(log_dir)
        print("initial done")

    def train(self):
        args = self.args
        use_lora = args.use_lora
        train_iterations = int(args.train_iterations)
        track_id = 0
        step_counter = 0
        mask_interval = max(int(args.dynamic_mask_interval), 1)
        loss_check_history = []
        if train_iterations <= 0:
            return
        prev_loss = None
        progress_bar = tqdm(range(train_iterations), desc="Optimization", dynamic_ncols=True)
        for epoch in progress_bar:
            step_counter += 1
            self.optimizer.zero_grad()
            pred_adj = self.forward(use_lora)
            args.loss_unreached = args.use_unreached
            node_w_sum = None
            if args.use_unreached:
                if self.dynamic_mask_builder is not None:
                    trans_loss, cons_loss, unreached, rls, node_w_sum = self.criterion.loss_with_aw(pred_adj, args)
                else:
                    trans_loss, cons_loss, unreached, rls = self.criterion.loss(pred_adj, args)
            else:
                if self.dynamic_mask_builder is not None:
                    trans_loss, cons_loss, rls, node_w_sum = self.criterion.loss_with_aw(pred_adj, args)
                else:
                    trans_loss, cons_loss, rls = self.criterion.loss(pred_adj, args)
                unreached = torch.tensor(0).to(self.device)
            loss = trans_loss + cons_loss + unreached
            loss_check_history.append(loss.item())
            loss.backward()

            self.optimizer.step()

            if self.dynamic_mask_builder is not None and node_w_sum is not None:
                if step_counter % mask_interval == 0:
                    with torch.no_grad():
                        new_mask = self.dynamic_mask_builder.compute_mask(pred_adj, node_w_sum)
                        should_update = True
                        if getattr(args, "dynamic_mask_loss_check", True):
                            check_steps = int(getattr(args, "dynamic_mask_loss_check_steps", 1))
                            if check_steps < 0:
                                raise ValueError("dynamic_mask_loss_check_steps must be >= 0")
                            history_index = len(loss_check_history) - 1 - check_steps
                            if history_index >= 0:
                                baseline_loss = loss_check_history[history_index]
                            else:
                                baseline_loss = loss_check_history[0]
                            new_loss = self._loss_for_mask(new_mask, use_lora)
                            should_update = new_loss.item() < baseline_loss
                        if should_update:
                            self.mask = new_mask

            unreached_str = f"{unreached:.4f}" if args.use_unreached else "None"
            progress_bar.set_postfix(
                OrderedDict([
                    ("trans_loss", f"{trans_loss:.4f}"),
                    ("cons_loss", f"{cons_loss:.4f}"),
                    ("Unreached", unreached_str),
                    ("loss", f"{loss:.4f}"),
                ]),
                refresh=True
            )
            if not args.fixed_train_iters:
                loss_val = loss.item()
                if prev_loss is not None:
                    tol = args.train_early_stop_tol
                    denom = max(abs(loss_val), 1e-12)
                    if abs(loss_val - prev_loss) < denom * tol:
                        progress_bar.close()
                        break
                prev_loss = loss_val
            track_id += 1
            self.tracker.track(
                step=track_id,
                trans_loss=trans_loss.item(),
                cons_loss=cons_loss.item(),
                unreached=unreached.item(),
                loss=loss.item(),
                rls=rls.item()
            )
        self.tracker.plot(output="training_curves.png", title="Training Metrics")

    def forward(self, use_lora=False, mask=None):
        args = self.args
        active_mask = self.mask if mask is None else mask
        if use_lora:
            lora_P = torch.mm(self.lora_Q, self.lora_K) / torch.sqrt(self.d) * args.alpha
        else:
            lora_P = None

        pred_adj = active_mask * self.edge_attr
        if lora_P is not None:
            lora_P = torch.mul(lora_P, active_mask)
            pred_adj = pred_adj + lora_P

        result = masked_softmax(pred_adj, active_mask)
        center_mask = torch.zeros_like(result, dtype=torch.bool)
        center_mask[:, self.center_node] = True
        center_mask_result = torch.where(center_mask, torch.tensor(0.0), result)
        return center_mask_result

    def _loss_for_mask(self, mask: torch.Tensor, use_lora: bool) -> torch.Tensor:
        pred_adj = self.forward(use_lora=use_lora, mask=mask)
        if self.args.use_unreached:
            trans_loss, cons_loss, unreached, _rls = self.criterion.loss(pred_adj, self.args)
        else:
            trans_loss, cons_loss, _rls = self.criterion.loss(pred_adj, self.args)
            unreached = torch.tensor(0.0, device=self.device)
        return trans_loss + cons_loss + unreached

    def _topological_order(self, mask: torch.Tensor) -> torch.Tensor:
        mask_cpu = mask.detach().cpu()
        n = mask_cpu.shape[0]
        valid_nodes = (mask_cpu.any(dim=0) | mask_cpu.any(dim=1)).numpy()
        valid_nodes[self.center_node] = True

        row_idx, col_idx = torch.nonzero(mask_cpu, as_tuple=True)
        row_np = row_idx.numpy()
        col_np = col_idx.numpy()

        indegree = np.zeros(n, dtype=np.int64)
        outgoing = [[] for _ in range(n)]
        for parent, child in zip(row_np, col_np):
            if not valid_nodes[parent] or not valid_nodes[child]:
                continue
            indegree[child] += 1
            outgoing[parent].append(child)

        queue = [node for node in range(n) if valid_nodes[node] and indegree[node] == 0]
        topo = []
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            topo.append(node)
            for child in outgoing[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(topo) != int(valid_nodes.sum()):
            raise ValueError("Prediction mask is not acyclic; topological order could not be constructed.")

        topo = [node for node in reversed(topo) if node != self.center_node]
        return torch.tensor(topo, device=mask.device, dtype=torch.long)

    def _expected_node_distance(self, prob_adj: torch.Tensor, mask: torch.Tensor, topo_order: torch.Tensor) -> torch.Tensor:
        expected = torch.zeros(prob_adj.shape[0], device=prob_adj.device, dtype=prob_adj.dtype)
        trans_adj = self.adj[1]
        for node in reversed(topo_order.tolist()):
            parents = torch.where(mask[:, node])[0]
            if parents.numel() == 0:
                continue
            probs = prob_adj[parents, node]
            if probs.sum() <= 0:
                continue
            expected[node] = torch.sum(probs * (trans_adj[parents, node] + expected[parents]))
        return expected

    def _build_incoming_csr(self, mask: torch.Tensor):
        mask_cpu = mask.detach().cpu()
        row_idx, col_idx = torch.nonzero(mask_cpu, as_tuple=True)
        order = torch.argsort(col_idx * mask_cpu.shape[0] + row_idx)
        row_idx = row_idx[order]
        col_idx = col_idx[order]
        n = mask_cpu.shape[0]
        counts = torch.bincount(col_idx, minlength=n)
        indptr = torch.zeros(n + 1, dtype=torch.int64)
        indptr[1:] = torch.cumsum(counts, dim=0)
        return (
            indptr.numpy().astype(np.int64, copy=False),
            row_idx.numpy().astype(np.int32, copy=False),
            col_idx.numpy().astype(np.int32, copy=False),
        )

    def _round_topo_greedy(self, prob_adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        topo_order = self._topological_order(mask)
        indptr, parents, cols = self._build_incoming_csr(mask)
        prob_cpu = prob_adj.detach().cpu()
        cons_cpu = self.adj[0].detach().cpu()
        trans_cpu = self.adj[1].detach().cpu()
        weights_cpu = self.vector[0].detach().cpu()

        probs = prob_cpu[parents, cols].numpy().astype(np.float32, copy=False)
        cons = cons_cpu[parents, cols].numpy().astype(np.float32, copy=False)
        trans = trans_cpu[parents, cols].numpy().astype(np.float32, copy=False)
        topo_np = topo_order.detach().cpu().numpy().astype(np.int32, copy=False)
        weights = weights_cpu.numpy().astype(np.float32, copy=False)

        chosen_parent = _round_topo_greedy_numba(
            indptr=indptr,
            parents=parents,
            probs=probs,
            cons=cons,
            trans=trans,
            topo_order=topo_np,
            weights=weights,
        )

        hard_adj = torch.zeros_like(prob_adj)
        child_nodes = np.nonzero(chosen_parent >= 0)[0]
        if child_nodes.size > 0:
            parent_nodes = chosen_parent[child_nodes]
            hard_adj[
                torch.from_numpy(parent_nodes).to(prob_adj.device),
                torch.from_numpy(child_nodes.astype(np.int64)).to(prob_adj.device),
            ] = 1.0
        return hard_adj * mask

    def _round_argmax(self, prob_adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        print("use argmax")
        pred_adj = prob_adj.T
        _, max_indices = pred_adj.max(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(pred_adj).scatter_(-1, max_indices, 1.0)
        pred_adj = (y_hard - pred_adj).detach() + pred_adj
        pred_adj = pred_adj.T
        return pred_adj * mask

    def _prune_zero_weight_leaves(self, pred_adj: torch.Tensor) -> torch.Tensor:
        node_weights = self.vector[0]
        child_counts = torch.sum(pred_adj != 0, dim=1)
        stack = [
            i for i in range(pred_adj.shape[0])
            if child_counts[i] == 0 and node_weights[i] == 0 and i != self.center_node
        ]

        while stack:
            node = stack.pop()
            parents = torch.where(pred_adj[:, node] > 0)[0]
            for parent in parents:
                pred_adj[parent, node] = 0
                child_counts[parent] -= 1
                parent_idx = int(parent.item())
                if (
                    parent_idx != self.center_node
                    and child_counts[parent] <= 0
                    and node_weights[parent] == 0
                ):
                    stack.append(parent_idx)
        return pred_adj

    def _renormalize_columns(self, prob_adj: torch.Tensor) -> torch.Tensor:
        col_sum = prob_adj.sum(dim=0, keepdim=True)
        return torch.where(
            col_sum > 0,
            prob_adj / col_sum.clamp_min(1e-12),
            prob_adj,
        )

    def predict(self):
        args = self.args
        prob_adj = self.forward(use_lora=args.use_lora)
        predict_mode = getattr(args, "predict_mode", "topo_greedy")

        if args.mask_method == "acyclic":
            pred_mask = self.mask
        else:
            pred_mask = acyclic_mask_from_adj(
                self.adj,
                center_node=self.center_node,
                dij_type=args.dij_type,
                vector=self.vector,
            )

        prob_adj = prob_adj * pred_mask

        if predict_mode == "argmax":
            return self._prune_zero_weight_leaves(self._round_argmax(prob_adj, pred_mask))

        prob_adj = self._renormalize_columns(prob_adj)
        print("use topology-aware deterministic policy")
        return self._round_topo_greedy(prob_adj, pred_mask)


def check_matrix_columns(matrix, atol=1e-6, max_show=20):
    """
    Check column sums:
    - Column 0 must sum to 0
    - Other columns must sum to 1
    If violated:
      - Report which columns are wrong
      - Report their sums
    Args:
      atol: tolerated float error
      max_show: max number of bad columns to show
    """
    col_sums = torch.sum(matrix, dim=0)  # [num_cols]

    num_cols = col_sums.numel()

    # ---- Check column 0: should be 0 ----
    ok_first = torch.isclose(col_sums[0], torch.tensor(0., device=col_sums.device), atol=atol)
    if not ok_first:
        raise ValueError(
            f"Column 0 sum must be 0, got {col_sums[0].item():.6f}"
        )

    # ---- Check columns 1..n-1: should be 1 ----
    if num_cols > 1:
        expected = torch.ones_like(col_sums[1:])
        ok_rest = torch.isclose(col_sums[1:], expected, atol=atol)

        if not torch.all(ok_rest):
            # Find violating columns (real index = mask index + 1)
            bad_indices = torch.nonzero(~ok_rest).squeeze(-1)
            bad_indices_full = (bad_indices + 1).tolist()  # Shift to real column index

            # Show a subset
            show_indices = bad_indices_full[:max_show]
            show_sums = col_sums[show_indices].tolist()

            msg = (
                f"Columns 1..{num_cols-1}: {len(bad_indices_full)} columns have sum != 1 (atol={atol}).\\n"
                f"First {len(show_indices)} examples:\\n"
            )
            for idx, s in zip(show_indices, show_sums):
                msg += f"  - Column {idx} sum = {s:.6f}\\n"

            if len(bad_indices_full) > max_show:
                msg += f"... {len(bad_indices_full) - max_show} more columns not shown.\\n"

            raise ValueError(msg)

    # print(\"Column-sum check passed.\")
    return True
