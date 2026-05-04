import heapq
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from numba import njit


@njit(cache=True)
def heap_push_or_decrease(keys, nodes, pos, size, key, node):
    idx = pos[node]
    if idx != -1:
        if key >= keys[idx]:
            return size
        keys[idx] = key
        i = idx
    else:
        i = size
        keys[i] = key
        nodes[i] = node
        pos[node] = i
        size += 1
    while i > 0:
        parent = (i - 1) // 2
        if keys[parent] <= keys[i]:
            break
        keys[parent], keys[i] = keys[i], keys[parent]
        nodes[parent], nodes[i] = nodes[i], nodes[parent]
        pos[nodes[parent]] = parent
        pos[nodes[i]] = i
        i = parent
    return size


@njit(cache=True)
def heap_pop(keys, nodes, pos, size):
    if size == 0:
        return np.float32(0.0), -1, 0
    key = keys[0]
    node = nodes[0]
    pos[node] = -1
    size -= 1
    if size > 0:
        keys[0] = keys[size]
        nodes[0] = nodes[size]
        pos[nodes[0]] = 0
        i = 0
        while True:
            left = 2 * i + 1
            right = left + 1
            smallest = i
            if left < size and keys[left] < keys[smallest]:
                smallest = left
            if right < size and keys[right] < keys[smallest]:
                smallest = right
            if smallest == i:
                break
            keys[i], keys[smallest] = keys[smallest], keys[i]
            nodes[i], nodes[smallest] = nodes[smallest], nodes[i]
            pos[nodes[i]] = i
            pos[nodes[smallest]] = smallest
            i = smallest
    return key, node, size


@njit(cache=True)
def _mark_touched(node, touched, touched_flag, touched_count, dist, prev, settled, heap_pos):
    if touched_flag[node] == 0:
        touched_flag[node] = 1
        touched[touched_count] = node
        touched_count += 1
        dist[node] = np.float32(np.inf)
        prev[node] = -1
        settled[node] = 0
        heap_pos[node] = -1
    return touched_count


@njit(cache=True)
def shortest_path_cost(
    start,
    end,
    coeff,
    indptr,
    indices,
    cons_edges,
    trans_edges,
    dist,
    prev,
    settled,
    heap_key,
    heap_node,
    heap_pos,
    touched,
    touched_flag,
):
    touched_count = 0
    touched_count = _mark_touched(
        start, touched, touched_flag, touched_count, dist, prev, settled, heap_pos
    )
    dist[start] = np.float32(0.0)
    prev[start] = -1
    heap_key[0] = np.float32(0.0)
    heap_node[0] = start
    heap_pos[start] = 0
    heap_size = 1
    best = np.float32(np.inf)

    while heap_size > 0:
        key, node, heap_size = heap_pop(heap_key, heap_node, heap_pos, heap_size)
        if node < 0:
            break
        if settled[node] == 1:
            continue
        if key != dist[node]:
            continue
        settled[node] = 1
        if node == end:
            best = key
            break

        for ei in range(indptr[node], indptr[node + 1]):
            nxt = indices[ei]
            touched_count = _mark_touched(
                nxt, touched, touched_flag, touched_count, dist, prev, settled, heap_pos
            )
            if settled[nxt] == 1:
                continue
            nd = key + cons_edges[ei] + coeff * trans_edges[ei]
            if nd < dist[nxt]:
                dist[nxt] = nd
                prev[nxt] = node
                heap_size = heap_push_or_decrease(
                    heap_key, heap_node, heap_pos, heap_size, nd, nxt
                )

    for i in range(touched_count):
        touched_flag[touched[i]] = 0
    return best


@njit(cache=True)
def build_candidates_exact(
    active_nodes,
    active_weights,
    root,
    indptr,
    indices,
    cons_edges,
    trans_edges,
    n,
):
    active_count = active_nodes.shape[0]
    max_candidates = active_count * (active_count + 1) // 2
    costs = np.empty(max_candidates, dtype=np.float32)
    us = np.empty(max_candidates, dtype=np.int32)
    vs = np.empty(max_candidates, dtype=np.int32)

    dist = np.empty(n, dtype=np.float32)
    prev = np.empty(n, dtype=np.int32)
    settled = np.empty(n, dtype=np.uint8)
    heap_key = np.empty(n, dtype=np.float32)
    heap_node = np.empty(n, dtype=np.int32)
    heap_pos = np.empty(n, dtype=np.int32)
    touched = np.empty(n, dtype=np.int32)
    touched_flag = np.zeros(n, dtype=np.uint8)

    count = 0
    for i in range(active_count):
        u = active_nodes[i]
        w_u = active_weights[i]
        cost = shortest_path_cost(
            root,
            u,
            w_u,
            indptr,
            indices,
            cons_edges,
            trans_edges,
            dist,
            prev,
            settled,
            heap_key,
            heap_node,
            heap_pos,
            touched,
            touched_flag,
        )
        if np.isfinite(cost):
            costs[count] = cost
            us[count] = u
            vs[count] = root
            count += 1

        for j in range(i + 1, active_count):
            v = active_nodes[j]
            w_v = active_weights[j]
            total = w_u + w_v
            coeff = np.float32(0.0)
            if total > 0:
                coeff = np.float32(2.0 * w_u * w_v / total)
            cost_uv = shortest_path_cost(
                u,
                v,
                coeff,
                indptr,
                indices,
                cons_edges,
                trans_edges,
                dist,
                prev,
                settled,
                heap_key,
                heap_node,
                heap_pos,
                touched,
                touched_flag,
            )
            cost_vu = shortest_path_cost(
                v,
                u,
                coeff,
                indptr,
                indices,
                cons_edges,
                trans_edges,
                dist,
                prev,
                settled,
                heap_key,
                heap_node,
                heap_pos,
                touched,
                touched_flag,
            )
            cost = cost_uv
            if cost_vu < cost:
                cost = cost_vu
            if np.isfinite(cost):
                costs[count] = cost
                us[count] = u
                vs[count] = v
                count += 1

    return costs[:count], us[:count], vs[:count]


@njit(cache=True)
def select_matching_sorted(active_nodes, cand_u, cand_v, root, n):
    matched = np.zeros(n, dtype=np.uint8)
    selected_u = np.empty(cand_u.shape[0], dtype=np.int32)
    selected_v = np.empty(cand_v.shape[0], dtype=np.int32)
    sel_count = 0
    root_used = 0

    for idx in range(cand_u.shape[0]):
        u = cand_u[idx]
        v = cand_v[idx]
        if matched[u] == 1:
            continue
        if v == root:
            if root_used == 1:
                continue
            selected_u[sel_count] = u
            selected_v[sel_count] = v
            sel_count += 1
            matched[u] = 1
            root_used = 1
            continue
        if matched[v] == 1:
            continue
        selected_u[sel_count] = u
        selected_v[sel_count] = v
        sel_count += 1
        matched[u] = 1
        matched[v] = 1

    unmatched = np.empty(active_nodes.shape[0], dtype=np.int32)
    unmatched_count = 0
    for idx in range(active_nodes.shape[0]):
        node = active_nodes[idx]
        if matched[node] == 0:
            unmatched[unmatched_count] = node
            unmatched_count += 1

    return selected_u[:sel_count], selected_v[:sel_count], unmatched[:unmatched_count]


@njit(cache=True)
def shortest_path_nodes(
    start,
    end,
    coeff,
    indptr,
    indices,
    cons_edges,
    trans_edges,
    n,
):
    dist = np.full(n, np.float32(np.inf), dtype=np.float32)
    prev = np.full(n, -1, dtype=np.int32)
    settled = np.zeros(n, dtype=np.uint8)
    heap_key = np.empty(n, dtype=np.float32)
    heap_node = np.empty(n, dtype=np.int32)
    heap_pos = np.full(n, -1, dtype=np.int32)

    dist[start] = np.float32(0.0)
    heap_key[0] = np.float32(0.0)
    heap_node[0] = start
    heap_pos[start] = 0
    heap_size = 1

    while heap_size > 0:
        key, node, heap_size = heap_pop(heap_key, heap_node, heap_pos, heap_size)
        if node < 0:
            break
        if settled[node] == 1:
            continue
        if key != dist[node]:
            continue
        settled[node] = 1
        if node == end:
            break
        for ei in range(indptr[node], indptr[node + 1]):
            nxt = indices[ei]
            if settled[nxt] == 1:
                continue
            nd = key + cons_edges[ei] + coeff * trans_edges[ei]
            if nd < dist[nxt]:
                dist[nxt] = nd
                prev[nxt] = node
                heap_size = heap_push_or_decrease(
                    heap_key, heap_node, heap_pos, heap_size, nd, nxt
                )

    if not np.isfinite(dist[end]):
        return np.empty(0, dtype=np.int32)

    rev_nodes = np.empty(n, dtype=np.int32)
    count = 0
    cur = end
    while cur != start:
        rev_nodes[count] = cur
        count += 1
        cur = prev[cur]
        if cur < 0:
            return np.empty(0, dtype=np.int32)
    rev_nodes[count] = start
    count += 1

    path = np.empty(count, dtype=np.int32)
    for i in range(count):
        path[i] = rev_nodes[count - 1 - i]
    return path


class Meyerson:
    """
    Exact Meyerson-Munagala-Plotkin layered aggregation with the paper's
    pair-dependent shortest-path metric and greedy matching variant from
    Section 3.1.

    The algorithm remains O(|S|^2 (m + n log n)) as in the paper. The
    acceleration here is purely constant-factor: CSR storage, numba-jitted
    single-pair Dijkstra, and numba candidate/matching kernels.
    """

    def __init__(self, adj: torch.Tensor, vector: torch.Tensor, args):
        self.args = args
        self.device = args.device
        self.rng = random.Random(args.seed)

        self.cons_adj = adj[0].detach().cpu().numpy().astype(np.float32)
        self.trans_adj = adj[1].detach().cpu().numpy().astype(np.float32)
        self.weights = vector[0].detach().cpu().numpy().astype(np.float32)
        self.node_types = vector[1].detach().cpu().numpy().astype(np.int32)

        self.n = self.cons_adj.shape[0]
        self.root = int(np.where(self.node_types == 0)[0][0])
        self.sources = np.asarray(
            [
                i
                for i in range(self.n)
                if self.node_types[i] == 1 and self.weights[i] > 0
            ],
            dtype=np.int32,
        )
        (
            self.cons_edges,
            self.trans_edges,
            self.indptr,
            self.indices,
        ) = self._build_csr()

        self.used_segments: set[Tuple[int, int]] = set()
        self.solve_info: Dict[str, float | int | str] = {}

    def _build_csr(self):
        base_mask = (self.cons_adj > 0) & (self.trans_adj > 0)
        rows, cols = np.nonzero(base_mask)
        cons_edges = self.cons_adj[rows, cols].astype(np.float32)
        trans_edges = self.trans_adj[rows, cols].astype(np.float32)

        indptr = np.zeros(self.n + 1, dtype=np.int64)
        counts = np.zeros(self.n, dtype=np.int64)
        for r in rows:
            counts[r] += 1
        for i in range(self.n):
            indptr[i + 1] = indptr[i] + counts[i]

        indices = np.zeros(rows.shape[0], dtype=np.int32)
        cons_csr = np.zeros_like(cons_edges)
        trans_csr = np.zeros_like(trans_edges)
        temp_counts = np.zeros(self.n, dtype=np.int64)
        for r, c, cons, trans in zip(rows, cols, cons_edges, trans_edges):
            idx = indptr[r] + temp_counts[r]
            indices[idx] = c
            cons_csr[idx] = cons
            trans_csr[idx] = trans
            temp_counts[r] += 1
        return cons_csr, trans_csr, indptr, indices

    def _pair_coeff(self, w_u: float, w_v: float) -> float:
        total = w_u + w_v
        if total <= 0:
            return 0.0
        return 2.0 * w_u * w_v / total

    def _recover_path(self, start: int, end: int, coeff: float) -> np.ndarray:
        path = shortest_path_nodes(
            start,
            end,
            np.float32(coeff),
            self.indptr,
            self.indices,
            self.cons_edges,
            self.trans_edges,
            self.n,
        )
        if path.size == 0:
            raise RuntimeError(
                f"Meyerson could not recover a shortest path between {start} and {end}."
            )
        return path

    def _recover_path_or_empty(self, start: int, end: int, coeff: float) -> np.ndarray:
        return shortest_path_nodes(
            start,
            end,
            np.float32(coeff),
            self.indptr,
            self.indices,
            self.cons_edges,
            self.trans_edges,
            self.n,
        )

    def _add_path_segments(self, path_nodes: np.ndarray) -> None:
        for a, b in zip(path_nodes[:-1], path_nodes[1:]):
            self.used_segments.add((int(a), int(b)))

    def _choose_pair_center_and_path(
        self,
        u: int,
        v: int,
        w_u: float,
        w_v: float,
        active_count: int,
    ) -> Tuple[int, np.ndarray]:
        coeff = self._pair_coeff(w_u, w_v)
        path_uv = self._recover_path_or_empty(u, v, coeff)
        path_vu = self._recover_path_or_empty(v, u, coeff)
        can_keep_u = path_uv.size > 0
        can_keep_v = path_vu.size > 0

        if not can_keep_u and not can_keep_v:
            raise RuntimeError(
                f"Meyerson could not recover either directed pair path between {u} and {v}."
            )
        if can_keep_u and can_keep_v:
            total = w_u + w_v
            if total <= 0:
                keep_u = self.rng.random() < 0.5
            else:
                keep_u = self.rng.random() < (w_u / total)
            return (int(u), path_uv) if keep_u else (int(v), path_vu)
        if can_keep_u:
            return int(u), path_uv
        return int(v), path_vu

    def _prepare_next_round(
        self,
        active_nodes: np.ndarray,
        active_weights: np.ndarray,
        selected_u: np.ndarray,
        selected_v: np.ndarray,
        unmatched_nodes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        weight_map = {
            int(node): float(weight)
            for node, weight in zip(active_nodes.tolist(), active_weights.tolist())
        }

        next_nodes: List[int] = []
        next_weights: List[float] = []

        for node in unmatched_nodes.tolist():
            next_nodes.append(int(node))
            next_weights.append(weight_map[int(node)])

        for u, v in zip(selected_u.tolist(), selected_v.tolist()):
            if v == self.root:
                path = self._recover_path(self.root, u, weight_map[u])
                self._add_path_segments(path)
                continue

            w_u = weight_map[u]
            w_v = weight_map[v]
            center, path = self._choose_pair_center_and_path(
                u, v, w_u, w_v, int(active_nodes.shape[0])
            )
            self._add_path_segments(path)

            total = w_u + w_v
            next_nodes.append(int(center))
            next_weights.append(float(total))

        if not next_nodes:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )

        order = np.argsort(np.asarray(next_nodes), kind="stable")
        next_nodes_arr = np.asarray(next_nodes, dtype=np.int32)[order]
        next_weights_arr = np.asarray(next_weights, dtype=np.float32)[order]
        return next_nodes_arr, next_weights_arr

    def _run_rounds(self) -> None:
        active_nodes = self.sources.copy()
        active_weights = self.weights[active_nodes].copy()

        stage_count = 0
        total_candidates = 0
        pair_merges = 0
        root_merges = 0

        while active_nodes.size > 0:
            print(
                f"Meyerson stage {stage_count + 1}: active terminals={active_nodes.size}"
            )
            costs, cand_u, cand_v = build_candidates_exact(
                active_nodes,
                active_weights,
                self.root,
                self.indptr,
                self.indices,
                self.cons_edges,
                self.trans_edges,
                self.n,
            )
            total_candidates += int(costs.shape[0])
            if costs.shape[0] == 0:
                raise RuntimeError(
                    "Meyerson could not find any feasible matching candidate."
                )

            order = np.argsort(costs, kind="stable")
            selected_u, selected_v, unmatched_nodes = select_matching_sorted(
                active_nodes,
                cand_u[order],
                cand_v[order],
                self.root,
                self.n,
            )
            if selected_u.shape[0] == 0:
                raise RuntimeError("Meyerson failed to select a matching in this round.")

            for v in selected_v.tolist():
                if v == self.root:
                    root_merges += 1
                else:
                    pair_merges += 1

            active_nodes, active_weights = self._prepare_next_round(
                active_nodes=active_nodes,
                active_weights=active_weights,
                selected_u=selected_u,
                selected_v=selected_v,
                unmatched_nodes=unmatched_nodes,
            )
            stage_count += 1

        self.solve_info = {
            "meyerson_stages": stage_count,
            "meyerson_pair_merges": pair_merges,
            "meyerson_root_merges": root_merges,
            "meyerson_union_segments": len(self.used_segments),
            "meyerson_candidate_paths": total_candidates,
            "meyerson_impl": "exact_numba_greedy_matching_directed_parent_orientation",
        }

    def _build_union_neighbors(self) -> List[List[Tuple[int, float]]]:
        union_neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(self.n)]
        for a, b in sorted(self.used_segments):
            if self.trans_adj[a, b] > 0:
                union_neighbors[a].append((b, float(self.trans_adj[a, b])))
        return union_neighbors

    def _dijkstra_on_union(self) -> np.ndarray:
        union_neighbors = self._build_union_neighbors()
        dist = np.full(self.n, np.inf, dtype=np.float64)
        prev = np.full(self.n, -1, dtype=np.int32)
        dist[self.root] = 0.0
        heap: List[Tuple[float, int]] = [(0.0, self.root)]

        while heap:
            cur_dist, node = heapq.heappop(heap)
            if cur_dist > dist[node]:
                continue
            for nxt, trans_cost in union_neighbors[node]:
                nd = cur_dist + trans_cost
                if nd >= dist[nxt]:
                    continue
                dist[nxt] = nd
                prev[nxt] = node
                heapq.heappush(heap, (nd, nxt))

        for src in self.sources.tolist():
            if not np.isfinite(dist[src]):
                raise RuntimeError(
                    f"Meyerson produced a disconnected union graph; source {src} "
                    "cannot reach the sink."
                )
        return prev

    def _extract_pred_adj(self, prev: np.ndarray) -> torch.Tensor:
        pred_adj = torch.zeros((self.n, self.n), dtype=torch.float32, device=self.device)
        needed_edges: set[Tuple[int, int]] = set()

        for src in self.sources.tolist():
            cur = int(src)
            while cur != self.root:
                parent = int(prev[cur])
                if parent < 0:
                    raise RuntimeError(
                        f"Meyerson failed to recover a root path for source {src}."
                    )
                needed_edges.add((parent, cur))
                cur = parent

        for parent, child in needed_edges:
            pred_adj[parent, child] = 1.0

        root_col_sum = float(pred_adj[:, self.root].sum().item())
        if root_col_sum != 0.0:
            raise RuntimeError(
                f"Meyerson root column must sum to 0, got {root_col_sum}."
            )
        col_sums = pred_adj.sum(dim=0)
        non_root_mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        non_root_mask[self.root] = False
        if not torch.all((col_sums[non_root_mask] == 0) | (col_sums[non_root_mask] == 1)):
            bad_cols = (
                torch.nonzero((col_sums != 0) & (col_sums != 1), as_tuple=False)
                .squeeze(-1)
                .tolist()
            )
            raise RuntimeError(
                f"Meyerson produced invalid parent columns: {bad_cols[:10]}"
            )
        return pred_adj

    def predict(self) -> torch.Tensor:
        if self.sources.size == 0:
            return torch.zeros((self.n, self.n), dtype=torch.float32, device=self.device)

        self._run_rounds()
        prev = self._dijkstra_on_union()
        return self._extract_pred_adj(prev)
