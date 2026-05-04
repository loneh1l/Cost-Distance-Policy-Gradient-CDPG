import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from numba import njit
from tqdm import tqdm


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
        return 0.0, -1, 0
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
def top_push(keys, srcs, nodes, vers, size, key, src, node, ver):
    i = size
    keys[i] = key
    srcs[i] = src
    nodes[i] = node
    vers[i] = ver
    size += 1
    while i > 0:
        parent = (i - 1) // 2
        if keys[parent] <= keys[i]:
            break
        keys[parent], keys[i] = keys[i], keys[parent]
        srcs[parent], srcs[i] = srcs[i], srcs[parent]
        nodes[parent], nodes[i] = nodes[i], nodes[parent]
        vers[parent], vers[i] = vers[i], vers[parent]
        i = parent
    return size


@njit(cache=True)
def top_pop(keys, srcs, nodes, vers, size):
    if size == 0:
        return 0.0, -1, -1, -1, 0
    key = keys[0]
    src = srcs[0]
    node = nodes[0]
    ver = vers[0]
    size -= 1
    if size > 0:
        keys[0] = keys[size]
        srcs[0] = srcs[size]
        nodes[0] = nodes[size]
        vers[0] = vers[size]
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
            srcs[i], srcs[smallest] = srcs[smallest], srcs[i]
            nodes[i], nodes[smallest] = nodes[smallest], nodes[i]
            vers[i], vers[smallest] = vers[smallest], vers[i]
            i = smallest
    return key, src, node, ver, size


@njit(cache=True)
def beta_fn(w1, w2, eta, d_bif):
    w_max = w1 if w1 >= w2 else w2
    w_min = w2 if w1 >= w2 else w1
    return d_bif * (eta * w_max + (1.0 - eta) * w_min)


@njit(cache=True)
def held_round(
    n,
    indptr,
    indices,
    cons_edges,
    trans_edges,
    root_rep_id,
    src_active,
    src_node,
    src_weight,
    dist_key,
    prev,
    settled,
    heap_key,
    heap_node,
    heap_pos,
    heap_size,
    version,
    top_key,
    top_src,
    top_node,
    top_ver,
    top_size,
    best_val,
    best_v_id_arr,
    best_meet_node_arr,
    term_head,
    term_next,
    eta,
    d_bif,
    total_weight,
    settled_count,
):
    active_ids = np.where(src_active == 1)[0]

    best_L = 1e100
    best_u_id = -1
    best_v_id = -1
    best_meet_node = -1

    for sid in active_ids:
        v_id = best_v_id_arr[sid]
        if v_id >= 0 and src_active[v_id] == 1:
            cand = best_val[sid]
            if cand < best_L:
                best_L = cand
                best_u_id = sid
                best_v_id = v_id
                best_meet_node = best_meet_node_arr[sid]
        if settled[sid, src_node[root_rep_id]] == 1:
            b_root = beta_fn(src_weight[sid], total_weight - src_weight[sid], eta, d_bif)
            cand = dist_key[sid, src_node[root_rep_id]] + b_root
            if cand < best_L:
                best_L = cand
                best_u_id = sid
                best_v_id = root_rep_id
                best_meet_node = src_node[root_rep_id]

    top_size = 0
    for sid in active_ids:
        if heap_size[sid] == 0:
            continue
        version[sid] += 1
        key = heap_key[sid, 0]
        node = heap_node[sid, 0]
        top_size = top_push(top_key, top_src, top_node, top_ver, top_size, key, sid, node, version[sid])

    while top_size > 0:
        key, sid, node, ver, top_size = top_pop(top_key, top_src, top_node, top_ver, top_size)
        if src_active[sid] == 0:
            continue
        if ver != version[sid]:
            continue
        if heap_size[sid] == 0:
            continue
        if heap_key[sid, 0] != key or heap_node[sid, 0] != node:
            version[sid] += 1
            key2 = heap_key[sid, 0]
            node2 = heap_node[sid, 0]
            top_size = top_push(top_key, top_src, top_node, top_ver, top_size, key2, sid, node2, version[sid])
            continue
        if best_L < 1e100 and key >= best_L:
            break

        key, u_node, heap_size[sid] = heap_pop(heap_key[sid], heap_node[sid], heap_pos[sid], heap_size[sid])
        if settled[sid, u_node] == 1:
            continue
        if key != dist_key[sid, u_node]:
            continue

        settled[sid, u_node] = 1
        settled_count[sid] += 1

        v_id = term_head[u_node]
        while v_id != -1:
            if v_id != sid:
                if v_id == root_rep_id:
                    b_root = beta_fn(src_weight[sid], total_weight - src_weight[sid], eta, d_bif)
                    cand = dist_key[sid, u_node] + b_root
                    if cand < best_L:
                        best_L = cand
                        best_u_id = sid
                        best_v_id = v_id
                        best_meet_node = u_node
                elif src_active[v_id] == 1:
                    b_uv = beta_fn(src_weight[sid], src_weight[v_id], eta, d_bif)
                    cand = dist_key[sid, u_node] + b_uv
                    if cand < best_val[sid]:
                        best_val[sid] = cand
                        best_v_id_arr[sid] = v_id
                        best_meet_node_arr[sid] = u_node
                    if cand < best_L:
                        best_L = cand
                        best_u_id = sid
                        best_v_id = v_id
                        best_meet_node = u_node
            v_id = term_next[v_id]

        for ei in range(indptr[u_node], indptr[u_node + 1]):
            v = indices[ei]
            if settled[sid, v] == 1:
                continue
            new_key = dist_key[sid, u_node] + cons_edges[ei] + src_weight[sid] * trans_edges[ei]
            if new_key < dist_key[sid, v]:
                dist_key[sid, v] = new_key
                prev[sid, v] = u_node
                heap_size[sid] = heap_push_or_decrease(
                    heap_key[sid], heap_node[sid], heap_pos[sid], heap_size[sid], new_key, v
                )

        if heap_size[sid] > 0:
            version[sid] += 1
            key2 = heap_key[sid, 0]
            node2 = heap_node[sid, 0]
            top_size = top_push(top_key, top_src, top_node, top_ver, top_size, key2, sid, node2, version[sid])

    return best_u_id, best_v_id, best_meet_node


class Held:
    def __init__(self, adj: torch.Tensor, vector: torch.Tensor, args):
        self.adj = adj
        self.vector = vector
        self.args = args
        self.device = args.device
        self.rng = random.Random(args.seed)
        self.cons_adj = adj[0].detach().cpu().numpy().astype(np.float32)
        self.trans_adj = adj[1].detach().cpu().numpy().astype(np.float32)
        self.n = self.cons_adj.shape[0]
        self.root = int((vector[1] == 0).nonzero().squeeze().item())
        self.sources = [
            i for i in range(self.n)
            if vector[1][i].item() == 1 and vector[0][i].item() > 0
        ]
        self.weights = vector[0].detach().cpu().numpy().astype(np.float32)
        (
            self.rows_csr,
            self.cols_csr,
            self.cons_edges,
            self.trans_edges,
            self.indptr,
            self.indices,
        ) = self._build_csr()
        self.edge_idx = self._build_edge_index()
        if self.trans_edges.size == 0:
            self.d_bif_scaled = 0.0
        else:
            self.d_bif_scaled = float(self.trans_edges.mean()) * float(self.args.d_bif)

        self.tree_edges: List[Tuple[int, int]] = []
        self.term_node: List[int] = []
        self.ds_created = 0
        self.settled_counts: List[int] = []

    def _build_csr(self):
        base_mask = (self.cons_adj > 0) & (self.trans_adj > 0)
        rows, cols = np.nonzero(base_mask)
        # pred_adj[parent, child] uses adj[parent, child], while flow moves from
        # child to parent. Held expands searches from terminals toward merge
        # points, so the search graph must traverse the reverse of adj.
        search_rows = cols
        search_cols = rows
        cons_edges = self.cons_adj[rows, cols].astype(np.float32)
        trans_edges = self.trans_adj[rows, cols].astype(np.float32)
        indptr = np.zeros(self.n + 1, dtype=np.int64)
        indices = np.zeros(search_rows.shape[0], dtype=np.int32)
        counts = np.zeros(self.n, dtype=np.int64)
        for r in search_rows:
            counts[r] += 1
        indptr[0] = 0
        for i in range(self.n):
            indptr[i + 1] = indptr[i] + counts[i]
        temp_counts = np.zeros(self.n, dtype=np.int64)
        for r, c in zip(search_rows, search_cols):
            idx = indptr[r] + temp_counts[r]
            indices[idx] = c
            temp_counts[r] += 1
        cons_csr = np.zeros_like(cons_edges)
        trans_csr = np.zeros_like(trans_edges)
        rows_csr = np.zeros_like(indices)
        cols_csr = np.zeros_like(indices)
        temp_counts[:] = 0
        for r, c, cons, trans in zip(search_rows, search_cols, cons_edges, trans_edges):
            idx = indptr[r] + temp_counts[r]
            cons_csr[idx] = cons
            trans_csr[idx] = trans
            rows_csr[idx] = r
            cols_csr[idx] = c
            temp_counts[r] += 1
        return rows_csr, cols_csr, cons_csr, trans_csr, indptr, indices

    def _build_edge_index(self) -> Dict[Tuple[int, int], int]:
        edge_idx: Dict[Tuple[int, int], int] = {}
        for u in range(self.n):
            for idx in range(self.indptr[u], self.indptr[u + 1]):
                v = int(self.indices[idx])
                edge_idx[(u, v)] = idx
        return edge_idx

    def _init_ds(self, sid: int, node: int, dist_key: np.ndarray, prev: np.ndarray,
                 settled: np.ndarray, heap_key: np.ndarray, heap_node: np.ndarray,
                 heap_pos: np.ndarray, heap_size: np.ndarray) -> None:
        dist_key[sid].fill(np.inf)
        prev[sid].fill(-1)
        settled[sid].fill(0)
        heap_key[sid].fill(np.inf)
        heap_node[sid].fill(-1)
        heap_pos[sid].fill(-1)
        dist_key[sid, node] = 0.0
        heap_key[sid, 0] = 0.0
        heap_node[sid, 0] = node
        heap_pos[sid, node] = 0
        heap_size[sid] = 1

    def _build_term_lists(
        self,
        n: int,
        active_ids: np.ndarray,
        root_rep_id: int,
        term_node: np.ndarray,
        term_head: np.ndarray,
        term_next: np.ndarray,
    ) -> None:
        term_head.fill(-1)
        term_next.fill(-1)
        term_ids = np.concatenate([active_ids, np.array([root_rep_id], dtype=np.int32)])
        nodes = term_node[term_ids]
        order = np.lexsort((term_ids, nodes))
        for idx in order[::-1]:
            tid = term_ids[idx]
            node = term_node[tid]
            term_next[tid] = term_head[node]
            term_head[node] = tid

    def _reconstruct_path(self, prev: np.ndarray, src: int, dst: int) -> List[int]:
        path = []
        cur = dst
        while cur >= 0:
            path.append(cur)
            if cur == src:
                break
            nxt = int(prev[cur])
            if nxt < 0:
                break
            cur = nxt
        if not path or path[-1] != src:
            return []
        path.reverse()
        return path

    def _verify(self, root_node: int, sources: List[int], tree_edges: List[Tuple[int, int]],
                ds_created: int, max_ds: int, settled_counts: List[int]) -> None:
        if ds_created > max_ds:
            raise RuntimeError(f"DS created {ds_created} exceeds limit {max_ds}.")
        pred_adj = self._build_parent_adj_from_edges(tree_edges)
        self._verify_parent_adj(pred_adj)
        child_to_parent = [
            torch.where(pred_adj[:, child] > 0)[0].detach().cpu().tolist()
            for child in range(self.n)
        ]
        disconnected = []
        for source in sources:
            visited = [False] * self.n
            stack = [source]
            visited[source] = True
            reached_root = False
            while stack:
                node = stack.pop()
                if node == root_node:
                    reached_root = True
                    break
                for nxt in child_to_parent[node]:
                    if visited[nxt]:
                        continue
                    visited[nxt] = True
                    stack.append(nxt)
            if not reached_root:
                disconnected.append(source)
        if disconnected:
            raise RuntimeError(f"Held produced disconnected directed tree; unreachable sources: {disconnected}")
        self.settled_counts = settled_counts

    def _path_to_parent_edges(self, path_nodes: List[int]) -> List[Tuple[int, int]]:
        return [(parent, child) for child, parent in zip(path_nodes, path_nodes[1:])]

    def _path_exists_in_solution(self, path_nodes: List[int], selected_edges: set[Tuple[int, int]]) -> bool:
        return all((parent, child) in selected_edges for child, parent in zip(path_nodes, path_nodes[1:]))

    def _component_nodes(self, start_node: int, selected_edges: set[Tuple[int, int]]) -> set[int]:
        undirected = [[] for _ in range(self.n)]
        for parent, child in selected_edges:
            undirected[parent].append(child)
            undirected[child].append(parent)
        visited = {int(start_node)}
        stack = [int(start_node)]
        while stack:
            node = stack.pop()
            for nxt in undirected[node]:
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)
        return visited

    def _verify_parent_adj(self, pred_adj: torch.Tensor) -> None:
        col_sums = pred_adj.sum(dim=0)
        root_col_sum = float(col_sums[self.root].item())
        if root_col_sum != 0.0:
            raise RuntimeError(f"Held root column must sum to 0, got {root_col_sum}.")
        non_root_cols = torch.cat([col_sums[:self.root], col_sums[self.root + 1:]])
        if non_root_cols.numel() and not torch.all((non_root_cols == 0) | (non_root_cols == 1)):
            bad_cols = torch.nonzero((col_sums != 0) & (col_sums != 1), as_tuple=False).squeeze(-1).tolist()
            raise RuntimeError(f"Held produced invalid parent columns: {bad_cols[:10]}")
        base_mask = torch.tensor(
            (self.cons_adj > 0) & (self.trans_adj > 0),
            dtype=torch.bool,
            device=pred_adj.device,
        )
        invalid = (pred_adj > 0) & (~base_mask)
        if invalid.any():
            bad_edges = torch.nonzero(invalid, as_tuple=False).squeeze(0).tolist()
            raise RuntimeError(f"Held produced nonexistent directed edges: {bad_edges[:10]}")

    def _build_parent_adj_from_edges(self, tree_edges: List[Tuple[int, int]]) -> torch.Tensor:
        parent_to_children = [[] for _ in range(self.n)]
        for parent, child in tree_edges:
            if self.cons_adj[parent, child] > 0 and self.trans_adj[parent, child] > 0:
                parent_to_children[parent].append(child)
            if self.cons_adj[child, parent] > 0 and self.trans_adj[child, parent] > 0:
                parent_to_children[child].append(parent)

        selected_parent: Dict[int, int] = {}
        visited = {int(self.root)}
        stack = [int(self.root)]
        while stack:
            parent = stack.pop()
            for child in parent_to_children[parent]:
                if child in visited:
                    continue
                visited.add(child)
                selected_parent[child] = parent
                stack.append(child)

        missing_sources = [source for source in self.sources if source not in visited]
        if missing_sources:
            raise RuntimeError(
                f"Held could not orient selected tree toward root; unreachable sources: {missing_sources[:10]}"
            )

        pred_adj = torch.zeros((self.n, self.n), dtype=torch.float32, device=self.device)
        for child, parent in selected_parent.items():
            pred_adj[parent, child] = 1.0
        return pred_adj

    def _discount_path_edges(self, path_nodes: List[int], cons_edges_work: np.ndarray) -> None:
        for a, b in zip(path_nodes, path_nodes[1:]):
            idx_ab = self.edge_idx.get((a, b))
            if idx_ab is not None:
                cons_edges_work[idx_ab] = 0.0
            idx_ba = self.edge_idx.get((b, a))
            if idx_ba is not None:
                cons_edges_work[idx_ba] = 0.0

    def _choose_steiner_rep(self, path_nodes: List[int], u_w: float, v_w: float) -> int:
        if len(path_nodes) <= 1:
            return int(path_nodes[0])
        prefix = [0.0]
        for a, b in zip(path_nodes, path_nodes[1:]):
            prefix.append(prefix[-1] + float(self.trans_adj[a, b]))
        total = prefix[-1]
        best_idx = 0
        best_val = float('inf')
        for idx, dist_u in enumerate(prefix):
            dist_v = total - dist_u
            val = u_w * dist_u + v_w * dist_v
            if val < best_val:
                best_val = val
                best_idx = idx
        return int(path_nodes[best_idx])

    def _debug_no_candidate(self, src_active: np.ndarray, src_node: np.ndarray,
                            heap_size: np.ndarray, settled: np.ndarray, root_rep_id: int) -> None:
        active_ids = np.where(src_active == 1)[0]
        root_node = int(src_node[root_rep_id])
        active_nodes = src_node[active_ids]
        heap_nonempty = int(np.sum(heap_size[active_ids] > 0))
        settled_root = int(np.sum(settled[active_ids, root_node] == 1))
        base_mask = (self.cons_adj > 0) & (self.trans_adj > 0)
        visited = np.zeros(self.n, dtype=np.uint8)
        stack = [root_node]
        visited[root_node] = 1
        while stack:
            node = stack.pop()
            neighbors = np.where(base_mask[node])[0]
            for nxt in neighbors:
                if visited[nxt] == 1:
                    continue
                visited[nxt] = 1
                stack.append(int(nxt))
        unreachable = [int(n) for n in active_nodes if visited[n] == 0]
        print("Held debug: no candidate found")
        print(f"active_count={active_ids.size} heap_nonempty={heap_nonempty} settled_root={settled_root}")
        print(f"root_node={root_node} unreachable_active_nodes_sample={unreachable[:10]}")

    def run(self):
        num_sources = len(self.sources)
        max_terms = num_sources * 2 + 2
        root_rep_id = max_terms - 1

        src_active = np.zeros(max_terms, dtype=np.uint8)
        src_node = np.full(max_terms, -1, dtype=np.int32)
        src_weight = np.zeros(max_terms, dtype=np.float32)

        dist_key = np.full((max_terms, self.n), np.inf, dtype=np.float32)
        prev = np.full((max_terms, self.n), -1, dtype=np.int32)
        settled = np.zeros((max_terms, self.n), dtype=np.uint8)

        heap_key = np.full((max_terms, self.n), np.inf, dtype=np.float32)
        heap_node = np.full((max_terms, self.n), -1, dtype=np.int32)
        heap_pos = np.full((max_terms, self.n), -1, dtype=np.int32)
        heap_size = np.zeros(max_terms, dtype=np.int64)
        version = np.zeros(max_terms, dtype=np.int64)
        settled_count = np.zeros(max_terms, dtype=np.int64)

        top_key = np.full((max_terms * 2,), np.inf, dtype=np.float32)
        top_src = np.full((max_terms * 2,), -1, dtype=np.int32)
        top_node = np.full((max_terms * 2,), -1, dtype=np.int32)
        top_ver = np.zeros((max_terms * 2,), dtype=np.int64)

        term_head = np.full(self.n, -1, dtype=np.int32)
        term_next = np.full(max_terms, -1, dtype=np.int32)
        best_val = np.full(max_terms, np.inf, dtype=np.float32)
        best_v_id_arr = np.full(max_terms, -1, dtype=np.int32)
        best_meet_node_arr = np.full(max_terms, -1, dtype=np.int32)

        next_id = 0
        for node in self.sources:
            src_active[next_id] = 1
            src_node[next_id] = node
            src_weight[next_id] = float(self.weights[node])
            self._init_ds(next_id, node, dist_key, prev, settled, heap_key, heap_node, heap_pos, heap_size)
            next_id += 1
            self.ds_created += 1

        src_node[root_rep_id] = self.root

        tree_edges: List[Tuple[int, int]] = []
        cons_edges_work = self.cons_edges.copy() if self.args.discount_components else self.cons_edges
        total_steps = max(num_sources, 1)
        progress = tqdm(total=total_steps, desc="Held merging", dynamic_ncols=True)

        while True:
            active_ids = np.where(src_active == 1)[0]
            if active_ids.size == 0:
                break

            total_weight = float(src_weight[active_ids].sum())
            self._build_term_lists(self.n, active_ids, root_rep_id, src_node, term_head, term_next)

            best_u, best_v, best_meet = held_round(
                self.n,
                self.indptr,
                self.indices,
                cons_edges_work,
                self.trans_edges,
                root_rep_id,
                src_active,
                src_node,
                src_weight,
                dist_key,
                prev,
                settled,
                heap_key,
                heap_node,
                heap_pos,
                heap_size,
                version,
                top_key,
                top_src,
                top_node,
                top_ver,
                0,
                best_val,
                best_v_id_arr,
                best_meet_node_arr,
                term_head,
                term_next,
                self.args.eta,
                self.d_bif_scaled,
                total_weight,
                settled_count,
            )

            if best_u < 0 or best_v < 0 or best_meet < 0:
                progress.close()
                self._debug_no_candidate(src_active, src_node, heap_size, settled, root_rep_id)
                raise RuntimeError("Held failed to find a merge candidate.")

            path_nodes = self._reconstruct_path(prev[best_u], src_node[best_u], best_meet)
            if not path_nodes:
                progress.close()
                raise RuntimeError("Held failed to reconstruct path.")
            selected_edges = set(tree_edges)
            if best_v != root_rep_id and self._path_exists_in_solution(path_nodes, selected_edges):
                path_edges = []
            else:
                path_edges = self._path_to_parent_edges(path_nodes)
            tree_edges.extend(path_edges)
            if self.args.discount_components:
                self._discount_path_edges(path_nodes, cons_edges_work)

            if best_v == root_rep_id:
                best_val[best_u] = np.inf
                best_v_id_arr[best_u] = -1
                best_meet_node_arr[best_u] = -1
                src_active[best_u] = 0
                active_ids = np.where(src_active == 1)[0]
                for sid in active_ids:
                    if best_v_id_arr[sid] == best_u:
                        best_val[sid] = np.inf
                        best_v_id_arr[sid] = -1
                        best_meet_node_arr[sid] = -1
                new_root = next_id
                next_id += 1
                if new_root >= max_terms - 1:
                    progress.close()
                    raise RuntimeError("Exceeded terminal capacity for root representatives.")
                src_node[new_root] = src_node[root_rep_id]
                src_weight[new_root] = src_weight[best_u]
                root_rep_id = new_root
            else:
                u_w = src_weight[best_u]
                v_w = src_weight[best_v]
                u_node = src_node[best_u]
                v_node = src_node[best_v]

                src_active[best_u] = 0
                src_active[best_v] = 0
                best_val[best_u] = np.inf
                best_val[best_v] = np.inf
                best_v_id_arr[best_u] = -1
                best_v_id_arr[best_v] = -1
                best_meet_node_arr[best_u] = -1
                best_meet_node_arr[best_v] = -1

                new_id = next_id
                next_id += 1
                if new_id >= max_terms - 1:
                    progress.close()
                    raise RuntimeError("Exceeded terminal capacity for Steiner nodes.")
                src_active[new_id] = 1
                src_weight[new_id] = u_w + v_w
                if self.args.better_steiner:
                    src_node[new_id] = self._choose_steiner_rep(path_nodes, u_w, v_w)
                elif best_meet in self._component_nodes(v_node, set(tree_edges)):
                    src_node[new_id] = best_meet
                elif self.rng.random() < (u_w / (u_w + v_w)):
                    src_node[new_id] = u_node
                else:
                    src_node[new_id] = v_node
                self._init_ds(new_id, src_node[new_id], dist_key, prev, settled, heap_key, heap_node, heap_pos, heap_size)
                self.ds_created += 1

                active_ids = np.where(src_active == 1)[0]
                for sid in active_ids:
                    if best_v_id_arr[sid] == best_u or best_v_id_arr[sid] == best_v:
                        best_val[sid] = np.inf
                        best_v_id_arr[sid] = -1
                        best_meet_node_arr[sid] = -1
                for sid in active_ids:
                    if sid == new_id:
                        continue
                    if settled[sid, src_node[new_id]] == 1:
                        b_uv = beta_fn(src_weight[sid], src_weight[new_id], self.args.eta, self.d_bif_scaled)
                        cand = dist_key[sid, src_node[new_id]] + b_uv
                        if cand < best_val[sid]:
                            best_val[sid] = cand
                            best_v_id_arr[sid] = new_id
                            best_meet_node_arr[sid] = src_node[new_id]

            progress.update(1)

        progress.total = progress.n
        progress.refresh()
        progress.close()

        self.tree_edges = tree_edges
        self.term_node = src_node[:next_id].tolist()
        self._verify(self.root, self.sources, tree_edges, self.ds_created, 2 * num_sources - 1,
                     settled_count[:next_id].tolist())
        return tree_edges, src_node[:next_id]

    def predict(self):
        if not self.sources:
            return torch.zeros((self.n, self.n), dtype=torch.float32, device=self.device)
        tree_edges, _ = self.run()
        pred_adj = self._build_parent_adj_from_edges(tree_edges)
        self._verify_parent_adj(pred_adj)
        return pred_adj
