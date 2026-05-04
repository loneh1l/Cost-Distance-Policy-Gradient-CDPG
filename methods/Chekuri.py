from typing import Dict, List, Tuple

import numpy as np
import torch

from methods.Meyerson import (
    Meyerson,
    build_candidates_exact,
    select_matching_sorted,
    shortest_path_nodes,
)


class Chekuri(Meyerson):
    """
    Exact pair-metric layered aggregation with deterministic center selection.

    This uses the same all-pairs, pair-dependent shortest-path candidate
    generation as Meyerson. The only algorithmic difference is the center
    selection for a matched non-root pair: Meyerson samples proportionally to
    weight, while this class chooses the endpoint minimizing an explicit
    cost-distance pessimistic score.

    The SODA note derandomizes using LP distances and omits implementation
    details. The estimator here is an implementable substitute: shortest
    root-connection cost under c(e) + W*l(e), where W is the merged weight.
    It keeps the same asymptotic shortest-path workload as Meyerson, plus one
    lower-order root-estimator shortest path per matched pair.
    """

    def __init__(self, adj: torch.Tensor, vector: torch.Tensor, args):
        super().__init__(adj, vector, args)
        self.gamma_scale = float(getattr(args, "chekuri_gamma_scale", 1.0))
        self.estimator_paths = 0
        self._root_estimator_cache: Dict[Tuple[int, float], float] = {}

    def _path_stats(self, path_nodes: np.ndarray) -> Tuple[float, float]:
        cons = 0.0
        trans = 0.0
        for a, b in zip(path_nodes[:-1], path_nodes[1:]):
            cons += float(self.cons_adj[int(a), int(b)])
            trans += float(self.trans_adj[int(a), int(b)])
        return cons, trans

    def _root_cost_distance_estimate(self, node: int, weight: float) -> float:
        key = (int(node), float(np.float32(weight)))
        cached = self._root_estimator_cache.get(key)
        if cached is not None:
            return cached

        path = shortest_path_nodes(
            self.root,
            int(node),
            np.float32(weight),
            self.indptr,
            self.indices,
            self.cons_edges,
            self.trans_edges,
            self.n,
        )
        if path.size == 0:
            value = float("inf")
        else:
            cons, trans = self._path_stats(path)
            value = cons + float(weight) * trans

        self.estimator_paths += 1
        self._root_estimator_cache[key] = value
        return value

    def _future_scale(self, active_count: int) -> float:
        if active_count <= 1:
            return 0.0
        return self.gamma_scale

    def _pair_score(
        self,
        center: int,
        carried_weight: float,
        merged_weight: float,
        path_nodes: np.ndarray,
        active_count: int,
    ) -> float:
        cons, trans = self._path_stats(path_nodes)
        future_scale = self._future_scale(active_count)
        return (
            cons
            + carried_weight * trans
            + future_scale * self._root_cost_distance_estimate(center, merged_weight)
        )

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
        total = w_u + w_v

        score_u = float("inf")
        score_v = float("inf")
        if path_uv.size > 0:
            score_u = self._pair_score(u, w_v, total, path_uv, active_count)
        if path_vu.size > 0:
            score_v = self._pair_score(v, w_u, total, path_vu, active_count)

        if score_u < score_v:
            return int(u), path_uv
        if score_v < score_u:
            return int(v), path_vu
        if np.isfinite(score_u):
            return (int(u), path_uv) if int(u) < int(v) else (int(v), path_vu)
        raise RuntimeError(
            f"Chekuri could not recover either directed pair path between {u} and {v}."
        )

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

        active_count = int(active_nodes.shape[0])
        for u, v in zip(selected_u.tolist(), selected_v.tolist()):
            if v == self.root:
                path = self._recover_path(self.root, u, weight_map[u])
                self._add_path_segments(path)
                continue

            w_u = weight_map[u]
            w_v = weight_map[v]
            center, path = self._choose_pair_center_and_path(
                u, v, w_u, w_v, active_count
            )
            self._add_path_segments(path)

            total = w_u + w_v
            next_nodes.append(center)
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
                f"Chekuri stage {stage_count + 1}: active terminals={active_nodes.size}"
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
                    "Chekuri could not find any feasible matching candidate."
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
                raise RuntimeError("Chekuri failed to select a matching in this round.")

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
            "chekuri_stages": stage_count,
            "chekuri_pair_merges": pair_merges,
            "chekuri_root_merges": root_merges,
            "chekuri_union_segments": len(self.used_segments),
            "chekuri_candidate_paths": total_candidates,
            "chekuri_estimator_paths": self.estimator_paths,
            "chekuri_gamma_scale": self.gamma_scale,
            "chekuri_impl": "exact_pair_metric_deterministic_center_directed_parent_orientation",
            "chekuri_center_estimator": "root_to_center_shortest_path_c_plus_weight_l",
        }
