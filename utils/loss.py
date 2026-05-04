import copy
import torch
import torch.nn.functional as F
import math
from utils.encode import dijkstra_distances

class ValueSolver:
    def __init__(self, adj:torch.Tensor, vector:torch.Tensor, graph=None):
        self.cons_adj = adj[0]
        self.trans_adj = adj[1]
        self.node_weights = vector[0]
        self.center_node = (vector[1] == 0).nonzero().squeeze().item()
        self.dij_dis = dijkstra_distances(
            adj,
            center_node=self.center_node,
            dij_type="trans",
            vector=vector,
            directed=True,
        )

    def _flow_unreached(self, node_w: torch.Tensor) -> torch.Tensor:
        dist = self.dij_dis
        finite = torch.isfinite(dist)
        if finite.any():
            max_dist = dist[finite].max()
            dist_safe = torch.where(finite, dist, max_dist + 1.0)
        else:
            dist_safe = torch.ones_like(dist)
        return torch.matmul(node_w, dist_safe).sum()

    def loss(self, pred_adj, args):
        iterations = args.loss_iterations
        unreached_weight = args.unreached_weight
        p_trans = self.trans_adj.clone()
        p_cons = self.cons_adj.clone()
        node_w = self.node_weights.clone()
        device = args.device
        n = pred_adj.size()[0]

        nw_trans = torch.mul(pred_adj, p_trans)
        # cons_term = pred_adj + pred_adj.T - torch.mul(pred_adj, pred_adj.T)
        # nw_cons = p_cons * cons_term / 2
        nw_cons = torch.mul(pred_adj, p_cons)
        trans_w = torch.matmul(torch.ones(n, device=device), nw_trans)
        min_path = 0
        prev_node_w = None
        if getattr(args, "loss_solver", "iter") == "inverse":
            eye = torch.eye(n, device=device)
            mat = eye - pred_adj
            mat = mat + 1e-6 * eye
            node_w = torch.linalg.solve(mat, node_w)
            min_path = torch.dot(trans_w, node_w)
        else:
            for i in range(iterations):
                min_path += torch.dot(trans_w, node_w)
                prev_node_w = node_w
                node_w = torch.matmul(pred_adj, node_w)
                if args.loss_early_stop:
                    denom = torch.norm(prev_node_w, p=2).item()
                    if denom == 0:
                        denom = 1e-12
                    diff = torch.norm(node_w - prev_node_w, p=2).item()
                    if diff < denom * args.loss_early_stop_tol:
                        break
        v = node_w
        u = F.normalize(v, p=2, dim=0)
        rls = u @ (pred_adj @ u) / ((u @ u)+1e-12)
        if args.loss_unreached:
            if args.unreached_type == 'flow':
                unreached = self._flow_unreached(node_w)
            elif args.unreached_type == 'rq':
                unreached = (n * torch.exp(rls) - n)**2
            elif args.unreached_type == 'notears':
                unreached = (torch.trace(torch.matrix_exp(pred_adj*pred_adj))-n)**2
            else:
                print("No such unreached type")
                exit(0)
            return min_path, nw_cons.sum(), unreached*unreached_weight, rls
        else:
            return min_path, nw_cons.sum(), rls

    def loss_with_aw(self, pred_adj, args):
        """
        Compute loss and return the summed node weights across iterations
        for dynamic mask construction.
        """
        iterations = args.loss_iterations
        unreached_weight = args.unreached_weight
        p_trans = self.trans_adj.clone()
        p_cons = self.cons_adj.clone()
        node_w = self.node_weights.clone()
        device = args.device
        n = pred_adj.size()[0]

        nw_trans = torch.mul(pred_adj, p_trans)
        nw_cons = torch.mul(pred_adj, p_cons)
        trans_w = torch.matmul(torch.ones(n, device=device), nw_trans)
        min_path = 0
        node_w_sum = node_w.clone()
        prev_node_w = None
        if getattr(args, "loss_solver", "iter") == "inverse":
            eye = torch.eye(n, device=device)
            mat = eye - pred_adj
            mat = mat + 1e-6 * eye
            node_w = torch.linalg.solve(mat, node_w)
            node_w_sum = node_w
            min_path = torch.dot(trans_w, node_w)
        else:
            for i in range(iterations):
                min_path += torch.dot(trans_w, node_w)
                prev_node_w = node_w
                node_w = torch.matmul(pred_adj, node_w)
                node_w_sum = node_w_sum + node_w
                if args.loss_early_stop:
                    denom = torch.norm(prev_node_w, p=2).item()
                    if denom == 0:
                        denom = 1e-12
                    diff = torch.norm(node_w - prev_node_w, p=2).item()
                    if diff < denom * args.loss_early_stop_tol:
                        break
        v = node_w
        u = F.normalize(v, p=2, dim=0)
        rls = u @ (pred_adj @ u) / ((u @ u)+1e-12)
        if args.loss_unreached:
            if args.unreached_type == 'flow':
                unreached = self._flow_unreached(node_w)
            elif args.unreached_type == 'rq':
                unreached = (n * torch.exp(rls) - n)**2
            elif args.unreached_type == 'notears':
                unreached = (torch.trace(torch.matrix_exp(pred_adj*pred_adj))-n)**2
            else:
                print("No such unreached type")
                exit(0)
            return min_path, nw_cons.sum(), unreached*unreached_weight, rls, node_w_sum
        else:
            return min_path, nw_cons.sum(), rls, node_w_sum

    def test_loss(self, pred_adj, args):
        args_eval = copy.copy(args)
        args_eval.loss_unreached = True
        args_eval.loss_iterations = args.eval_loss_iterations
        args_eval.loss_early_stop = args.eval_loss_early_stop
        args_eval.loss_early_stop_tol = args.eval_loss_early_stop_tol
        args_eval.loss_solver = "iter"
        loss_trans, loss_cons, unreached, rls = self.loss(pred_adj, args_eval)
        return loss_trans, loss_cons, unreached*args_eval.unreached_weight, rls
