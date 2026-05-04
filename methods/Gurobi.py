import torch

from utils.encode import get_mask

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - depends on local solver install
    gp = None
    GRB = None


class GurobiTree:
    def __init__(self, adj: torch.Tensor, vector: torch.Tensor, args):
        if gp is None:
            raise ImportError(
                "gurobipy is not installed. Install Gurobi and ensure the license is available "
                "before using the 'gurobi' method."
            )

        self.args = args
        self.device = args.device
        self.adj = adj.detach().cpu()
        self.vector = vector.detach().cpu()
        self.cons_adj = self.adj[0]
        self.trans_adj = self.adj[1]
        self.node_weights = self.vector[0]
        self.node_types = self.vector[1]
        self.center_node = (self.node_types == 0).nonzero().squeeze().item()
        self.mask = get_mask(adj, vector, args.mask_method, args).detach().cpu()
        self.n = self.cons_adj.shape[0]
        # Root demand is already at the sink and does not need to be routed
        # through a parent edge. Including it makes root-flow infeasible when
        # TN centers carry nonzero local demand.
        routed_weights = self.node_weights.clone()
        routed_weights[self.center_node] = 0.0
        self.total_weight = float(routed_weights.to(torch.float64).sum().item())
        self.solve_info = {}

    def _candidate_edges(self):
        rows, cols = torch.nonzero(self.mask, as_tuple=True)
        edges = []
        parent_in = [[] for _ in range(self.n)]
        child_out = [[] for _ in range(self.n)]
        for edge_idx, (parent, child) in enumerate(zip(rows.tolist(), cols.tolist())):
            if child == self.center_node:
                continue
            cons = float(self.cons_adj[parent, child].item())
            trans = float(self.trans_adj[parent, child].item())
            edges.append((parent, child, cons, trans))
            parent_in[child].append(edge_idx)
            child_out[parent].append(edge_idx)
        return edges, parent_in, child_out

    def predict(self):
        edges, parent_in, child_out = self._candidate_edges()
        if not edges:
            raise ValueError("No candidate edges available for Gurobi model.")

        model = gp.Model("codi_pg_tree")
        model.Params.OutputFlag = 1 if self.args.gurobi_verbose else 0
        model.Params.TimeLimit = float(self.args.gurobi_time_limit)
        model.Params.MIPGap = float(self.args.gurobi_mip_gap)
        model.Params.MIPFocus = int(self.args.gurobi_mip_focus)
        if self.args.gurobi_threads > 0:
            model.Params.Threads = int(self.args.gurobi_threads)

        x_vars = []
        f_vars = []
        for edge_idx, (parent, child, cons, trans) in enumerate(edges):
            x = model.addVar(vtype=GRB.BINARY, name=f"x_{parent}_{child}")
            f = model.addVar(lb=0.0, ub=self.total_weight, vtype=GRB.CONTINUOUS, name=f"f_{parent}_{child}")
            x_vars.append(x)
            f_vars.append(f)
            model.addConstr(f <= self.total_weight * x, name=f"cap_{parent}_{child}")

        model.update()

        for node in range(self.n):
            if node == self.center_node:
                continue
            incoming_x = gp.quicksum(x_vars[idx] for idx in parent_in[node])
            if float(self.node_weights[node].item()) > 0.0:
                model.addConstr(incoming_x == 1, name=f"parent_req_{node}")
            else:
                model.addConstr(incoming_x <= 1, name=f"parent_opt_{node}")

        for node in range(self.n):
            if node == self.center_node:
                continue
            up_flow = gp.quicksum(f_vars[idx] for idx in parent_in[node])
            down_flow = gp.quicksum(f_vars[idx] for idx in child_out[node])
            rhs = float(self.node_weights[node].item())
            model.addConstr(up_flow - down_flow == rhs, name=f"flow_{node}")

        root_child_flow = gp.quicksum(f_vars[idx] for idx in child_out[self.center_node])
        model.addConstr(root_child_flow == self.total_weight, name="root_flow")

        objective = gp.quicksum(
            cons * x_vars[idx] + trans * f_vars[idx]
            for idx, (_, _, cons, trans) in enumerate(edges)
        )
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()

        self.solve_info = {
            "gurobi_status": int(model.Status),
            "gurobi_status_name": str(model.Status),
            "gurobi_sol_count": int(model.SolCount),
            "gurobi_runtime": float(model.Runtime),
            "gurobi_mip_focus": int(self.args.gurobi_mip_focus),
        }
        if model.SolCount > 0:
            self.solve_info["gurobi_obj_val"] = float(model.ObjVal)
        if hasattr(model, "ObjBound"):
            self.solve_info["gurobi_obj_bound"] = float(model.ObjBound)
        if hasattr(model, "MIPGap"):
            self.solve_info["gurobi_mip_gap_actual"] = float(model.MIPGap)
        self.solve_info["gurobi_is_optimal"] = bool(model.Status == GRB.OPTIMAL)

        if model.SolCount == 0:
            raise RuntimeError(f"Gurobi failed to produce a solution. Status: {model.Status}")

        pred_adj = torch.zeros((self.n, self.n), dtype=torch.float32)
        for idx, (parent, child, _, _) in enumerate(edges):
            if x_vars[idx].X > 0.5:
                pred_adj[parent, child] = 1.0
        return pred_adj.to(self.device)
