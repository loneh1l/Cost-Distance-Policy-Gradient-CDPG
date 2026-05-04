# main.py
import os
import math
import torch
import random
try:
    import numpy as np
except ImportError:
    np = None
from datetime import datetime
from args_config import parse_args
from methods.CDPG import CDPG
from methods.Held import Held
from methods.Chekuri import Chekuri
from methods.Meyerson import Meyerson
from methods.Gurobi import GurobiTree
from utils.encode import get_mask
from utils.loss import ValueSolver
from utils.data_loader import DataLoader, GbaCDNetLoader
from utils.exp_tracker import MainTracker

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if np is not None:
        np.random.seed(seed)

def drop_sources(vector):

    old_rand_state = random.getstate()
    random.seed(35)

    # Get all source indices (tensor[1] == 1)
    sources_mask = vector[1] == 1
    source_indices = torch.nonzero(sources_mask).squeeze(-1)  # Shape: [k]
    k = len(source_indices)

    rate = args.source_num_rate
    if k == 0 or (rate == 1 and not args.source_num_rate_log):
        return vector  # No sources, return directly

    n = vector[0].size()[0]
    # Compute number of sources to keep
    if args.source_num_rate_log:
        keep_num = int(rate * math.log2(n))
    else:
        keep_num = int(rate * k)

    # Randomly choose sources to keep
    kept_indices = random.sample(source_indices.tolist(), keep_num)
    kept_mask = torch.zeros_like(vector[1], dtype=torch.bool)
    kept_mask[kept_indices] = True
    kept_mask[vector[1] == 0] = True

    # Update weight and type
    vector[0] = torch.where(kept_mask, vector[0], torch.tensor(0))
    vector[1] = torch.where(kept_mask, vector[1], torch.tensor(2))

    random.setstate(old_rand_state)

    return vector

def test_one_pic(adj, vector):
    """
    adj(2 x num_nodes x num_nodes)
    {
        cons_adj & trans_adj
        cons_adj: construction cost of each edge
        trans_adj: transport cost of each edge
    }
    vector(2 x num_nodes)
    {
        weights & types
        weights: the weight of each node
        types: the type of each node (0:sink, 1:source, 2:steiner point)
    }
    """
    global main_tracker
    orig_adj = adj.clone()
    orig_vector = vector.clone()
    vector = drop_sources(vector.clone())
    ones_count = torch.sum(vector[1].eq(1)).item()
    n = adj.shape[1]
    if args.loss_iterations_mode == "logn":
        args.loss_iterations = max(1, int(args.loss_iterations_logn_mul * math.log2(n)))
    base_mask = (adj[0] > 0) & (adj[1] > 0)
    edge_count = int(torch.triu(base_mask).sum().item())
    print("number of nodes {}".format(n))
    print("number of edges {}".format(edge_count))
    print("number of sources {}".format(ones_count))
    print("generate mask...")
    mask = get_mask(adj, vector, args.mask_method, args)
    print("done.")

    if args.method == "cdpg":
        planner = CDPG(adj, vector, mask, args)
        main_tracker.time_begin("train_time")
        planner.train()
        main_tracker.time_end("train_time")
    elif args.method == "held":
        planner = Held(adj, vector, args)
    elif args.method == "chekuri":
        planner = Chekuri(adj, vector, args)
    elif args.method == "meyerson":
        planner = Meyerson(adj, vector, args)
    elif args.method == "gurobi":
        planner = GurobiTree(adj, vector, args)
    else:
        print("NO THAT METHOD")
        exit(0)

    main_tracker.time_begin("predict_time")
    with torch.no_grad():
        pred_adj = planner.predict()
        if not isinstance(pred_adj, torch.Tensor):
            pred_adj = torch.tensor(pred_adj).to(args.device)
    if args.save_pred_adj:
        save_adj = pred_adj.detach().cpu()
        torch.save(save_adj,f"{args.log_dir}/pred_adj.pt")
    main_tracker.time_end("predict_time")

    criterion = ValueSolver(orig_adj, orig_vector)
    trans_loss, cons_loss, unreached, rls = criterion.test_loss(pred_adj, args)
    loss = trans_loss + cons_loss + unreached
    print('_________________________________________________________________________________')
    loss_dict = {
        'trans_loss': trans_loss.item(),
        'cons_loss': cons_loss.item(),
        'unreached': unreached.item(),
        'total_loss': loss.item()
    }
    print(loss_dict)
    main_tracker.print_time()
    main_tracker.track(loss_dict)
    if hasattr(planner, "solve_info") and planner.solve_info:
        main_tracker.track(planner.solve_info)
    main_tracker.save(f"{args.log_dir}/main_info.json")
    torch.cuda.empty_cache()
    return loss_dict


def _apply_trans_scale(adj: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return adj
    if adj.dim() == 3:
        adj[1].mul_(scale)
        return adj
    if adj.dim() == 4:
        adj[:, 1].mul_(scale)
        return adj
    raise ValueError(f"Unexpected adj shape for trans scaling: {adj.shape}")


def _json_safe_args(args):
    safe = {}
    for key, value in vars(args).items():
        if isinstance(value, torch.device):
            safe[key] = str(value)
        else:
            safe[key] = value
    return safe

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = f"./logs/{timestamp}"
    os.makedirs(args.log_dir, exist_ok=True)
    if args.device == 'gpu':
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        args.device = torch.device('cpu')
    if args.dataset == 'gba':
        gba_loader = GbaCDNetLoader(args.gba_root, args.subgraph, args.device)
        adj, vector = gba_loader.load()
    else:
        data_path = args.data_dir
        dataloader = DataLoader(data_path, args.device)
        adj, vector = dataloader.load()

    adj = _apply_trans_scale(adj, args.trans_scale)
    main_tracker = MainTracker()
    main_tracker.track(_json_safe_args(args))

    if args.dataset in ('dir',):
        test_id = args.subgraph # Subgraph index; -1 loops through all (not recommended)
        if test_id == -1: # Known bug, avoid using
            for i, j in zip(adj, vector):
                test_one_pic(i, j)
        else:
            test_one_pic(adj[test_id], vector[test_id])
    elif args.dataset in ('gba',):
        test_one_pic(adj, vector)
