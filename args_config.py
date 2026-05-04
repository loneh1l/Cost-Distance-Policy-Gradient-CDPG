# args_config.py for main

import argparse
from random import choices


def add_value_solver_args(parser):
    parser.add_argument('--unreached_weight', type=float, default=10,
                        help='Weight for unreached term (float value)')
    parser.add_argument('--loss_iterations', type=int, default=100,
                        help='Number of iterations for loss calculation (default: 100)')
    parser.add_argument('--loss_iterations_mode', type=str, choices=['fixed', 'logn'], default='logn',
                        help='Loss iteration mode: fixed or logn (default: logn)')
    parser.add_argument('--loss_iterations_logn_mul', type=float, default=1.0,
                        help='Multiplier for logn loss iterations (default: 1.0)')
    parser.add_argument('--loss_solver', type=str, choices=['iter', 'inverse'], default='iter',
                        help='Loss solver: iter or inverse (default: iter)')
    parser.add_argument('--loss_early_stop', type=str_to_bool, default=False,
                        help='Early stop for loss iterations based on node_w change (default: False)')
    parser.add_argument('--loss_early_stop_tol', type=float, default=1e-9,
                        help='Relative tolerance for node_w early stop (default: 1e-9)')
    parser.add_argument('--eval_loss_iterations', type=int, default=40000,
                        help='Eval loss iterations (default: 40000)')
    parser.add_argument('--eval_loss_early_stop', type=str_to_bool, default=True,
                        help='Eval loss early stop based on node_w change (default: False)')
    parser.add_argument('--eval_loss_early_stop_tol', type=float, default=1e-9,
                        help='Eval loss early stop tolerance (default: 1e-9)')
    parser.add_argument('--unreached_type', type=str, choices=['flow', 'rq', 'notears'], default='flow',
                        help='Type of unreached term (default: flow)')

def add_main_args(parser):
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu',
                        help='Device to use (cpu or gpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument("--mask_method", type=str, choices=['acyclic', 'edge'], default='edge',
                        help="Mask method : acyclic or edge")
    parser.add_argument("--dataset", type=str, choices=['rand', 'dir', 'tn', 'gba'], default='gba',
                        help="Dataset to use")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory of data (default: None)")
    parser.add_argument("--tn_name", type=str, default='Anaheim',
                        help="TransportationNetworks dataset name (default: Anaheim)")
    parser.add_argument("--tn_root", type=str, default='./data/TransportationNetworks',
                        help="TransportationNetworks root dir (default: ./data/TransportationNetworks)")
    parser.add_argument("--tn_mode", type=str, choices=['all', 'single'], default='all',
                        help="TN run mode: all origins or a single origin by --subgraph (default: all)")
    parser.add_argument("--tn_graph_mode", type=str, choices=['road_only', 'raw'], default='road_only',
                        help="TN graph representation: projected road-only graph or raw centroid graph (default: road_only)")
    parser.add_argument("--gba_root", type=str, default='./data/GBA-CDNet',
                        help="GBA-CDNet root dir (default: ./data/GBA-CDNet)")
    parser.add_argument("--rand_root", type=str, default='./data/rand',
                        help="Rand graph root dir (default: ./data/rand)")
    parser.add_argument("--subgraph", type=int, default=0,
                        help="the id of Subgraph (default: 0)")
    parser.add_argument("--save_pred_adj", type=str_to_bool, default=False,
                        help="Save adj matrix for prediction (default: False)")
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory to save logs (default: None)')
    parser.add_argument('--dij_type', type=str, choices=['trans', 'weighted'],default='trans',
                        help='Type of dijkstra (default: trans)')
    parser.add_argument('--source_num_rate', type=float, default=1, help='Source rate (default: 1)')
    parser.add_argument('--source_num_rate_log', type=str_to_bool, default=False,
                        help='To add logarithm to source rate (default: False)')
    parser.add_argument('--dynamic_mask', type=str_to_bool, default=False,
                        help='Recompute acyclic mask from pred_adj each step (default: False)')
    parser.add_argument('--dynamic_mask_interval', type=int, default=1,
                        help='Steps between dynamic mask updates (default: 1)')
    parser.add_argument('--dynamic_mask_loss_check', type=str_to_bool, default=True,
                        help='Only update dynamic mask when the candidate mask has lower loss (default: True)')
    parser.add_argument('--dynamic_mask_loss_check_steps', type=int, default=1,
                        help='Compare dynamic mask candidate loss against recorded training loss N steps back: 0=current step, 1=previous step (default: 1)')
    parser.add_argument('--trans_scale', type=float, default=1.0,
                        help='Scale factor for transport costs (default: 1.0)')

def parse_args():
    parser = argparse.ArgumentParser(description="main")
    subparsers = parser.add_subparsers(dest='method', required=True, help="chose method")

    # parsers for CDPG
    parser_cdpg = subparsers.add_parser('cdpg', help='cdpg')
    add_value_solver_args(parser_cdpg)
    add_main_args(parser_cdpg)
    parser_cdpg.add_argument("--lora_rank", type=int, default=6,
                                help="LoRA rank")
    parser_cdpg.add_argument("--lr", type=float, default=0.1,
                               help="Learning rate")
    parser_cdpg.add_argument("--optimizer", type=str,
                               choices=['adagrad', 'adam', 'nadam', 'rmsprop'],
                               default='rmsprop',
                               help="Optimizer for CDPG training (default: rmsprop)")
    parser_cdpg.add_argument('--train_iterations', type=int, default=300,
                               help="Training iterations (int, default: 300)")
    parser_cdpg.add_argument('--alpha', type=float, default=1.0,
                               help='Alpha parameter (float value, default: 1.0)')
    parser_cdpg.add_argument('--use_lora', type=str_to_bool, default=True,
                               help='Enable LoRA (default: True)')
    parser_cdpg.add_argument('--use_unreached', type=str_to_bool, default=True,
                               help='Enable unreached penalty (default: True)')
    parser_cdpg.add_argument('--initial_point_type', type=str, choices=['random', 'trans', 'cons'], default='random',
                               help='Initial point type')
    parser_cdpg.add_argument('--fixed_train_iters', type=str_to_bool, default=False,
                               help='Use fixed train iterations instead of early stop (default: False)')
    parser_cdpg.add_argument('--train_early_stop_tol', type=float, default=1e-4,
                               help='Train early stop threshold as fraction of loss (default: 1e-4)')
    parser_cdpg.add_argument(
        '--predict_mode',
        type=str,
        choices=['argmax', 'topo_greedy'],
        default='topo_greedy',
        help='Prediction hardening mode for CDPG (default: argmax)',
    )

    # parsers for DAC25
    parser_dac25 = subparsers.add_parser('dac25', help='dac25')
    add_value_solver_args(parser_dac25)
    add_main_args(parser_dac25)
    parser_dac25.add_argument("--chose_node", type=str, default='min_max',
                              help="Chose node to use (default: min_max)")

    # parsers for Held
    parser_held = subparsers.add_parser('held', help='Held & Perner cost-distance')
    add_value_solver_args(parser_held)
    add_main_args(parser_held)
    parser_held.add_argument("--d_bif", type=float, default=0.0,
                             help="Bifurcation delay multiplier on mean transport (default: 0.0)")
    parser_held.add_argument("--eta", type=float, default=0.0,
                             help="Bifurcation split lower bound eta (default: 0.0)")
    parser_held.add_argument("--discount_components", type=str_to_bool, default=False,
                             help="Discount existing tree components by zeroing cons cost (default: False)")
    parser_held.add_argument("--better_steiner", type=str_to_bool, default=False,
                             help="Pick Steiner representative on path by weighted delay (default: False)")

    # parsers for Chekuri
    parser_chekuri = subparsers.add_parser('chekuri', help='Chekuri-style deterministic aggregation')
    add_value_solver_args(parser_chekuri)
    add_main_args(parser_chekuri)
    parser_chekuri.add_argument("--chekuri_candidate_k", type=int, default=None,
                                help=argparse.SUPPRESS)
    parser_chekuri.add_argument("--chekuri_neighbor_pool", type=int, default=None,
                                help=argparse.SUPPRESS)
    parser_chekuri.add_argument("--chekuri_gamma_scale", type=float, default=1.0,
                                help="Scale on deterministic root-estimator potential (default: 1.0)")

    # parsers for Meyerson
    parser_meyerson = subparsers.add_parser('meyerson', help='Meyerson randomized aggregation')
    add_value_solver_args(parser_meyerson)
    add_main_args(parser_meyerson)

    # parsers for GA
    parser_ga = subparsers.add_parser('ga', help='Genetic Algorithm')
    add_value_solver_args(parser_ga)
    add_main_args(parser_ga)
    parser_ga.add_argument("--population_size", type=int, default=50,
                           help="Population size (default: 50)")
    parser_ga.add_argument("--generations", type=int, default=50,
                           help="Number of generations (default: 50)")
    parser_ga.add_argument("--mutation_rate", type=float, default=0.1,
                           help="Mutation rate (default: 0.1)")
    parser_ga.add_argument("--crossover_rate", type=float, default=0.8,
                           help="Crossover rate (default: 0.8)")

    # parsers for SA
    parser_sa = subparsers.add_parser('sa', help='SA')
    add_value_solver_args(parser_sa)
    add_main_args(parser_sa)
    parser_sa.add_argument("--iterations", type=int, default=100,
                           help="Number of iterations (default: 100)")
    parser_sa.add_argument("--initial_temp", type=float, default=1.0,
                           help="Initial temperature (default: 1.0)")
    parser_sa.add_argument("--temp_decay", type=float, default=0.99,
                           help="Temperature decay rate (default: 0.99)")

    # parsers for Gurobi
    parser_gurobi = subparsers.add_parser('gurobi', help='Gurobi MILP baseline')
    add_value_solver_args(parser_gurobi)
    add_main_args(parser_gurobi)
    parser_gurobi.add_argument("--gurobi_time_limit", type=float, default=600.0,
                               help="Gurobi time limit in seconds (default: 600)")
    parser_gurobi.add_argument("--gurobi_mip_gap", type=float, default=1e-4,
                               help="Relative MIP gap tolerance (default: 1e-4)")
    parser_gurobi.add_argument("--gurobi_threads", type=int, default=0,
                               help="Gurobi thread count, 0 lets Gurobi decide (default: 0)")
    parser_gurobi.add_argument("--gurobi_mip_focus", type=int, default=1, choices=[0, 1, 2, 3],
                               help="Gurobi MIPFocus parameter (default: 1)")
    parser_gurobi.add_argument("--gurobi_verbose", type=str_to_bool, default=False,
                               help="Enable Gurobi solver log output (default: False)")

    # parsers for RLSAC
    parser_rlsac = subparsers.add_parser('rlsac', help='RLSAC')
    add_value_solver_args(parser_rlsac)
    add_main_args(parser_rlsac)
    parser_rlsac.add_argument('--train_iterations', nargs=3, type=int, default=[150, 150, 150],
                               help="Three int: x y z")
    parser_rlsac.add_argument("--lr", type=float, default=0.1,
                               help="Learning rate")
    parser_rlsac.add_argument('--initial_point_type', type=str, choices=['random', 'trans', 'cons'], default='random',
                               help='Initial point type')
    parser_rlsac.add_argument("--epsilon", type=float, default=1.0,
                              help="Epsilon (default: 1.0)")
    parser_rlsac.add_argument("--des", type=float, default=0.002,
                              help="Decay rate (default: 0.002)")
    parser_rlsac.add_argument("--batch_size", type=int, default=32,
                              help="Batch size (default: 32)")

    # prsers for GS
    parser_gs = subparsers.add_parser('gs', help='Gumbel Softmax')
    add_value_solver_args(parser_gs)
    add_main_args(parser_gs)
    parser_gs.add_argument('--train_iterations', nargs=3, type=int, default=[150, 150, 150],
                              help="Three int: x y z")
    parser_gs.add_argument("--lr", type=float, default=0.1,
                               help="Learning rate")
    parser_gs.add_argument('--initial_point_type', type=str, choices=['random', 'trans', 'cons'], default='random',
                              help='Initial point type')

    # Parse CLI args and return
    return parser.parse_args()

def str_to_bool(value):
    """Convert 'True'/'False' strings to booleans."""
    if value.lower() in ('true', 't', '1'):
        return True
    elif value.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
