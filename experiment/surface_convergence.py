import argparse
import matplotlib.pyplot as plt
from deepshape.surfaces import *
from surface_utils import *
from curve_utils import plot_depth_convergence, plot_width_convergence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reparametrize and Plot surfaces")
    parser.add_argument("--fig_path", default="../figures/surfaces")
    parser.add_argument("--surface0", default=None)
    parser.add_argument("--surface1", default="HyperbolicParaboloid")
    parser.add_argument("--diffeomorphism",
                        default="RotationDiffeomorphism", nargs="*")
    parser.add_argument("--p", default=1, type=int,
                    help="Projection Argument, p-lipshitz")
    parser.add_argument("--transform", default="qmap")
    parser.add_argument("--verbosity", default=-1, type=int)
    parser.add_argument("--num_layers", default=5, type=int)
    parser.add_argument("--num_funcs", default=5, type=int)
    parser.add_argument("--k", default=32, type=int)
    parser.add_argument("--blocking", default=False, type=bool)
    args = parser.parse_args()

    fig_path, f0, f1, diffeo, transform, num_layers, num_funcs, projection_kwargs, logger = reparametrization_parser(args)


    num_layers_list = list(range(1, 12))
    num_functions_list = list(range(1, 12))
    subset = [1, 3, 5, 7, 10]    
    d = create_convergence_dict(f0, f1, num_layers_list, num_functions_list, transform=transform, projection_kwargs=projection_kwargs, logger=logger)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plot_depth_convergence(d, ax, subset=subset)
    ax.legend()
    ax.set_xlabel("# Layers")
    plt.savefig(f"{fig_path}/depth_convergence-{args.transform}-{args.surface0}-{args.surface1}-{args.diffeomorphism}-{args.num_layers}-{args.num_funcs}-{args.p}.png", bbox_inches="tight")


    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plot_width_convergence(d, ax, subset=subset)
    ax.legend()
    ax.set_xlabel("# Functions Per Layer")
    plt.savefig(f"{fig_path}/width_convergence-{args.transform}-{args.surface0}-{args.surface1}-{args.diffeomorphism}-{args.num_layers}-{args.num_funcs}-{args.p}.png", bbox_inches="tight")
    plt.show()

    