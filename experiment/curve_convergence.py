import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from deepshape.curves import *
from curve_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reparametrize and Plot Curves")
    parser.add_argument("--fig_path", default="../figures")
    parser.add_argument("--curve0", default=None)
    parser.add_argument("--curve1", default="Circle")
    parser.add_argument("--diffeomorphism", default="LogStepDiff", nargs="*")
    parser.add_argument("--p", default=1, type=int,
                        help="Projection Argument, p-lipshitz")
    parser.add_argument("--transform", default="qmap")
    parser.add_argument("--verbosity", default=-1)
    parser.add_argument("--num_layers", default=5, type=int)
    parser.add_argument("--num_funcs", default=5, type=int)
    parser.add_argument("--k", default=256, type=int)
    args = parser.parse_args()
    fig_path, c0, c1, diffeo, transform, num_layers, num_funcs, projection_kwargs, logger = reparametrization_parser(args)


    num_layers_list = list(range(1, 16))
    num_functions_list = list(range(1, 16))
    subset = [1, 3, 5, 7, 10, 15]
    d = create_convergence_dict(c0, c1, num_layers_list, num_functions_list, transform=transform, projection_kwargs=projection_kwargs, logger=logger)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plot_depth_convergence(d, ax, subset=subset)
    ax.legend()
    ax.set_xlabel("# Layers")
    plt.savefig(f"{fig_path}/depth_convergence-{args.transform}-{args.curve0}-{args.curve1}-{args.diffeomorphism}-{args.num_layers}-{args.num_funcs}-{args.p}.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plot_width_convergence(d, ax, subset=subset)
    ax.legend()
    ax.set_xlabel("# Functions Per Layer")
    plt.savefig(f"{fig_path}/width_convergence-{args.transform}-{args.curve0}-{args.curve1}-{args.diffeomorphism}-{args.num_layers}-{args.num_funcs}-{args.p}.png", bbox_inches="tight")
    plt.show()
