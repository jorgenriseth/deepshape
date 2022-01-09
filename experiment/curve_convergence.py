import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from deepshape.curves import *
from utils_common import *
from curve_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reparametrize and Plot Curves")
    parser.add_argument("--fig_path", default="../figures")
    parser.add_argument("--shape0", default=None)
    parser.add_argument("--shape1", default="Circle")
    parser.add_argument("--diffeomorphism", default="LogStepDiff", nargs="*")
    parser.add_argument("--p", default=1, type=int,
                        help="Projection Argument, p-lipshitz")
    parser.add_argument("--transform", default="qmap")
    parser.add_argument("--verbosity", default=-1)
    parser.add_argument("--num_layers", default=0, type=int)
    parser.add_argument("--num_funcs", default=0, type=int)
    parser.add_argument("--k", default=256, type=int)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    fig_path, c0, c1, diffeo, transform, num_layers, num_funcs, projection_kwargs, logger = reparametrization_parser("curve-convergence", args)


    num_layers_list = list(range(1, 16))
    num_functions_list = list(range(1, 16))
    subset = [1, 3, 5, 7, 10, 15]
    d = create_convergence_dict(c0, c1, num_layers_list, num_functions_list, transform=transform, projection_kwargs=projection_kwargs, logger=logger)

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    plot_depth_convergence(d, ax, subset=subset)
    ax.legend()
    ax.set_xlabel("# Layers")
    plt.savefig(fig_path / "convergence-depth.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))

    plot_width_convergence(d, ax, subset=subset)
    ax.legend()
    ax.set_xlabel("# Functions Per Layer")
    plt.savefig(fig_path / "convergence-width.pdf", bbox_inches="tight")

    if args.show:
        plt.show()

