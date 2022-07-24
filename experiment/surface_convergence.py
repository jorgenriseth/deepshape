import argparse
from cProfile import label
import matplotlib.pyplot as plt
from deepshape.surfaces import *
from utils_common import *
from surface_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reparametrize and Plot surfaces")
    parser.add_argument("--fig_path", default="../figures")
    parser.add_argument("--shape0", default=None)
    parser.add_argument("--shape1", default="HyperbolicParaboloid")
    parser.add_argument("--diffeomorphism",
                        default="RotationDiffeomorphism", nargs="*")
    parser.add_argument("--p", default=1, type=int,
                    help="Projection Argument, p-lipshitz")
    parser.add_argument("--transform", default="qmap")
    parser.add_argument("--verbosity", default=-1, type=int)
    parser.add_argument("--num_layers", default=0, type=int)
    parser.add_argument("--num_funcs", default=0, type=int)
    parser.add_argument("--k", default=32, type=int)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    fig_path, f0, f1, diffeo, transform, num_layers, num_funcs, projection_kwargs, logger = reparametrization_parser("surface-convergence", args)


    num_layers_list = list(range(1, 15))
    num_functions_list = list(range(1, 15))
    subset = [1, 3, 5, 7, 10, 15]

    # Testset
    # num_layers_list = list(range(1, 4))
    # num_functions_list = list(range(1, 4))
    # subset = [1, 2, 3]

    d = create_convergence_dict(f0, f1, num_layers_list, num_functions_list, transform=transform, projection_kwargs=projection_kwargs, logger=logger)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plot_depth_convergence(d, ax, subset=subset, label_identifier="N")
    ax.legend()
    ax.set_xlabel("$L$", fontsize=18)
    plt.legend(loc=3)
    plt.savefig(fig_path / "surface-depth.pdf", bbox_inches="tight")


    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plot_width_convergence(d, ax, subset=subset)
    ax.legend()
    ax.set_xlabel("$N$", fontsize=18)
    plt.legend(loc=3)
    plt.savefig(fig_path / "surface-width.pdf", bbox_inches="tight")
    if args.show:
        plt.show()

