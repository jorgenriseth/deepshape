import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from deepshape.surfaces import *
from surface_utils import reparametrization_parser, surface_reparametrization


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
    args = parser.parse_args()

    fig_path, f0, f1, diffeo, transform, num_layers, num_funcs, projection_kwargs, logger = reparametrization_parser(
        args)

    error, RN = surface_reparametrization(
        f0, f1, num_layers, num_funcs, transform, args.k, logger=logger)

    f_reparam = f1.compose(RN)
    colornorm = get_common_colornorm([f0, f1, f_reparam])

    # Plot surfaces and their transforms
    view_angle = (30, 225)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(131, projection="3d")
    plot_surface(f0, ax=ax, k=128, colornorm=colornorm, camera=view_angle)
    ax = fig.add_subplot(132, projection="3d")
    plot_surface(f1, ax=ax, k=128, colornorm=colornorm, camera=view_angle)
    ax = fig.add_subplot(133, projection="3d")
    plot_surface(f_reparam, ax=ax, k=128, colornorm=colornorm, camera=view_angle)
    plt.savefig(f"{fig_path}/surfaces-{args.transform}-{args.surface0}-{args.surface1}-{args.diffeomorphism}-{args.num_layers}-{args.num_funcs}-{args.p}.png", bbox_inches="tight")

    plt.figure(figsize=(12, 4))
    plt.semilogy(error)
    plt.savefig(f"{fig_path}/error-convergence-{args.transform}-{args.surface0}-{args.surface1}-{args.diffeomorphism}-{args.num_layers}-{args.num_funcs}-{args.p}.png", bbox_inches="tight")

    # Plot Diffeomorphism with derivative
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plot_diffeomorphism(diffeo, ax=ax[0], color='k', k=20)
    ax[0].set_title("True")
    plot_diffeomorphism(RN, ax=ax[1], color='k', k=20)
    ax[1].set_title("Found")
    plt.savefig(f"{fig_path}/diffeomorphism-{args.transform}-{args.surface0}-{args.surface1}-{args.diffeomorphism}-{args.num_layers}-{args.num_funcs}-{args.p}.png", bbox_inches="tight")
    plt.show()
