import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from deepshape.surfaces import *
from utils_common import reparametrization_parser
from surface_utils import surface_reparametrization


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
    parser.add_argument("--num_layers", default=5, type=int)
    parser.add_argument("--num_funcs", default=5, type=int)
    parser.add_argument("--k", default=32, type=int)
    parser.add_argument("--show", action='store_true')
    args = parser.parse_args()

    fig_path, f0, f1, diffeo, transform, num_layers, num_funcs, projection_kwargs, logger = reparametrization_parser("surfaces", args)

    error, RN = surface_reparametrization(
        f0, f1, num_layers, num_funcs, transform, args.k, logger=logger)

    f_reparam = f1.compose(RN)
    colornorm = get_common_colornorm([f0, f1, f_reparam])

    # Plot surfaces and their transforms
    view_angle = (30, 225)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_surface(f0, ax=ax, k=128, colornorm=colornorm, camera=view_angle)
    plt.savefig(fig_path / "surfaces0.png", bbox_inches="tight")


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_surface(f1, ax=ax, k=128, colornorm=colornorm, camera=view_angle)
    plt.savefig(fig_path / "surfaces1.png", bbox_inches="tight")


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_surface(f_reparam, ax=ax, k=128, colornorm=colornorm, camera=view_angle)
    plt.savefig(fig_path / "surfaces-reparam.png", bbox_inches="tight")

    plt.figure(figsize=(12, 4))
    plt.semilogy(error)
    plt.savefig(fig_path / "error-iter-convergence.png", bbox_inches="tight")

    # Plot Diffeomorphism with derivative
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plot_diffeomorphism(diffeo, ax=ax, color='k', k=20)
    plt.savefig(fig_path / "diffeomorphism-true.png", bbox_inches="tight")


    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plot_diffeomorphism(RN, ax=ax, color='k', k=20)
    plt.savefig(fig_path / "diffeomorphism-found.png", bbox_inches="tight")
    
    if args.show: 
        plt.show()

