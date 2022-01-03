import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from deepshape.curves import *
from curve_utils import curve_reparametrization, reparametrization_parser


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
    parser.add_argument("--verbosity", default=-1, type=int)
    parser.add_argument("--num_layers", default=5, type=int)
    parser.add_argument("--num_funcs", default=5, type=int)
    parser.add_argument("--k", default=256, type=int)
    parser.add_argument("--show", action='store_true')
    args = parser.parse_args()

    fig_path, c0, c1, diffeo, transform, num_layers, num_funcs, projection_kwargs, logger = reparametrization_parser(args)

    error, RN = curve_reparametrization(
        c0, c1, num_layers, num_funcs, transform, args.k, logger=logger)

    # Plot curves and their transforms
    fig, ax = plt.subplots(1, 1)
    plot_curve(c0, npoints=501, dotpoints=41, ax=ax)
    ax.set_aspect("equal")
    plt.savefig(fig_path / "curve0.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1)
    plot_curve(c1, npoints=501, dotpoints=41, ax=ax)
    ax.set_aspect("equal")
    plt.savefig(fig_path / "curve1.png", bbox_inches="tight")

    plt.figure(figsize=(12, 4))
    plt.semilogy(error)
    plt.savefig(fig_path / "error-iter-convergence.png", bbox_inches="tight")

    # Plot Diffeomorphism with derivative
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plot_diffeomorphism(RN, ax=ax)
    plot_diffeomorphism(diffeo, ax=ax, c="k", lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.savefig(fig_path / "diffeomorphism-matching.png", bbox_inches="tight") 

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plot_diffeomorphism(RN.derivative, ax=ax) 
    plot_derivative(diffeo, ax=ax, c="k", lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    plt.savefig(fig_path / "derivatives-matching.png", bbox_inches="tight")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.3, 5.6))
    plot_diffeomorphism(lambda x: c0(x)[:, 0], ls="dashed", ax=ax1)
    plot_diffeomorphism(lambda x: c1(x)[:, 0], ax=ax1)
    plot_diffeomorphism(lambda x: c0(x)[:, 1], ls="dashed", ax=ax2)
    plot_diffeomorphism(lambda x: c1(x)[:, 1], ax=ax2)
    plt.savefig(fig_path / "coordinates-before.png", bbox_inches="tight")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.3, 5.6))
    plot_diffeomorphism(lambda x: c0(x)[:, 0], ls="dashed", ax=ax1)
    plot_diffeomorphism(lambda x: c1(RN(x))[:, 0], ax=ax1)
    plot_diffeomorphism(lambda x: c0(x)[:, 1], ls="dashed", ax=ax2)
    plot_diffeomorphism(lambda x: c1(RN(x))[:, 1], ax=ax2)
    plt.savefig(fig_path / "coordinates-after.png", bbox_inches="tight")

    if args.show:
        plt.show()

