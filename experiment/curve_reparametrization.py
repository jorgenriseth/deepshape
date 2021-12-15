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
    args = parser.parse_args()

    fig_path, c0, c1, diffeo, transform, num_layers, num_funcs, projection_kwargs, logger = reparametrization_parser(args)

    error, RN = curve_reparametrization(
        c0, c1, num_layers, num_funcs, transform, args.k, logger=logger)

    # Plot curves and their transforms
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    plot_curve(c0, dotpoints=21, ax=axes[0])
    plot_curve(c1, dotpoints=21, ax=axes[1])
    for ax in axes:
        ax.set_aspect("equal")
    plt.savefig(f"{fig_path}/curves-{args.transform}-{args.curve0}-{args.curve1}-{args.diffeomorphism}-{args.num_layers}-{args.num_funcs}-{args.p}.png", bbox_inches="tight")


    plt.figure(figsize=(12, 4))
    plt.semilogy(error)
    plt.savefig(f"{fig_path}/error-convergence-{args.transform}-{args.curve0}-{args.curve1}-{args.diffeomorphism}-{args.num_layers}-{args.num_funcs}-{args.p}.png", bbox_inches="tight")


    # Plot Diffeomorphism with derivative
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plot_diffeomorphism(RN, ax=ax)
    plot_diffeomorphism(diffeo, ax=ax, c="k", lw=0.8, ls="--")
    plt.savefig(f"{fig_path}/diffeomorphism-{args.transform}-{args.curve0}-{args.curve1}-{args.diffeomorphism}-{args.num_layers}-{args.num_funcs}-{args.p}.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plot_diffeomorphism(RN.derivative, ax=ax)
    plot_derivative(diffeo, ax=ax, c="k", lw=0.8, ls="--")
    plt.savefig(f"{fig_path}/derivative-{args.transform}-{args.curve0}-{args.curve1}-{args.diffeomorphism}-{args.num_layers}-{args.num_funcs}-{args.p}.png", bbox_inches="tight")
    plt.show()
