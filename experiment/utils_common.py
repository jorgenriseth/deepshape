from pathlib import Path
import matplotlib.pyplot as plt
# from deepshape.surfaces import *
# from deepshape.curves.curves import *
import deepshape
from deepshape.curves.curves import *
from deepshape.surfaces.surfaces import *

def parse_diffeomorphism(args):
    if type(args.diffeomorphism) is not list:
        diff_list = [args.diffeomorphism]
    else:
        diff_list = args.diffeomorphism
    composition = f"{diff_list[-1]}()"
    for i in range(len(diff_list)-2, -1, -1):
        composition = f"{diff_list[i]}().compose({composition})"
    return eval(composition)


def create_savename(prefix, args):
    if type(args.diffeomorphism) is not list:
        diff_list = [args.diffeomorphism]
    else:
        diff_list = args.diffeomorphism
    
    if len(diff_list) > 1:
        diffstring = '_'.join(args.diffeomorphism)
    else:
        diffstring = args.diffeomorphism
    figpath = (f"{args.fig_path}/{prefix}-{args.transform}-{args.shape0}-"
               f"{args.shape1}-{diffstring}-{args.num_layers}-"
               f"{args.num_funcs}-{args.p}")
    figpath = Path(figpath)
    figpath.mkdir(exist_ok=True)
    return figpath


def reparametrization_parser(prefix, args):
    fig_path = create_savename(prefix, args)
    diffeo = parse_diffeomorphism(args)
    transform = args.transform
    projection_kwargs = {'p': args.p}

    c1 = eval(f"{ args.shape1}()")
    if args.shape0 is None:
        c0 = c1.compose(diffeo)
    else:
        c0 = eval(f"{args.shape0}()")

    verbosity = args.verbosity
    if verbosity == -1:
        logger = deepshape.common.Silent()
    else:
        logger = deepshape.common.Logger(verbosity)

    num_layers = args.num_layers
    num_funcs = args.num_funcs
    return fig_path, c0, c1, diffeo, transform, num_layers, num_funcs, projection_kwargs, logger


def depth_convergence(d):
    return {j: [d[i][j] for i in d] for j in list(d.values())[0]}


def width_convergence(d):
    return {i: [d[i][j] for j in list(d.values())[0]] for i in d}


def create_convergence_matrix(d):
    Eij = np.zeros((len(d), len(list(d.values())[0])))
    for i, num_layers in enumerate(d):
        for j, num_funcs in enumerate(d[num_layers]):
            Eij[i, j] = d[num_layers][num_funcs]
    return Eij


def plot_depth_convergence(d, ax=None, subset=None, log=True):
    E = depth_convergence(d)
    N = list(width_convergence(d))

    if ax is None:
        fig, ax = plt.subplots()

    for num_funcs, error in E.items():
        if num_funcs in subset or subset is None: 
            if log:
                ax.semilogy(N, error, label=f"{num_funcs} functions")
            else:
                ax.plot(N, error, label=f"{num_funcs} functions")
    ax.set_xticks(N)
    return ax

def plot_width_convergence(d, ax=None, subset=None, log=True, **kwargs):
    E = width_convergence(d)
    N = list(depth_convergence(d))

    if ax is None:
        fig, ax = plt.subplots()

    for num_layers, error in E.items():
        if num_layers in subset or subset is None: 
            if log:
                ax.semilogy(N, error, label=f"{num_layers} layers", **kwargs)
            else:
                ax.plot(N, error, label=f"{num_layers} layers", **kwargs)
    ax.set_xticks(N)
    return ax
