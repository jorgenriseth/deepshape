import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from deepshape.curves import *

def curve_reparametrization(c1, c2, num_layers, num_functions, transform="qmap", k=256, projection_kwargs=None, **kwargs):
    if projection_kwargs is None:
        projection_kwargs = {}

    if transform.lower() == "qmap":
        q, r = Qmap(c1), Qmap(c2)
    elif transform.lower() == "srvt":
        q, r = SRVT(c1), SRVT(c2)
    else:
        raise ValueError("Transform should be 'qmap' or 'srvt'")

    RN = CurveReparametrizer(
        [SineSeries(num_functions) for _ in range(num_layers)]
    )
    optimizer = torch.optim.LBFGS(RN.parameters(), max_iter=200,
                                  line_search_fn="strong_wolfe")
    loss = CurveDistance(q, r, k=k)
    error = reparametrize(RN, loss, optimizer, 1, **kwargs)
    return error, RN


def create_convergence_dict(c0, c1, num_layer_list, num_function_list, parser=None, *args, **kwargs):
    if parser is None:
        def parser(x): return x[0][-1]

    return {
        i: {
            j: parser(curve_reparametrization(c0, c1, i, j, **kwargs)) for j in num_function_list
        }
        for i in num_layer_list
    }


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

def plot_width_convergence(d, ax=None, subset=None, log=True):
    E = width_convergence(d)
    N = list(depth_convergence(d))

    if ax is None:
        fig, ax = plt.subplots()

    for num_layers, error in E.items():
        if num_layers in subset or subset is None: 
            if log:
                ax.semilogy(N, error, label=f"{num_layers} layers")
            else:
                ax.plot(N, error, label=f"{num_layers} layers")
    ax.set_xticks(N)
    return ax


def parse_diffeomorphism(args):
    if type(args.diffeomorphism) is not list:
        diff_list = [args.diffeomorphism]
    else:
        diff_list = args.diffeomorphism
    composition = f"{diff_list[-1]}()"
    for i in range(len(diff_list)-2, -1, -1):
        composition = f"{diff_list[i]}().compose({composition})"
    return eval(composition)


def reparametrization_parser(args):
    fig_path = Path(args.fig_path)
    diffeo = parse_diffeomorphism(args)
    transform = args.transform
    projection_kwargs = {'p': args.p}

    c1 = eval(f"{ args.curve1}()")
    if args.curve0 is None:
        c0 = c1.compose(diffeo)
    else:
        c0 = eval(f"{args.curve0}()")

    verbosity = args.verbosity
    if verbosity == -1:
        logger = Silent()
    else:
        logger = Logger(verbosity)


    num_layers = args.num_layers
    num_funcs = args.num_funcs
    return fig_path, c0, c1, diffeo, transform, num_layers, num_funcs, projection_kwargs, logger
