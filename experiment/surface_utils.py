from pathlib import Path
import matplotlib.pyplot as plt
from deepshape.surfaces import *


def surface_reparametrization(c1, c2, num_layers, num_functions, transform="qmap", k=32, projection_kwargs=None, **kwargs):
    if projection_kwargs is None:
        projection_kwargs = {}

    if transform.lower() == "qmap":
        q, r = Qmap(c1), Qmap(c2)
    elif transform.lower() == "SRNF":
        q, r = SRNF(c1), SRNF(c2)
    else:
        raise ValueError("Transform should be 'qmap' or 'srvt'")

    RN = SurfaceReparametrizer(
        [SineLayer(num_functions) for _ in range(num_layers)]
    )
    optimizer = torch.optim.LBFGS(RN.parameters(), max_iter=200,
                                  line_search_fn="strong_wolfe")
    loss = SurfaceDistance(q, r, k=k)
    error = reparametrize(RN, loss, optimizer, 1, **kwargs)
    return error, RN

def create_convergence_dict(c0, c1, num_layer_list, num_function_list, parser=None, *args, **kwargs):
    if parser is None:
        def parser(x): return x[0][-1]

    return {
        i: {
            j: parser(surface_reparametrization(c0, c1, i, j, **kwargs)) for j in num_function_list
        }
        for i in num_layer_list
    }


def reparametrization_parser(args):
    fig_path = Path(args.fig_path)
    diffeo = parse_diffeomorphism(args)
    transform = args.transform
    projection_kwargs = {'p': args.p}

    c1 = eval(f"{ args.surface1}()")
    if args.surface0 is None:
        c0 = c1.compose(diffeo)
    else:
        c0 = eval(f"{args.surface0}()")

    verbosity = args.verbosity
    if verbosity == -1:
        logger = Silent()
    else:
        logger = Logger(verbosity)

    num_layers = args.num_layers
    num_funcs = args.num_funcs
    return fig_path, c0, c1, diffeo, transform, num_layers, num_funcs, projection_kwargs, logger

def parse_diffeomorphism(args):
    if type(args.diffeomorphism) is not list:
        diff_list = [args.diffeomorphism]
    else:
        diff_list = args.diffeomorphism
    composition = f"{diff_list[-1]}()"
    for i in range(len(diff_list)-2, -1, -1):
        composition = f"{diff_list[i]}().compose({composition})"
    return eval(composition)