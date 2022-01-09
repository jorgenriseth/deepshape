from argparse import ArgumentParser
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Union, Tuple
from typing_extensions import TypeAlias
from torch.optim import LBFGS
from deepshape.common.networks import CurveReparametrizer, ShapeReparamBase
from deepshape.curves.curves import Curve
from deepshape.common.Loss import ShapeDistanceBase

from deepshape.surfaces import SurfaceReparametrizer, SineSeries, Surface
from deepshape.common import Loss, reparametrize
import deepshape


class BaseProblem(ABC):
    @abstractmethod
    def transform(self, c):
        pass

    @abstractmethod
    def loss(self, **kwargs):
        pass

    def reparametrize(self):
        pass


Shape = Union(Curve, Surface)

Transform = Union(
    deepshape.curves.Qmap,
    deepshape.curves.SRVT,
    deepshape.surfaces.Qmap,
    deepshape.surfaces.SRNF
)


@dataclass
class BaseProblem:
    shape0: Shape
    shape1: Shape
    network: ShapeReparamBase
    loss: ShapeDistanceBase
    transform: Transform

    def solve(self, **kwargs):
        q, r = self.transform(self.shape0), self.transform(self.shape1)
        optimizer = LBFGS(self.network.parameters(), max__iter=200,
                          line_search_fn="strong_wolfe")
        error = reparametrize(self.network, self.loss, optimizer, 1, **kwargs)
        return error

