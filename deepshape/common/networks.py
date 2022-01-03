from abc import ABC, abstractmethod
from torch import no_grad, eye, ones_like
from torch.nn import Module, ModuleList
from .derivatives import jacobian
from .LayerBase import DeepShapeLayer


class ShapeReparamBase(Module, ABC):
    def __init__(self, layerlist):
        super().__init__()
        self.layerlist = ModuleList(layerlist)
        for layer in layerlist:
            assert isinstance(
                layer, DeepShapeLayer), "Layers must inherit DeepShapeLayer"

    def forward(self, x):
        for layer in self.layerlist:
            x = layer(x)
        return x

    @abstractmethod
    def derivative(self, x, h=1e-4):
        pass

    def project(self, **kwargs):
        with no_grad():
            for module in self.modules():
                if isinstance(module, DeepShapeLayer):
                    module.project(**kwargs)

    def to(self, device):
        super().to(device)
        for module in self.modules():
            if isinstance(module, DeepShapeLayer):
                module.to(device)

        return self


class CurveReparametrizer(ShapeReparamBase):
    def derivative(self, x, h=1e-4):
        dc = ones_like(x)
        for layer in self.layerlist:
            dc = dc * layer.derivative(x, h)
            x = layer(x)
        return dc


class SurfaceReparametrizer(ShapeReparamBase):
    def derivative(self, x, h=1e-4):
        Df = eye(2, 2, device=x.device)
        for layer in self.layerlist:
            Df = layer.derivative(x, h) @ Df
            x = layer(x)
        return Df

