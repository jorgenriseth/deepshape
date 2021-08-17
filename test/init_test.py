import torch
import numpy as np
import deepshape.curves.funcapprox as fa

from torch import cos, sin
from numpy import pi


def test_fourier_approx():
    # Define a test function
    def f(x):
        return 2. + 3. * sin(2.0*pi*x) + 4. * cos(10* 2.0*pi*x)

    # Create a linearly spaced sample of points on [0, 1]
    Xtr = torch.linspace(0, 1, 201).unsqueeze(-1)

    # Evaluate f at these points, and find the fourier 
    # approximation
    y = f(Xtr)
    F = fa.FourierApprox(Xtr, y, 10)

    # Check that the values are equal on random sampples of points
    # on [0, 1]
    Xtest = torch.rand(51, 1)
    print(F(Xtest) - f(Xtest))
    assert torch.allclose(F(Xtest), f(Xtest), atol=1e-5)

    # Check correctness of coefficients
    coeff = F.w[[0, 1, 20]]
    true_coeff = torch.tensor([2., 3., 4.]).unsqueeze(-1)
    print(coeff - true_coeff)
    assert torch.allclose(coeff, true_coeff, atol=1e-5)


def test_fourier_basis_eval():
    Xtr = torch.Tensor([
        [0.],
        [0.5],
        [1.0]
    ])

    y = sin(2.*pi*Xtr)
    F = fa.FourierApprox(Xtr, y, 1)
    Btrue = torch.tensor([
        [1., 0., 1.],
        [1., 0., -1.],
        [1., 0., 1.]
    ])
    print(F.eval_basis(Xtr) - Btrue)
    assert torch.allclose(Btrue, F.eval_basis(Xtr), atol=1e-6)
