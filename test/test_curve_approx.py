import torch
import numpy as np
import deepshape.curves.funcapprox as fa
from deepshape.curves.utils import *

from torch import cos, sin
from numpy import pi
import pytest
import matplotlib.pyplot as plt


def test_fourier_approx():
    # Define a test function
    def f(x):
        return 2. + 3. * sin(2.0*pi*x) + 4. * cos(10* 2.0*pi*x)

    # Create a linearly spaced sample of points on [0, 1]
    Xtr = torch.linspace(0, 1, 401).unsqueeze(-1)

    # Evaluate f at these points, and find the fourier 
    # approximation
    y = f(Xtr)
    F = fa.FourierApprox(Xtr, y, 10)

    # Check that the values are equal on random sampples of points
    # on [0, 1]
    Xtest = torch.rand(101, 1)
    print(F(Xtest) - f(Xtest))
    assert torch.allclose(F(Xtest), f(Xtest), atol=1e-5)

    # Check correctness of coefficients
    coeff = F.w[[0, 1, 20]]
    true_coeff = torch.tensor([2., 3., 4.]).unsqueeze(-1)
    print(coeff - true_coeff)
    assert torch.allclose(coeff, true_coeff, atol=1e-5)


def test_fourier_basis_eval():
    # Create known test points
    Xtr = torch.Tensor([
        [0.],
        [0.5],
        [1.0]
    ])

    
    # Define function to evaluate in the given points
    y = sin(2.*pi*Xtr)
    F = fa.FourierApprox(Xtr, y, 1)
    Btrue = torch.tensor([
        [1., 0., 1.],
        [1., 0., -1.],
        [1., 0., 1.]
    ])
    print(F.eval_basis(Xtr) - Btrue)
    assert torch.allclose(Btrue, F.eval_basis(Xtr), atol=1e-6)


@pytest.mark.curves
@pytest.mark.plotting
def test_curve_data_approc():
    # Load testpoints
    x, Y = get_function_data("test/001.txt", (0, 1))
    xs = col_linspace(0, 1, 401)
    c1 = fa.FourierApprox(x, Y, 3)
    c2 = fa.FourierApprox(x, Y, 5)
    c3 = fa.FourierApprox(x, Y, 50)

    plt.figure()
    plt.plot(*Y.T, 'k--')
    plt.plot(*c1(xs).T, 'r')
    plt.plot(*c2(xs).T, 'b')
    plt.plot(*c3(xs).T, 'g')
    plt.legend()
    plt.show(block=False)
    plt.pause(2.0)
