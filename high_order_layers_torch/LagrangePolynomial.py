import math
import torch
from torch import Tensor
import torch.nn as nn

from .Basis import *


def chebyshevLobatto(n: int):
    """
    Compute the Chebyshev-Lobatto points which are in the range [-1.0, 1.0].

    Args:
        n (int): Number of points.

    Returns:
        Tensor: A tensor of length n with x locations from negative to positive including -1 and 1 [-1,...,+1].
    """
    if n == 1:
        return torch.tensor([0.0])

    return -torch.cos(torch.pi * torch.arange(n, dtype=torch.float32) / (n - 1))


class FourierBasis(nn.Module):
    def __init__(self, length: float):
        """
        Fourier basis functions [sin, cos].

        Args:
            length (float): The length of the basis function. A value of 1 means there is periodicity 1.
        """
        super(FourierBasis, self).__init__()
        self.length = length
        self.num_basis = None  # Should be set appropriately elsewhere.

    def forward(self, x: Tensor, j: int):
        """
        Compute the value at x for the given component of the Fourier basis function.

        Args:
            x (Tensor): The point(s) of interest (any shape).
            j (int): Index of the basis function.

        Returns:
            Tensor: Evaluated basis function at x.
        """
        if j == 0:
            return 0.5 + 0.0 * x

        i = (j + 1) // 2
        if j % 2 == 0:
            ans = torch.cos(2.0 * math.pi * i * x / self.length)
        else:
            ans = torch.sin(2.0 * math.pi * i * x / self.length)
        return ans

    def __call__(self, x: Tensor, j: int):
        return self.forward(x, j)


class LagrangeBasis(nn.Module):
    def __init__(self, n: int, length: float = 2.0):
        """
        Lagrange basis functions using Chebyshev-Lobatto points.

        Args:
            n (int): Number of basis functions.
            length (float, optional): Length of the domain. Defaults to 2.0.
        """
        super(LagrangeBasis, self).__init__()
        self.n = n
        self.register_buffer('X', (length / 2.0) * chebyshevLobatto(n))
        self.register_buffer('denominators', self._compute_denominators())
        self.num_basis = n

    def _compute_denominators(self):
        denom = torch.ones((self.n, self.n), dtype=torch.float32)
        for j in range(self.n):
            for m in range(self.n):
                if m != j:
                    denom[j, m] = self.X[j] - self.X[m]
        return denom

    def forward(self, x: Tensor, j: int):
        """
        Compute the value at x for the given component of the Lagrange basis function.

        Args:
            x (Tensor): The point(s) of interest (any shape).
            j (int): Index of the basis function.

        Returns:
            Tensor: Evaluated basis function at x.
        """
        # Ensure that buffers are on the same device as x
        X = self.X.to(x.device)
        denominators = self.denominators.to(x.device)

        x_diff = x.unsqueeze(-1) - X  # Ensure broadcasting
        arange_n = torch.arange(self.n, device=x.device)

        one_tensor = torch.tensor(1.0, device=x.device)
        b = torch.where(
            arange_n != j, x_diff / denominators[j], one_tensor
        )
        ans = torch.prod(b, dim=-1)
        return ans

    def __call__(self, x: Tensor, j: int):
        return self.forward(x, j)


class LagrangeBasisND(nn.Module):
    def __init__(self, n: int, length: float = 2.0, dimensions: int = 2):
        """
        Multidimensional Lagrange basis functions.

        Args:
            n (int): Number of basis functions per dimension.
            length (float, optional): Length of the domain. Defaults to 2.0.
            dimensions (int, optional): Number of dimensions. Defaults to 2.
        """
        super(LagrangeBasisND, self).__init__()
        self.n = n
        self.dimensions = dimensions
        self.register_buffer('X', (length / 2.0) * chebyshevLobatto(n))
        self.register_buffer('denominators', self._compute_denominators())
        self.num_basis = int(math.pow(n, dimensions))

    def _compute_denominators(self):
        denom = torch.ones((self.n, self.n), dtype=torch.float32)
        for j in range(self.n):
            for m in range(self.n):
                if m != j:
                    denom[j, m] = self.X[j] - self.X[m]
        return denom

    def forward(self, x: Tensor, index: list):
        """
        Evaluate the multidimensional Lagrange basis function at x.

        Args:
            x (Tensor): Input tensor of shape [batch, inputs, dimensions].
            index (list): List of indices for each dimension [dimensions].

        Returns:
            Tensor: Evaluated basis function at x, shape [batch, inputs].
        """
        # Ensure that buffers are on the same device as x
        X = self.X.to(x.device)
        denominators = self.denominators.to(x.device)

        x_diff = x.unsqueeze(-1) - X  # [batch, inputs, dimensions, basis]

        r = 1.0
        for i, basis_i in enumerate(index):
            arange_n = torch.arange(self.n, device=x.device)
            one_tensor = torch.tensor(1.0, device=x.device)

            b = torch.where(
                arange_n != basis_i,
                x_diff[:, :, i, :] / denominators[basis_i],
                one_tensor,
            )
            r *= torch.prod(b, dim=-1)

        return r

    def __call__(self, x: Tensor, index: list):
        return self.forward(x, index)


class LagrangeBasis1(nn.Module):
    """
    Degenerate case for n=1.
    """

    def __init__(self, length: float = 2.0):
        super(LagrangeBasis1, self).__init__()
        self.n = 1
        self.register_buffer('X', torch.tensor([0.0]))
        self.num_basis = 1

    def forward(self, x: Tensor, j: int):
        return torch.ones_like(x)

    def __call__(self, x: Tensor, j: int):
        return self.forward(x, j)


def get_lagrange_basis(n: int, length: float = 2.0):
    if n == 1:
        return LagrangeBasis1(length=length)
    else:
        return LagrangeBasis(n, length=length)


class LagrangeExpand(BasisExpand):
    def __init__(self, n: int, length: float = 2.0):
        super().__init__(get_lagrange_basis(n, length), n)


class PiecewisePolynomialExpand(PiecewiseExpand):
    def __init__(self, n: int, segments: int, length: float = 2.0):
        super().__init__(
            basis=get_lagrange_basis(n, length), n=n, segments=segments, length=length
        )


class PiecewisePolynomialExpand1d(PiecewiseExpand1d):
    def __init__(self, n: int, segments: int, length: float = 2.0):
        super().__init__(
            basis=get_lagrange_basis(n, length), n=n, segments=segments, length=length
        )


class PiecewiseDiscontinuousPolynomialExpand(PiecewiseDiscontinuousExpand):
    def __init__(self, n: int, segments: int, length: float = 2.0):
        super().__init__(
            basis=get_lagrange_basis(n, length), n=n, segments=segments, length=length
        )


class PiecewiseDiscontinuousPolynomialExpand1d(PiecewiseDiscontinuousExpand1d):
    def __init__(self, n: int, segments: int, length: float = 2.0):
        super().__init__(
            basis=get_lagrange_basis(n, length), n=n, segments=segments, length=length
        )


class FourierExpand(BasisExpand):
    def __init__(self, n: int, length: float):
        super().__init__(FourierBasis(length=length), n)


class LagrangePolyFlat(BasisFlat):
    def __init__(self, n: int, length: float = 2.0, **kwargs):
        super().__init__(n, get_lagrange_basis(n, length), **kwargs)


class LagrangePolyFlatND(BasisFlatND):
    def __init__(self, n: int, length: float = 2.0, dimensions: int = 2, **kwargs):
        super().__init__(
            n,
            LagrangeBasisND(n, length, dimensions=dimensions),
            dimensions=dimensions,
            **kwargs
        )


class LagrangePolyFlatProd(BasisFlatProd):
    def __init__(self, n: int, length: float = 2.0, **kwargs):
        super().__init__(n, get_lagrange_basis(n, length), **kwargs)


class LagrangePoly(Basis):
    def __init__(self, n: int, length: float = 2.0, **kwargs):
        super().__init__(n, get_lagrange_basis(n, length=length), **kwargs)


class LagrangePolyProd(BasisProd):
    def __init__(self, n: int, length: float = 2.0, **kwargs):
        super().__init__(n, get_lagrange_basis(n, length), **kwargs)


class FourierSeriesFlat(BasisFlat):
    def __init__(self, n: int, length: float = 1.0, **kwargs):
        super().__init__(n, FourierBasis(length), **kwargs)
