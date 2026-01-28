from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils import DEFAULT_DTYPE, get_device, set_seed_torch


class AbstractGraphon:
    def __init__(
        self,
        theta: torch.Tensor,
        device: str = None,
        dtype: torch.dtype = DEFAULT_DTYPE,
    ):
        self.theta = theta
        self.device = device if device is not None else get_device()
        self.dtype = dtype

    def _sample_latents(self, n: int, seed: Optional[int] = None) -> torch.Tensor:
        set_seed_torch(seed)
        node_blocks = torch.randint(0, self.theta.shape[0], (n,), device=self.device)
        return node_blocks

    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
        include_self_loops: bool = False,
        sorted: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        set_seed_torch(seed)
        node_blocks = self._sample_latents(n, seed)
        if sorted:
            node_blocks, _ = torch.sort(node_blocks)
        theta = self.theta[node_blocks][:, node_blocks]

        if not include_self_loops:
            torch.diagonal(theta).zero_()

        r_idx, c_idx = torch.triu_indices(n, n, device=self.device)
        edges_flat = torch.bernoulli(theta[r_idx, c_idx])

        C = torch.zeros((n, n), device=self.device, dtype=self.dtype)
        C[r_idx, c_idx] = edges_flat
        C[c_idx, r_idx] = edges_flat

        return C, theta

    def to(self, device: str):
        self.device = device
        self.theta = self.theta.to(device=device)
        return self


class SBM(AbstractGraphon):
    def __init__(
        self,
        B: Union[np.ndarray, torch.Tensor],
        p: Union[np.ndarray, torch.Tensor] = None,
        device: str = None,
        dtype: torch.dtype = DEFAULT_DTYPE,
    ):
        if p is None:
            p = torch.ones(B.shape[0], device=device, dtype=dtype) / B.shape[0]
        # Input handling
        if isinstance(B, np.ndarray):
            B = torch.from_numpy(B)
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p)

        p /= p.sum()

        super().__init__(theta=B.to(device=device, dtype=dtype), device=device, dtype=dtype)
        self.p = p.to(device=device, dtype=dtype)

    def _sample_latents(self, n, seed=None):
        set_seed_torch(seed)
        return torch.multinomial(self.p, n, replacement=True)

    def to(self, device: str):
        self.device = device
        self.theta = self.theta.to(device=device)
        self.p = self.p.to(device=device)
        return self


class ER(SBM):
    def __init__(
        self,
        p: float,
        device: str = None,
        dtype: torch.dtype = DEFAULT_DTYPE,
    ):
        theta = torch.full((1, 1), p, device=device, dtype=dtype)
        p = torch.tensor([1.0], device=device, dtype=dtype)
        super().__init__(B=theta, p=p, device=device, dtype=dtype)

    def _sample_latents(self, n, seed=None):
        # dtype to mimick torch.randint behavior
        return torch.zeros(n, device=self.device, dtype=torch.int64)


class Graphon(AbstractGraphon):
    def __init__(
        self,
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        r: int = 1000,
        device: str = None,
        dtype: torch.dtype = DEFAULT_DTYPE,
    ):
        theta = self.synthesize_graphon(r=r, func=func, device=device, dtype=dtype)
        super().__init__(theta=theta, device=device, dtype=dtype)
        self.r = r
        self.func = func

    def synthesize_graphon(
        self,
        r: int = 1000,
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        device: str = "cpu",
        dtype: torch.dtype = DEFAULT_DTYPE,
    ) -> torch.Tensor:
        """
        Synthesize graphons

        Args:
            r: Resolution of discretized graphon (number of nodes/pixels) func: Function
            u,v -> [0,1] defining the graphon

        Returns:
            w: (r, r) torch.Tensor with elements in range [0, 1]
        """
        # Create grid [0, 1/r, 2/r, ..., 1.0]
        xs = (torch.arange(0, r, device=device, dtype=dtype)) / (r - 1)
        # Reshape to broadcast: u is (r, 1), v is (1, r)
        u = xs.view(-1, 1)
        v = xs.view(1, -1)
        return func(u, v)


def plot_sbm_heatmap(C, p, ax=None, cmap="Greys", vmin=0, vmax=1, colorbar=False, transform=lambda x: x):
    """
    Plots a heatmap of square matrix C where the size of row/col i is given by p[i].
    """
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()

    # Calculate the boundaries of the cells np.cumsum(p) gives the end coordinates, we
    # prepend 0 for the start.
    bounds = np.concatenate(([0], np.cumsum(p)))

    # Plot using pcolormesh bounds are used for both X and Y edges since C is square and p
    # applies to both.
    im = ax.pcolormesh(bounds, bounds, transform(C), cmap=cmap, shading="flat", vmin=vmin, vmax=vmax)

    # Invert Y-axis so that matrix row 0 is at the top (standard matrix visualization)
    ax.invert_yaxis()

    # Optional: ensure the plot is square (if p sums to 1, total area is 1x1)
    ax.set_aspect("equal")

    if colorbar:
        plt.colorbar(im, ax=ax)
    return ax, im


def plot_graphon(
    g: AbstractGraphon,
    ax=None,
    cmap="Greys",
    vmin=0,
    vmax=1,
    colorbar=False,
    transform=lambda x: x,
):
    """
    Plots the graphon heatmap.

    Args:
        g: Graphon instance ax: Matplotlib axis to plot on cmap: Colormap vmin: Minimum
        value for colormap vmax: Maximum value for colormap colorbar: Whether to display a
        colorbar

    Returns:
        ax: Matplotlib axis with the graphon plot
    """
    if ax is None:
        fig, ax = plt.subplots()
    if isinstance(g, Graphon):
        theta, p = g.theta, torch.ones(g.r, device=g.device, dtype=g.dtype) / g.r
    elif isinstance(g, SBM):
        theta, p = g.theta, g.p
    else:
        raise ValueError("Unsupported graphon type for plotting, expected Graphon or SBM.")

    ax, im = plot_sbm_heatmap(
        theta,
        p=p,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
        transform=transform,
    )
    return ax, im
