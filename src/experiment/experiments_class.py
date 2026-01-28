import math
from typing import List, Optional, Tuple

import torch

from ..method.graphons import SBM, Graphon
from ..method.utils import set_seed_torch


class Experiment:
    """Base class for running graphon estimation experiments.

    This class encapsulates the configuration for sampling graphs from a graphon and
    provides methods for generating experimental data.

    Attributes
    ----------
    name : str
        Human-readable name for the experiment.
    graphon : Graphon
        The graphon object used to sample graphs.
    num_graphs : int
        Number of graphs to sample per experiment iteration.
    name_save : str
        Name used for saving results (defaults to `name` if not provided).
    """

    def __init__(
        self,
        name: str,
        graphon: Graphon,
        num_graphs: int = 1,
        name_save: str = None,
        **kwargs,
    ):
        """Initialize an Experiment.

        Parameters
        ----------
        name : str
            Human-readable name for the experiment.
        graphon : Graphon
            The graphon object used to sample graphs.
        num_graphs : int, optional
            Number of graphs to sample per iteration, by default 1.
        name_save : str, optional
            Name used for saving results. If None, defaults to `name`.
        """
        self.name = name
        self.graphon = graphon
        self.num_graphs = num_graphs
        if name_save is None:
            self.name_save = name

    def get_k(self, n: int) -> int:
        """Compute the maximum number of clusters for the barycenter.

        Parameters
        ----------
        n : int
            Number of nodes in the graph.

        Returns
        -------
        int
            Maximum number of clusters, computed as floor(sqrt(n)).
        """
        return math.floor(math.sqrt(n))

    def get_size_graphs(self, n: int) -> List[int]:
        """Get the sizes of all graphs to sample.

        Parameters
        ----------
        n : int
            Number of nodes for each graph.

        Returns
        -------
        List[int]
            List of graph sizes, each equal to n.
        """
        return [n] * self.num_graphs

    def get_lambdas(self, As: List[torch.Tensor]) -> List[float]:
        """Compute barycenter weights proportional to graph sizes.

        Parameters
        ----------
        As : List[torch.Tensor]
            List of adjacency matrices.

        Returns
        -------
        List[float]
            Weights for each graph, proportional to their number of nodes.  Weights sum to
            1.
        """
        total_n = sum(A.shape[0] for A in As)
        return [A.shape[0] / total_n for A in As]

    def get_data(
        self, n: int, seed: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[float]]:
        """Sample graphs from the graphon and prepare experiment data.

        Parameters
        ----------
        n : int
            Number of nodes for each sampled graph.
        seed : Optional[int], optional
            Random seed for reproducibility, by default None.

        Returns
        -------
        As : List[torch.Tensor]
            List of adjacency matrices of shape (n, n).
        ps : List[torch.Tensor]
            List of uniform probability distributions over nodes.
        thetas : List[torch.Tensor]
            List of latent node positions used during sampling.
        lambdas : List[float]
            Barycenter weights for each graph (proportional to size).
        """
        set_seed_torch(seed)
        As = []
        thetas = []
        for ns in self.get_size_graphs(n):
            A, theta = self.graphon.sample(n=ns)
            As.append(A)
            thetas.append(theta)
        ps = [
            torch.ones(A.shape[0], device=A.device, dtype=A.dtype) / A.shape[0]
            for A in As
        ]
        lambdas = self.get_lambdas(As)
        return As, ps, thetas, lambdas


class ExperimentSBM(Experiment):
    """Experiment class for Stochastic Block Model (SBM) graphons.

    This class extends Experiment for SBM-specific configurations, where the number of
    clusters k is fixed based on the block structure.

    Attributes
    ----------
    k_max : int
        Maximum number of clusters, typically equal to the number of blocks.
    """

    def __init__(
        self,
        name: str,
        B: torch.Tensor,
        p: torch.Tensor,
        k_max: int = None,
        num_graphs: int = 1,
        **kwargs,
    ):
        """Initialize an SBM Experiment.

        Parameters
        ----------
        name : str
            Human-readable name for the experiment.
        B : torch.Tensor
            Block connectivity matrix of shape (K, K), where K is the number of
            communities. Entry B[i,j] is the edge probability between communities i and j.
        p : torch.Tensor
            Community membership probabilities of shape (K,). Must sum to 1.
        k_max : int, optional
            Maximum number of clusters for the barycenter. If None, defaults to the number
            of blocks in B.
        num_graphs : int, optional
            Number of graphs to sample per iteration, by default 1.
        **kwargs
            Additional keyword arguments passed to parent Experiment class.
        """
        super().__init__(
            name=name,
            graphon=SBM(B=B, p=p, device=B.device),
            num_graphs=num_graphs,
            **kwargs,
        )
        self.k_max = k_max if k_max is not None else B.shape[0]

    def get_k(self, n: int) -> int:
        """Return the fixed number of clusters for SBM.

        Parameters
        ----------
        n : int
            Number of nodes (unused, included for interface compatibility).

        Returns
        -------
        int
            Maximum number of clusters (fixed to number of SBM blocks).
        """
        return self.k_max
