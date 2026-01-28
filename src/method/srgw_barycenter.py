## Code from the POT library, modified for GPU Will make a PR for fixing this in POT
##

import numpy as np
import torch
from loguru import logger
from ot.gromov import (
    semirelaxed_gromov_wasserstein,
    semirelaxed_init_plan,
    update_barycenter_structure,
)
from ot.utils import get_backend, list_to_array, unif
from torch_kmeans import KMeans

from .alignment import align_all_to_first
from .torch_spectral import spectral_clustering
from .utils import DEFAULT_DTYPE


# modified starting strategy G0_C and G0
def srgw_barycenter(
    N,
    Cs,
    ps=None,
    lambdas=None,
    loss_fun="square_loss",
    symmetric=True,
    max_iter=1000,
    tol=1e-9,
    stop_criterion="barycenter",
    warmstartT=True,
    verbose=False,
    log=False,
    init_C=None,
    G0_C="product",
    G0="product",
    n_tresh=5000,
    random_state=None,
    **kwargs,
):
    r"""
    Returns the Semi-relaxed Gromov-Wasserstein barycenters of `S` measured similarity
    matrices :math:`(\mathbf{C}_s)_{1 \leq s \leq S}`

    The function solves the following optimization problem with block coordinate descent:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad
        \sum_s \lambda_s \mathrm{srGW}(\mathbf{C}_s, \mathbf{p}_s, \mathbf{C})

    Where :

    - :math:`\mathbf{C}_s`: input metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S array-like of shape (ns, ns)
        Metric cost matrices
    ps : list of S array-like of shape (ns,), optional
        Sample weights in the `S` spaces.  If let to its default value None, uniform
        distributions are taken.
    lambdas : array-like of shape (S,) , optional
        List of the `S` spaces' weights.  If let to its default value None, uniform
        weights are taken.
    loss_fun : callable, optional
        tensor-matrix multiplication function based on specific loss function
    symmetric : bool, optional.
        Either structures are to be assumed symmetric or not. Default value is True.  Else
        if set to True (resp. False), C1 and C2 will be assumed symmetric (resp.
        asymmetric).
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on relative error (>0)
    stop_criterion : str, optional. Default is 'barycenter'.
        Stop criterion taking values in ['barycenter', 'loss']. If set to 'barycenter'
        uses absolute norm variations of estimated barycenters. Else if set to 'loss' uses
        the relative variations of the loss.
    warmstartT: bool, optional
        Either to perform warmstart of transport plans in the successive fused
        gromov-wasserstein transport problems.s
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : array-like of shape (N,N), optional.
        Random initial value for the :math:`\mathbf{C}` matrix provided by user.  Default
        is None and relies `G0` to produce an initial structure.
    init_T : list of S array-like of shape (N, ns), optional.
        Random initial value for the transport plans provided by user.  Default is None
        and relies on `G0` to produce initial transport plans.
    G0: str, optional. Default is 'product'.
        Initialization method for transport plans calling
        :func:`ot.gromov.semirelaxed_init_plan`, and taking values in "product",
        "random_product", "random", "fluid", "fluid_soft", "spectral", "spectral_soft",
        "kmeans", "kmeans_soft".
    G0_C : str, optional. Default is 'product'.
        Initialization method for barycenter structure when `init_C=None`, calling
        :func:`ot.gromov.semirelaxed_init_plan`.  Transport plans are used to deduce an
        initial barycenter structure if `init_C=None`.
    n_tresh : int, optional
        Threshold on graph size to use partial cost matrix in `kmeans_torch`
        initialization.  If graph size is larger than `n_tresh`, only half of the cost
        matrix columns are used to perform clustering. (To reduce memory consumption).
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    C : array-like, shape (`N`, `N`)
        Barycenters' structure matrix
    log : dict
        Only returned when log=True. It contains the keys:

        - :math:`\mathbf{T}`: list of (`N`, `ns`) transport matrices
        - :math:`\mathbf{p}`: (`N`,) barycenter weights
        - values used in convergence evaluation.

    References
    ----------
    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas
            Courty.  "Semi-relaxed Gromov-Wasserstein divergence and applications on
            graphs" International Conference on Learning Representations (ICLR), 2022.

    """
    if stop_criterion not in ["barycenter", "loss"]:
        raise ValueError(
            f"Unknown `stop_criterion='{stop_criterion}'`. Use one of: {'barycenter', 'loss'}."
        )

    arr = [*Cs]
    if ps is not None:
        arr += [*ps]
    else:
        ps = [unif(C.shape[0], type_as=C) for C in Cs]

    nx = get_backend(*arr)

    S = len(Cs)
    if lambdas is None:
        lambdas = nx.ones(S) / S
    else:
        lambdas = list_to_array(lambdas)
        lambdas = nx.from_numpy(lambdas)

    # Initialization of transport plans and C (if not provided by user)
    if init_C is None:
        init_C = nx.zeros((N, N), type_as=Cs[0])
        if G0_C in ["product", "random_product", "random"]:
            T = [
                semirelaxed_init_plan(
                    Cs[i],
                    init_C,
                    p=(
                        nx.to_numpy(ps[i]) if G0_C == "random" else ps[i]
                    ),  # TODO: hack to avoid issues, need to be fixed in POT
                    method=G0_C,
                    use_target=False,
                    random_state=random_state,
                    nx=nx,
                )
                for i in range(S)
            ]
            T = align_init_plans(Cs, T, nx, loss_fun=loss_fun)
            C = update_barycenter_structure(T, Cs, lambdas, loss_fun=loss_fun, nx=nx)

            if G0_C in ["product", "random_product"]:
                # initial structure is constant so we add a small random noise to avoid
                # getting stuck at init
                np.random.seed(random_state)
                noise = np.random.uniform(-0.01, 0.01, size=(N, N))
                if symmetric:
                    noise = (noise + noise.T) / 2.0
                # TODO: make PR (line below): otherwise this errors with CUDA
                noise = nx.from_numpy(noise, type_as=C)
                C = C + noise

        # this is also new, more performant for big Cs
        elif G0_C in ["kmeans_torch", "kmeans_torch_soft"]:
            # this is very hacky and assume torch tensors...
            model = KMeans(n_clusters=torch.tensor(N).item(), num_init=1, verbose=False)
            T = [None] * S
            for s in range(S):
                if Cs[s].shape[0] > n_tresh:
                    indices = torch.randperm(Cs[s].shape[0], device=Cs[s].device)[
                        0:n_tresh
                    ]
                    Cs_tensor = Cs[s][:, indices].to(dtype=torch.half)
                else:
                    Cs_tensor = Cs[s]
                part = model(torch.unsqueeze(Cs_tensor, 0)).labels.squeeze()
                T_, q_ = get_transport_from_partition(part, Cs[s], init_C, ps[s], nx)

                if "soft" in G0_C:
                    T_ = (T_ + nx.outer(ps[0], q_)) / 2.0
                else:
                    T[s] = T_
                del Cs_tensor
            del model
            T = align_init_plans(Cs, T, nx, loss_fun=loss_fun)
            C = update_barycenter_structure(
                T, Cs, lambdas, loss_fun=loss_fun, nx=nx
            ).to(dtype=DEFAULT_DTYPE)

        elif G0_C in ["spectral_torch", "spectral_torch_soft"]:
            T = [None] * S
            for s in range(S):
                part = spectral_clustering(Cs[s], N, normalize=True)
                T_, q_ = get_transport_from_partition(part, Cs[s], init_C, ps[s], nx)

                if "soft" in G0_C:
                    T_ = (T_ + nx.outer(ps[0], q_)) / 2.0
                else:
                    T[s] = T_
            T = align_init_plans(Cs, T, nx, loss_fun=loss_fun)
            C = update_barycenter_structure(T, Cs, lambdas, loss_fun=loss_fun, nx=nx)

        else:  # relies on partitioning of inputs
            shapes = np.array([C.shape[0] for C in Cs])
            large_graphs_idx = np.where(shapes > N)[0]
            small_graphs_idx = np.where(shapes <= N)[0]
            T = [None] * S
            list_init_C = []  # store different barycenter structure to average

            # we first compute an initial informative barycenter structure on graphs we
            # can compress then use it on graphs to expand
            for indices in [large_graphs_idx, small_graphs_idx]:
                if len(indices) > 0:
                    sub_T = [
                        semirelaxed_init_plan(
                            Cs[i],
                            init_C,
                            ps[i],
                            method=G0_C,
                            use_target=False,
                            random_state=random_state,
                            nx=nx,
                        )
                        for i in indices
                    ]
                    sub_Cs = [Cs[i] for i in indices]
                    sub_lambdas = lambdas[indices] / nx.sum(lambdas[indices])
                    init_C = update_barycenter_structure(
                        sub_T, sub_Cs, sub_lambdas, loss_fun=loss_fun, nx=nx
                    )
                    for i, idx in enumerate(indices):
                        T[idx] = sub_T[i]
                    list_init_C.append(init_C)

            if len(list_init_C) == 2:
                init_C = update_barycenter_structure(
                    T, Cs, lambdas, loss_fun=loss_fun, nx=nx
                )
            C = init_C

    else:
        C = init_C
        T = [
            semirelaxed_init_plan(
                Cs[i],
                C,
                ps[i],
                method=G0_C,
                use_target=True,
                random_state=random_state,
                nx=nx,
            )
            for i in range(S)
        ]

    if stop_criterion == "barycenter":
        inner_log = False
    else:
        inner_log = True
        curr_loss = 1e15

    if log:
        log_ = {}
        log_["err"] = []
        if stop_criterion == "loss":
            log_["loss"] = []

    for cpt in range(max_iter):
        if stop_criterion == "barycenter":
            Cprev = C
        else:
            prev_loss = curr_loss

        # get transport plans
        if warmstartT:
            res = [
                semirelaxed_gromov_wasserstein(
                    Cs[s],
                    C,
                    ps[s],
                    loss_fun,
                    symmetric,
                    G0=T[s],
                    max_iter=max_iter,
                    tol_rel=tol,
                    tol_abs=0.0,
                    log=inner_log,
                    verbose=verbose,
                    **kwargs,
                )
                for s in range(S)
            ]
        else:
            res = [
                semirelaxed_gromov_wasserstein(
                    Cs[s],
                    C,
                    ps[s],
                    loss_fun,
                    symmetric,
                    G0=G0,
                    max_iter=max_iter,
                    tol_rel=tol,
                    tol_abs=0.0,
                    log=inner_log,
                    verbose=verbose,
                    **kwargs,
                )
                for s in range(S)
            ]

        if stop_criterion == "barycenter":
            T = res
        else:
            T = [output[0] for output in res]
            curr_loss = np.sum([output[1]["srgw_dist"] for output in res])

        # update barycenters
        p = nx.concatenate([nx.sum(T[s], 0)[None, :] for s in range(S)], axis=0)

        C = update_barycenter_structure(T, Cs, lambdas, p, loss_fun, nx=nx)

        # update convergence criterion
        if stop_criterion == "barycenter":
            err = nx.norm(C - Cprev)
            if log:
                log_["err"].append(err)

        else:
            err = abs(curr_loss - prev_loss) / prev_loss if prev_loss != 0.0 else np.nan
            if log:
                log_["loss"].append(curr_loss)
                log_["err"].append(err)

        if verbose:
            if cpt % 200 == 0:
                print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
            print("{:5d}|{:8e}|".format(cpt, err))

        if err <= tol:
            break
    if log:
        log_["T"] = T
        log_["p"] = p

        return C, log_
    else:
        return C


# this also has been modified, only used in kmeans_torch
def get_transport_from_partition(part, A, C, p, nx):
    """
    Convert a partition of nodes (strong assignments) to the corresponding transport map
    """
    T = nx.eye(C.shape[0], type_as=A)[part]
    T = p[:, None] * T
    q = nx.sum(T, 0)
    return T, q


def gw_loss_barycenter(
    Cs: list,
    C2: torch.Tensor,
    Ts: list,
    lambdas: list = None,
) -> torch.Tensor:
    """
    Computes the sum of Gromov-Wasserstein costs between a barycenter Cb and a list of
    cost matrices Cs, weighted by lambdas.

    Args:
        Cs: List of (Ns, Ns) Cost matrices of sources.  C2: (N, N) Cost matrix of
        barycenter.  Ts: List of (Ns, N) Transport plans (coupling matrices).  lambdas:
        List of weights for each source cost matrix.

    Returns:
        Scalar tensor representing the weighted sum of GW costs.
    """

    mus = [torch.sum(T, dim=1) for T in Ts]
    nus = [torch.sum(T, dim=0) for T in Ts]
    if lambdas is None:
        lambdas = [1.0 / len(Cs)] * len(Cs)

    C2_sq = C2**2

    total_loss = 0.0
    # TODO: check if batched implementation is faster (would need to convert list of
    # tensors to a single tensor)
    for i, C1 in enumerate(Cs):
        c2_sq_term = torch.dot(nus[i], torch.mv(C2_sq, nus[i]))
        c1_sq_term = torch.dot(mus[i], torch.mv(C1**2, mus[i]))
        cross_term_mat = torch.matmul(torch.matmul(C1, Ts[i]), C2.t())
        cross_term = -2 * torch.sum(cross_term_mat * Ts[i])
        loss_inter = c1_sq_term + c2_sq_term + cross_term

        # floating point error: forcing input matrices to be double solves the issue, but
        # is not an option on l40s GPUs
        if loss_inter < 0:
            logger.warning(f"Negative GW loss {loss_inter.item()}, recomputing.")
            try:
                loss_inter = gromov_wasserstein_vec_broadcasting(C1, C2, Ts[i])
            except Exception as e:
                logger.error(f"Recomputation of GW loss failed, {e}")
                logger.error("Setting loss to 0.")
                loss_inter = torch.tensor(0.0, device=C1.device)
            logger.warning(f"Recomputed GW loss: {loss_inter.item()}")
        total_loss += lambdas[i] * loss_inter
    return total_loss


def gromov_wasserstein_vec_broadcasting(
    C1: torch.Tensor, C2: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized implementation using 4D broadcasting.  Complexity: Time O(N^2 M^2), Memory
    O(N^2 M^2)
    """
    C1_ik = C1[:, None, :, None]
    C2_jl = C2[None, :, None, :]
    cost_sq_ijkl = (C1_ik - C2_jl) ** 2
    T_ij = T[:, :, None, None]
    T_kl = T[None, None, :, :]
    loss = (cost_sq_ijkl * T_ij * T_kl).sum()
    return loss


def convert_plan_to_map(T_plan: torch.Tensor, method: str = "argmax"):
    """
    Convert a probabilistic plan to a deterministic map via `method`

    Args:
        T_plan: (n, k) Transport plan (coupling matrix).  method: default to `"argmax"`
        way to convert plan

    Returns:
        T_map: torch.Tensor (n,k) transport map, assigning each row to exactly one column
        (no mass splitting on [n])
    """
    n = T_plan.shape[0]
    if method == "argmax":
        z_n_k = torch.argmax(T_plan, dim=1)
    else:
        raise ValueError(f"method {method} not implemented")
    T_map = torch.zeros_like(T_plan)
    T_map[torch.arange(n), z_n_k] = 1.0 / n
    return T_map


def convert_plans_to_maps_and_drop_zero_mass(
    plan: torch.Tensor, method: str = "argmax", thresh: float = 0
):
    return drop_zero_mass(convert_plan_to_map(plan, method), thresh=thresh)


def make_deterministic(
    plans, Cs, lambdas=None, method="argmax", thresh=0, drop_zero_mass=True
):
    if lambdas is None:
        lambdas = [1.0 / len(plans)] * len(plans)

    # multiple graphs as input -> first convert plans to maps then update barycenter,
    # cannot drop zero mass of plan
    if not drop_zero_mass or len(plans) > 1:
        maps = [convert_plan_to_map(plan, method) for plan in plans]
        theta = update_barycenter_structure(maps, Cs, lambdas=lambdas)
    else:
        maps = [
            convert_plans_to_maps_and_drop_zero_mass(plan, method, thresh=thresh)[0]
            for plan in plans
        ]
        theta = update_barycenter_structure(maps, Cs, lambdas=lambdas)
    return theta, maps


def drop_zero_mass(plan, mu=None, thresh=0):
    """Drop the columns of plans with zero mass.

    Parameters
    ----------
    plan : torch.Tensor
        Transport plan to process. A 2D tensor (n x k).
    mu : torch.Tensor, optional
        Right marginal corresponding to the plan of shape (k,), by default None.
    thresh : float, optional
        Threshold below which a mass is considered zero, by default 0.
    """
    if mu is None:
        mu = torch.sum(plan, dim=0)

    nonzero_cols = torch.where(mu > thresh)[0]
    new_plan = plan[:, nonzero_cols]
    new_mu = torch.sum(new_plan, dim=0)
    return new_plan, new_mu


def align_init_plans(Cs, T, nx, loss_fun="square_loss"):
    # at initialization, align all lower dimensional representation
    # maybe this will help with aligning the barycenters learned
    S = len(Cs)
    if len(Cs) > 1:
        C_inter = [
            update_barycenter_structure(
                [T[i]], [Cs[i]], [1.0], loss_fun=loss_fun, nx=nx
            )
            for i in range(S)
        ]
        C_inter, _, T = align_all_to_first(C_inter, p_list=None, plans=T)
    return T
