from time import time

import torch
from loguru import logger
from ot.utils import unif

from ..method.alignment import align
from ..method.graphons import SBM
from ..method.srgw_barycenter import (
    gw_loss_barycenter,
    make_deterministic,
    srgw_barycenter,
)


def multistart_srgw(Cs, ps, n_start, k, initialization, lambdas=None, random_state=None, **kwargs):
    theta_hat, log = None, None
    loss_to_A = float("inf")

    for multistart in range(n_start):
        random_state = None if random_state is None else random_state + multistart
        theta_hat_, log_ = srgw_barycenter(
            N=k,
            Cs=Cs,
            ps=ps,
            lambdas=lambdas,
            G0_C=initialization,
            log=True,
            tol=1e-6,
            random_state=random_state,
            **kwargs,
        )

        loss_to_A_ = gw_loss_barycenter(Cs, theta_hat_, log_["T"], lambdas=lambdas)

        if n_start > 1:
            message = f"Multistart iteration {multistart} loss: {loss_to_A_:.4f}"
        else:
            message = f"Loss to A: {loss_to_A_:.4f}"

        logger.log(5, message)

        if loss_to_A_ < loss_to_A:
            theta_hat = theta_hat_
            log = log_
            loss_to_A = loss_to_A_

    log["loss_to_A"] = loss_to_A

    return theta_hat, log


def get_barycenter_and_losses(
    k,
    As,
    thetas,
    ps=None,
    lambdas=None,
    initialization="kmeans_torch",
    multistart=1,
    timing=False,
    **kwargs,
):
    if ps is None:
        ps = [unif(A.shape[0], type_as=A) for A in As]
    if lambdas is None:
        lambdas = [1.0 / len(As) for _ in As]

    if timing:
        start_time = time()
    theta_hat, log = multistart_srgw(
        Cs=As,
        ps=ps,
        n_start=multistart,
        k=k,
        initialization=initialization,
        lambdas=lambdas,
        **kwargs,
    )
    if timing:
        end_time = time()
        log["time"] = end_time - start_time

    loss_to_theta = gw_loss_barycenter(thetas, theta_hat, log["T"], lambdas=lambdas)

    logger.debug(f"Loss to theta: {loss_to_theta:.4f}")

    # convert to deterministic
    theta_hat_det, T_dets = make_deterministic(log["T"], As, lambdas=lambdas)
    q_dets = [torch.sum(plan, dim=0) for plan in T_dets]
    loss_to_A_det = gw_loss_barycenter(As, theta_hat_det, T_dets, lambdas=lambdas)

    loss_to_theta_det = gw_loss_barycenter(thetas, theta_hat_det, T_dets, lambdas=lambdas)

    results = {
        "theta_hat": theta_hat.detach().cpu(),
        "q": [p.detach().cpu() for p in log["p"]],
        "theta_hat_det": theta_hat_det.detach().cpu(),
        "q_det": [q.detach().cpu() for q in q_dets],
        "loss_to_A": log["loss_to_A"].item(),
        "loss_to_theta": loss_to_theta.item(),
        "loss_to_A_det": loss_to_A_det.item(),
        "loss_to_theta_det": loss_to_theta_det.item(),
        "k_max": k,
        "k": theta_hat.shape[0],
        "k_det": theta_hat_det.shape[0],
    }
    if timing:
        results["time"] = log["time"]

    return results


def get_sbm_estimator(
    A,
    k=None,
    initialization="kmeans_torch",
    n_start=1,
    graphon=None,
    return_log=False,
    random_state=None,
    warmstartT=True,
):
    """
    Estimate a Stochastic Block Model from an adjacency matrix A using
    the SRGW barycenter method in the paper "Network Learning with Semi-relaxed Gromov-Wasserstein".

    Parameters
    ----------
    A : torch.Tensor
        Adjacency matrix of the graph to estimate the SBM from.
    k : int, optional
        Number of blocks in the SBM. If None, it will be set to the square root of the number of nodes.
    initialization : str, optional
        Initialization method for the barycenter computation. Default is "kmeans_torch". Can be "random", "product", "kmeans_torch", or "spectral_torch".
    n_start : int, optional
        Number of random initializations for the multistart procedure. Default is 1.
    graphon : torch.Tensor or SBM, optional
        If provided, the estimated SBM will be aligned to this graphon. (Only used for visualization purposes.)
    return_log : bool, optional
        If True, the function will return the log dictionary containing additional information about the optimization process. Default is False.
    random_state : int, optional
        Random seed for reproducibility. Default is None.
    warmstartT : bool, optional
        Whether to use warm-starting for the transport plans. Default is True.
    """
    theta_hat, log = multistart_srgw(
        k=k,
        Cs=[A],
        ps=None,
        lambdas=[1.0],
        initialization=initialization,
        n_start=n_start,
        warmstartT=warmstartT,
        random_state=random_state,
    )

    theta_hat_det, T_dets = make_deterministic(log["T"], [A], lambdas=[1.0])
    log["T_map"] = log["T"]
    log["T"] = T_dets

    if graphon is not None:
        if torch.is_tensor(graphon):
            theta_hat_aligned, q_aligned = align(
                graphon,
                theta_hat_det,
                p=None,
                q=T_dets[0].sum(dim=0),
            )
        elif isinstance(graphon, SBM):
            theta_hat_aligned, q_aligned = align(
                graphon.theta,
                theta_hat_det,
                p=graphon.p,
                q=T_dets[0].sum(dim=0),
            )
        else:
            theta_hat_aligned, q_aligned = align(
                graphon.synthesize_graphon(r=2 * theta_hat_det.shape[0], func=graphon.func),
                theta_hat_det,
                p=None,
                q=T_dets[0].sum(dim=0),
            )

        sbm_hat = SBM(theta_hat_aligned, q_aligned)
    else:
        sbm_hat = SBM(theta_hat_det, T_dets[0].sum(dim=0))

    if return_log:
        return sbm_hat, log
    else:
        return sbm_hat
