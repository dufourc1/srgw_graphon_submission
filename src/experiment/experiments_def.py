from enum import Enum

import torch
from loguru import logger

from ..method.graphons import ER, SBM, Graphon
from .experiments_class import Experiment, ExperimentSBM


class ExperimentType(str, Enum):
    continuous = "continuous"
    continuous_multiple = "continuous_multiple"
    sbm_with_true_k = "sbm_with_true_k"
    sbm_sparsity = "sbm_sparsity"
    sbm_multiple = "sbm_multiple"


GRAPHONS_FUNC = [
    lambda u, v: u * v,
    lambda u, v: torch.exp(-(u**0.7 + v**0.7)),
    lambda u, v: 0.25 * (u**2 + v**2 + u**0.5 + v**0.5),
    lambda u, v: 0.5 * (u + v),
    lambda u, v: 1 / (1 + torch.exp(-2 * (u**2 + v**2))),
    lambda u, v: 1 / (1 + torch.exp(-(torch.maximum(u, v) ** 2 + torch.minimum(u, v) ** 4))),
    lambda u, v: torch.exp(-(torch.maximum(u, v) ** 0.75)),
    lambda u, v: torch.exp(-0.5 * (torch.minimum(u, v) + u**0.5 + v**0.5)),
    lambda u, v: torch.log(1 + 0.5 * torch.maximum(u, v)),
    lambda u, v: torch.abs(u - v),
    lambda u, v: 1 - torch.abs(u - v),
    lambda u, v: 0.5 + 0.5 * torch.sin(2 * torch.pi * u) * torch.sin(2 * torch.pi * v),
    lambda u, v: 0.5
    + torch.cos(2 * torch.pi * (u - v)) * 0.25 * (torch.sin(4 * torch.pi * u) + torch.sin(4 * torch.pi * v)),
]


GRAPHONS_STR = [
    "u*v",
    "exp(-(u^0.7 + v^0.7))",
    "0.25*(u^2 + v^2 + u^0.5 + v^0.5)",
    "0.5*(u + v)",
    "1/(1+exp(-2*(u^2 + v^2)))",
    "1/(1+exp(-(max(u,v)^2 + min(u,v)^4)))",
    "exp(-max(u,v)^0.75)",
    "exp(-0.5*(min(u,v) + u^0.5 + v^0.5))",
    "log(1 + 0.5*max(u,v))",
    "|u - v|",
    "1 - |u - v|",
    "0.5 + 0.5*sin(2*pi*u)*sin(2*pi*v)",
    "0.5 + cos(2*pi*(u - v))*0.25*(sin(4*pi*u) + sin(4*pi*v))",
]


SBMs = [
    SBM(torch.tensor([[0.8, 0.2], [0.2, 0.8]])),
    SBM(torch.tensor([[0.8, 0.2], [0.2, 0.8]]), torch.tensor([0.3, 0.7])),
    SBM(torch.tensor([[0.2, 0.8], [0.8, 0.2]])),
    SBM(
        torch.tensor([[0.8, 0.1, 0.2], [0.1, 0.7, 0.15], [0.2, 0.15, 0.6]]),
        torch.tensor([0.3, 0.3, 0.4]),
    ),
    ER(0.5),
    ER(0.8),
    ER(0.3),
]


SBMs_STR = [
    "SBM_2blocks_associative",
    "SBM_2blocks_associative_unequal",
    "SBM_2blocks_disassociative",
    "SBM_3blocks_associative_unequal",
    "ER_0.5",
    "ER_0.8",
    "ER_0.3",
]


def get_experiment(
    id: int,
    experiment_type: ExperimentType = ExperimentType.continuous,
    **kwargs,
) -> Experiment:
    number_graphs: int = kwargs.get("number_graphs", 1)
    if number_graphs > 1 and "multiple" not in experiment_type:
        logger.error(
            "number_graphs > 1 ignored for experiment_type "
            f"'{experiment_type}'. Use 'continuous_multiple' or "
            "'sbm_multiple' instead."
        )
    elif number_graphs == 1 and "multiple" in experiment_type:
        logger.error(f"number_graphs = 1 for experiment_type '{experiment_type}'Setting number_graphs = 2.")
        number_graphs = 2
    kwargs.update({"number_graphs": number_graphs})

    if experiment_type == ExperimentType.continuous:
        return get_experiment_continuous(id, **kwargs)
    elif experiment_type == ExperimentType.continuous_multiple:
        return get_experiment_continuous(id, **kwargs)
    elif experiment_type == ExperimentType.sbm_with_true_k:
        k_max = kwargs.get("k_max", None)
        if k_max is not None:
            kwargs.update({"k_max": k_max})
            logger.warning("k_max parameter is ignored for 'sbm' experiment_type. Use 'sbm_sparsity instead.")
        return get_experiment_sbm(id, **kwargs)
    elif experiment_type == ExperimentType.sbm_sparsity:
        k_max: int = kwargs.get("k_max", 20)
        kwargs.update({"k_max": k_max})
        return get_experiment_sbm(id, **kwargs)
    elif experiment_type == ExperimentType.sbm_multiple:
        return get_experiment_sbm(id, **kwargs)
    else:
        logger.error(f"Unknown experiment type {experiment_type}")


def get_elt_and_name(list_obj: list, list_str: list, index: int) -> tuple:
    if index < 0 or index >= len(list_obj):
        logger.error(f"Invalid {index}. Must be in [0,{len(list_obj) - 1}]")
    return list_obj[index], list_str[index]


def get_experiment_continuous(
    experiment_id: int,
    number_graphs: int = 1,
    device: str = None,
) -> Experiment:
    func, name = get_elt_and_name(GRAPHONS_FUNC, GRAPHONS_STR, experiment_id)
    name_save = f"graphon_continuous_{experiment_id}"
    graphon = Graphon(
        func=func,
        device=device,
        r=15000,
    )
    return Experiment(
        name=name,
        graphon=graphon,
        num_graphs=number_graphs,
        name_save=name_save,
    )


def get_experiment_sbm(
    experiment_id: int,
    number_graphs: int = 1,
    device: str = None,
    k_max: int = None,
) -> ExperimentSBM:
    sbm, name = get_elt_and_name(SBMs, SBMs_STR, experiment_id)
    if device is not None:
        sbm = sbm.to(device)
    return ExperimentSBM(
        name=name,
        B=sbm.theta,
        p=sbm.p,
        num_graphs=number_graphs,
        k_max=k_max,
    )
