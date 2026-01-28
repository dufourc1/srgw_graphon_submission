import torch
from src.method.graphons import Graphon
import pytest
from src.experiment.experiments_def import GRAPHONS_FUNC


def slow_apply_function(x, y, func, out):
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            out[i, j] = func(x[i], y[j])
    return out


@pytest.mark.parametrize("func", GRAPHONS_FUNC)
def test_graphon(func):
    r = 100
    graphon = Graphon(r=r, func=func)
    theta = graphon.theta.cpu()

    # check square
    assert theta.shape == (r, r)
    # check symmetric
    assert torch.allclose(theta, theta.T)
    theta_ref = torch.zeros_like(theta)
    xs = (torch.arange(0, r, dtype=torch.float32)) / (r - 1)
    slow_apply_function(xs, xs, func, theta_ref)

    # check same as slow func apply
    assert torch.allclose(theta, theta_ref), f"Graphon {id} test failed!"

    # check generated graphs are symmetric
    A, P = graphon.sample(20, 0, False)
    unique_A = torch.sort(torch.unique(A)).values
    assert torch.all(unique_A == torch.tensor([0, 1], dtype=A.dtype))
    assert torch.all(A == A.T)
    assert torch.allclose(P, P.T)
