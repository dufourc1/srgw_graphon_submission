import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import os
    import sys

    import marimo as mo

    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = True
    import numpy as np
    import torch

    from src.experiment.experiments_def import GRAPHONS_FUNC, GRAPHONS_STR, SBMs_STR, SBMs
    from src.experiment.experiments_runner import multistart_srgw, get_sbm_estimator
    from src.method.alignment import align
    from src.method.graphons import SBM, Graphon, plot_graphon
    from src.method.srgw_barycenter import make_deterministic
    return (
        GRAPHONS_FUNC,
        GRAPHONS_STR,
        Graphon,
        SBMs,
        SBMs_STR,
        get_sbm_estimator,
        mo,
        plot_graphon,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(GRAPHONS_STR, mo):
    graphon_index = mo.ui.dropdown(
        options={GRAPHONS_STR[i]: i for i in range(len(GRAPHONS_STR))},
        value=GRAPHONS_STR[0],
        label="pick a graphon",
    )

    init_slider = mo.ui.dropdown(
        options={
            "random": "random",
            "product": "product",
            "kmeans_torch": "kmeans_torch",
            "spectral_torch": "spectral_torch",
        },
        value="product",
        label="init",
    )

    n_slider = mo.ui.slider(
        start=10, stop=1000, step=10, value=200, label="n", debounce=True
    )
    k_slider = mo.ui.slider(
        start=1, stop=100, step=1, value=10, label="k", debounce=True
    )
    sort_latent_check = mo.ui.checkbox(value=False, label="sorted latents")

    mo.hstack([graphon_index, n_slider, k_slider, sort_latent_check, init_slider])
    return graphon_index, init_slider, k_slider, n_slider, sort_latent_check


@app.cell(hide_code=True)
def _(
    GRAPHONS_FUNC,
    Graphon,
    graphon_index,
    k_slider,
    n_slider,
    sort_latent_check,
    torch,
):
    graphon = Graphon(GRAPHONS_FUNC[graphon_index.value])
    n = n_slider.value
    k = k_slider.value
    torch.manual_seed(1235436)

    latents = torch.rand(n)
    if sort_latent_check.value:
        latents = torch.sort(latents)[0]

    A, theta = graphon.sample(n, sorted=sort_latent_check.value)
    return A, graphon, k, n


@app.cell(hide_code=True)
def _(A, get_sbm_estimator, graphon, init_slider, k):
    sbm_hat = get_sbm_estimator(A, k, graphon=graphon, initialization=init_slider.value)
    return (sbm_hat,)


@app.cell(hide_code=True)
def _(
    A,
    GRAPHONS_STR,
    graphon,
    graphon_index,
    k,
    mo,
    n,
    plot_graphon,
    plt,
    sbm_hat,
):
    fig, axs = plt.subplots(1, 3, layout="compressed", figsize=(15, 4))

    ax_theta_hat = axs[1]
    ax_A = axs[0]
    ax_graphon = axs[2]

    im_A = ax_A.spy(A)
    ax_A.set_title(f"A, n={n}")
    plot_graphon(sbm_hat, ax=ax_theta_hat)
    ax_theta_hat.set_title(rf"B,  $k={sbm_hat.theta.shape[0]} \leq k_{{max}}={k}$")
    _, im_graphon = plot_graphon(graphon, ax=ax_graphon, colorbar=True)
    ax_graphon.set_title(rf"${GRAPHONS_STR[graphon_index.value]}$")

    for ax_ in axs.flatten():
        ax_.set_box_aspect(1)
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)
    mo.mpl.interactive(plt.gcf())
    fig
    return (ax_A,)


@app.cell
def _(SBMs_STR, mo):
    sbm_index = mo.ui.dropdown(
        options={SBMs_STR[i]: i for i in range(len(SBMs_STR))},
        value=SBMs_STR[0],
        label="pick a sbm",
    )

    init_slider_sbm = mo.ui.dropdown(
        options={
            "random": "random",
            "product": "product",
            "kmeans_torch": "kmeans_torch",
            "spectral_torch": "spectral_torch",
        },
        value="product",
        label="init",
    )

    n_slider_sbm = mo.ui.slider(
        start=10, stop=1000, step=10, value=200, label="n", debounce=True
    )
    k_slider_sbm = mo.ui.slider(
        start=1, stop=100, step=1, value=10, label="k", debounce=True
    )

    mo.hstack([sbm_index, n_slider_sbm, k_slider_sbm, init_slider_sbm])
    return init_slider_sbm, k_slider_sbm, n_slider_sbm, sbm_index


@app.cell
def _(
    SBMs,
    get_sbm_estimator,
    init_slider_sbm,
    k_slider_sbm,
    n_slider_sbm,
    sbm_index,
    torch,
):
    sbm = SBMs[sbm_index.value]
    n_sbm = n_slider_sbm.value
    k_sbm = k_slider_sbm.value
    torch.manual_seed(123543)

    A_sbm, theta_sbm = sbm.sample(n_sbm, sorted=True)
    perm = torch.randperm(A_sbm.shape[0])
    A_sbm = A_sbm[:,perm][perm]

    sbm_hat_sbm = get_sbm_estimator(A_sbm, k_sbm,initialization=init_slider_sbm.value, graphon = theta_sbm, n_start=20)
    return A_sbm, k_sbm, n_sbm, sbm, sbm_hat_sbm


@app.cell
def _(
    A_sbm,
    SBMs_STR,
    ax_A,
    k_sbm,
    mo,
    n_sbm,
    plot_graphon,
    plt,
    sbm,
    sbm_hat_sbm,
    sbm_index,
):
    _fig, _axs = plt.subplots(1, 3, layout="compressed", figsize=(15, 4))

    _ax_theta_hat = _axs[1]
    _ax_A = _axs[0]
    _ax_graphon = _axs[2]

    _im_A = _ax_A.spy(A_sbm)
    ax_A.set_title(f"A, n={n_sbm}")
    plot_graphon(sbm_hat_sbm, ax=_ax_theta_hat)
    _ax_theta_hat.set_title(rf"B,  $k={sbm_hat_sbm.theta.shape[0]} \leq k_{{max}}={k_sbm}$")
    _, _ = plot_graphon(sbm, ax=_ax_graphon, colorbar=True)
    _ax_graphon.set_title(f"{SBMs_STR[sbm_index.value]}")

    for _ax_ in _axs.flatten():
        _ax_.set_box_aspect(1)
        _ax_.get_xaxis().set_visible(False)
        _ax_.get_yaxis().set_visible(False)
    mo.mpl.interactive(plt.gcf())
    _fig
    return


if __name__ == "__main__":
    app.run()
