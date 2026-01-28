# Estimating Graphons with Gromov-Wasserstein Barycenters

This repository contains the code for the method described in the paper:

> **"Network Learning with Semi-relaxed Gromov-Wasserstein: Continuous Relaxation with Discrete Limit"**  



## Installation

### Prerequisites

- Python ≥ 3.13

### Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.
Install with:

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup repository
git clone <repository-url>
cd <repository-directory>

# Create virtual env
uv venv
# Install dependencies (CPU)
uv sync
```

For installing `torch` for GPU, please refer to the [uv documentation](https://docs.astral.sh/uv/guides/integration/pytorch/#installing-pytorch).

### Using Marimo Notebooks

An interactive notebook is available for exploration the method of the paper:

```bash
uv run marimo edit example.py
```


## Overview

This project implements methods for estimating graphons (limit objects of dense
graph sequences) using Gromov-Wasserstein (GW) barycenters. Graphons are
fundamental objects in network analysis that can be used to model and analyze
large networks, providing a dimensionality reduction viewpoint on complex
networks.

The codebase includes:

- **Graphon sampling and models**: Implementation of various graphon models
  (continuous functions, Stochastic Block Models, Erdős-Rényi models)
- **GW barycenter computation**: Semi-relaxed Gromov-Wasserstein barycenter
  algorithms optimized for GPU computation
- **Multiple initialization strategies**: Various methods for initializing the
  barycenter (random, spectral clustering, K-means, etc.)

## Features

### Graphon Models

- **Continuous graphons**: Arbitrary functions on [0,1]²
- **Stochastic Block Models (SBM)**: Configurable block structures with various
  edge probabilities
- **Erdős-Rényi random graphs**: Simple probabilistic models

### GW Barycenter Algorithms

- **Semi-relaxed Gromov-Wasserstein (SRGW)**: Efficient computation of
  barycenters
- **GPU-optimized implementation**: Handles large-scale problems on CUDA
  devices
- **Multiple loss functions**: Support for different distance metrics
- **Warm-start capability**: Reuse previous solutions to accelerate convergence

### Initialization Methods

- **Product initialization**: Based on the product of pairwise GW alignments
- **Random initialization**: Baseline random starting point
- **Spectral clustering**: Initialization from spectral decomposition
- **K-means clustering**: CPU and GPU-accelerated variants
- **Soft variants**: Continuous relaxations of discrete clustering methods
