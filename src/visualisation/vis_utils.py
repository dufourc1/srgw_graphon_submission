import os
import re

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from ..method.graphons import SBM


def index_to_str(index: int) -> str:
    return f"{index:02d}"


def get_info_from_yaml(exp_path: str, graphon_index: int):
    filepath = os.path.join(exp_path, index_to_str(graphon_index), "config.yaml")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, "r") as f:
        content = f.read()

    # Fix potential YAML issues with leading zeros (e.g. 09) treated as octal
    content = re.sub(r"(: +)0([0-9]*[89][0-9]*)(?=\s|$)", r'\1"0\2"', content)

    params = yaml.safe_load(content)
    n_start = params["nodes"]["n_start"]
    n_step = params["nodes"]["n_step"]
    n_end = params["nodes"]["n_end"]
    ns = list(range(n_start, n_end + 1, n_step))
    reps = params["num_reps"]
    number_graphs = params.get("number_graphs", 1)
    return ns, reps, number_graphs


def get_path_to_file(graphon_index: int, n: int, rep: int, base_path: str) -> str:
    return os.path.join(base_path, index_to_str(graphon_index), f"res_n{n}_rep{rep}.pt")


def merging_sizes_estimator(sizes, ns):
    if len(sizes) == 0:
        return np.array([])
    unified_size = np.zeros_like(sizes[0])
    n_tot = sum(ns)
    for i, n in enumerate(ns):
        weight = n / n_tot
        unified_size += weight * sizes[i]
    return unified_size


def load_sizes(res, key):
    qs = res[key]
    n = res["n"]
    if torch.is_tensor(n):
        n = n.item()

    sizes_list = []
    if isinstance(qs, list):
        for q in qs:
            sizes_list.append(q.numpy() if torch.is_tensor(q) else q)
    elif torch.is_tensor(qs):
        if qs.dim() == 1:
            sizes_list = [qs.numpy()]
        else:
            sizes_list = [t.numpy() for t in qs]
    else:
        sizes_list = [qs]

    if len(sizes_list) == 1:
        return sizes_list[0]

    return merging_sizes_estimator(sizes_list, [n] * len(sizes_list))


def remove_0_mass(theta, sizes, warn_cols=True):
    nonzero_inds = np.where(sizes > 1e-9)[0]

    if len(nonzero_inds) == 0:
        print("Warning: All estimated cluster weights are zero or close to zero.")
        sizes = np.ones_like(sizes) / len(sizes)
    else:
        theta = theta[np.ix_(nonzero_inds, nonzero_inds)]
        sizes = sizes[nonzero_inds]
        sizes = sizes / sizes.sum()

    theta = np.clip(theta, 0.0, 1.0)
    return theta, sizes


def load_estimator_from_file(filepath, warn_cols=False, remove_0_mass_bool=True):
    """
    Loads SBM estimators and metrics from a .pt file.
    """
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return None

    try:
        res = torch.load(filepath, map_location="cpu")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Process Plan estimator
    theta_hat = res["theta_hat"]
    if torch.is_tensor(theta_hat):
        theta_hat = theta_hat.numpy()
    q = load_sizes(res, "q")

    # Process Det (MAP) estimator
    theta_hat_det = res["theta_hat_det"]
    if torch.is_tensor(theta_hat_det):
        theta_hat_det = theta_hat_det.numpy()
    q_det = load_sizes(res, "q_det")

    def get_val(key):
        val = res.get(key, 0)
        if torch.is_tensor(val):
            return val.item()
        return val

    metrics = {
        "gw_A": get_val("loss_to_A"),
        "gw_theta": get_val("loss_to_theta"),
        "gw_A_det": get_val("loss_to_A_det"),
        "gw_theta_det": get_val("loss_to_theta_det"),
    }

    if remove_0_mass_bool:
        theta_hat, q = remove_0_mass(theta_hat, q, False)
        theta_hat_det, q_det = remove_0_mass(theta_hat_det, q_det, warn_cols)

    return {
        "model": SBM(theta_hat, q),
        "model_det": SBM(theta_hat_det, q_det),
        "metrics": metrics,
        "sizes_raw": res.get("q", []),
        "sizes_det_raw": res.get("q_det", []),
    }


def process_experiments_to_df(
    path_exp, indices_graphon, lb=1e-10, replace_negative_by_median=True
):
    """
    Iterates through experiment files and collects metrics into a Pandas DataFrame.
    """
    records = []
    logger.info(f"Processing experiments from {path_exp}")

    for graph_idx in indices_graphon:
        try:
            ns, reps, number_graphs = get_info_from_yaml(path_exp, graph_idx)
        except Exception as e:
            logger.warning(
                f"Error getting info from YAML for graph index {graph_idx}: {e}"
            )
            continue

        for n in ns:
            for rep in range(reps):
                filepath = get_path_to_file(graph_idx, n, rep, base_path=path_exp)

                # Check file existence
                if not os.path.exists(filepath):
                    # Record nan values to keep track of missing runs if desired, or just
                    # skip. Julia code tracked them. Let's record NaNs.
                    records.append(
                        {
                            "graphon_index": graph_idx,
                            "n": n,
                            "rep": rep,
                            "number_graphs": number_graphs,
                            "gw_theta": np.nan,
                            "gw_A": np.nan,
                            "num_blocks": np.nan,
                            "gw_theta_det": np.nan,
                            "gw_A_det": np.nan,
                            "num_blocks_det": np.nan,
                        }
                    )
                    continue

                data = load_estimator_from_file(filepath, warn_cols=False)
                if data is None:
                    continue

                metrics = data["metrics"]

                # Handle negative values
                gw = metrics["gw_theta"]
                if gw < 0:
                    gw = lb

                gw_d = metrics["gw_theta_det"]
                if gw_d < 0:
                    gw_d = lb

                records.append(
                    {
                        "graphon_index": graph_idx,
                        "n": n,
                        "rep": rep,
                        "number_graphs": number_graphs,
                        "gw_theta": gw,
                        "gw_A": metrics["gw_A"],
                        "num_blocks": data["model"].theta.shape[0],
                        "gw_theta_det": gw_d,
                        "gw_A_det": metrics["gw_A_det"],
                        "num_blocks_det": data["model_det"].theta.shape[0],
                    }
                )

    df = pd.DataFrame(records)

    if replace_negative_by_median and not df.empty:
        # Define a helper to replace NaN/Neg with group median
        def fill_median(g):
            return g.fillna(g.median())

        # Group by graphon_index and n to calculate medians across repetitions We process
        # specific columns
        cols_to_fix = [
            "gw_theta",
            "gw_A",
            "gw_theta_det",
            "gw_A_det",
            "num_blocks",
            "num_blocks_det",
        ]

        # Note: transform() returns a series/df aligned with original index
        for col in cols_to_fix:
            if col in df.columns:
                # Replace NaNs with median of the (graphon_index, n) group
                df[col] = df.groupby(["graphon_index", "n"])[col].transform(
                    lambda x: x.fillna(x.median())
                )

    return df


def save_results_to_csv(df, output_path):
    """Saves the DataFrame to a CSV file. If file exists, append without header."""
    if os.path.exists(output_path):
        df.to_csv(output_path, mode="a", header=False, index=False)
        print(f"Appended results to {output_path}")
    else:
        df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")


def load_results(
    path_exp, indices_graphon, csv_output_path=None, exp_name=None, **kwargs
):
    """
    Main entry point to load results.  Returns a DataFrame. If csv_output_path is
    provided, also saves to CSV.
    """
    df = process_experiments_to_df(path_exp, indices_graphon, **kwargs)

    if exp_name is not None:
        df["exp_name"] = exp_name

    if csv_output_path:
        save_results_to_csv(df, csv_output_path)
    return df
