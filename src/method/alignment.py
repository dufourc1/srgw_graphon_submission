import numpy as np
import ot


def get_permutation(C1, C2, p=None, q=None):
    """
    Computes the optimal permutation to align C2 to C1 using Gromov-Wasserstein.

    Parameters
    ----------
    C1 : array-like of shape (ns, ns)
        Target matrix
    C2 : array-like of shape (ns, ns)
        Matrix to be aligned
    p : array-like of shape (ns,), optional
        Distribution of C1 nodes. If None, defaults to uniform.
    q : array-like of shape (ns,), optional
        Distribution of C2 nodes. If None, defaults to uniform.

    Returns:
        A permutation array (indices) to reorder C2.
    """

    # Default to uniform distributions if p or q are not provided
    if p is None:
        p = ot.unif(C1.shape[0], type_as=C1)
    if q is None:
        q = ot.unif(C2.shape[0], type_as=C2)

    # 1. Compute Gromov-Wasserstein transport plan C1 is target, C2 is model
    plan = ot.gromov.gromov_wasserstein(C1, C2, p, q, loss_fun="square_loss")

    # 2. Derive the optimal permutation Find the row in C1 (target) that matches best with
    #    each col in C2 (model)
    best_match_rows = plan.argmax(axis=0)
    current_col_indices = np.arange(plan.shape[1])

    #    Sort the model columns based on which target row they map to.  This effectively
    #    reorders C2 nodes to align with C1 nodes 0, 1, 2...
    ordering = sorted(zip(best_match_rows, current_col_indices), key=lambda x: x[0])

    #    Extract the permutation indices
    perm = np.array([x[1] for x in ordering])

    return perm


def align(C1, C2, p=None, q=None, return_perm=False):
    """
    Aligns C2 to C1 using Gromov-Wasserstein optimal transport.

    Parameters
    ----------
    C1 : array-like of shape (ns, ns)
        Target matrix
    C2 : array-like of shape (ns, ns)
        Matrix to be aligned
    p : array-like of shape (ns,), optional
        Distribution of C1 nodes. If None, defaults to uniform.
    q : array-like of shape (ns,), optional
        Distribution of C2 nodes. If None, defaults to uniform.
    return_perm : bool, optional
        If True, also returns the permutation used for alignment.

    Returns:
        Aligned version of C2.
    """
    if q is None:
        q = ot.unif(C2.shape[0], type_as=C2)
    perm = get_permutation(C1, C2, p=p, q=q)
    C2_aligned = C2[perm][:, perm]

    if return_perm:
        return C2_aligned, q[perm], perm
    else:
        return C2_aligned, q[perm]


def align_all_to_first(C_list, p_list=None, plans=None):
    """
    Aligns all matrices in C_list to the reference matrix C_list[0].

    Parameters
    ----------
    C_list : list of array-like of shape (ns, ns)
        Matrices to be aligned
    p_list : list of array-like of shape (ns,), optional
        Distributions for each matrix in C_list. If None, defaults to uniform.
    plans: list of array-like of shape (*,ns), optional
        Plans linked to each matrix in C_list. If provided they will be permuted accordingly.

    Returns
    --------
        C_list: list of array-like of shape (ns, ns)
            Aligned matrices
        p_list: list of array-like of shape (*,ns)
            Updated distributions
        plans: list of array-like of shape (*,ns)
            Updated plans. Only returned if plans is provided.
    """
    if p_list is None:
        if plans is not None:
            p_list = [plan.sum(axis=0) for plan in plans]
        else:
            p_list = [ot.unif(C.shape[0], type_as=C) for C in C_list]

    for i in range(1, len(C_list)):
        C_list[i], p_list[i], perm = align(
            C_list[0], C_list[i], p=p_list[0], q=p_list[i], return_perm=True
        )
        if plans is not None:
            plans[i] = plans[i][:, perm]

    results = (C_list, p_list) if plans is None else (C_list, p_list, plans)
    return results
