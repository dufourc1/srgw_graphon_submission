import torch


def kmeans(X, num_clusters, distance="euclidean", tol=1e-4, max_iter=100, device=None):
    """
    A pure PyTorch implementation of K-Means clustering.

    Args:
        X (torch.Tensor): Input data of shape (num_samples, num_features).  num_clusters
        (int): Number of clusters.  distance (str): 'euclidean' or 'cosine'.  tol (float):
        Tolerance for convergence.  max_iter (int): Maximum iterations.  device
        (torch.device): Target device (CPU/GPU).

    Returns:
        labels (torch.Tensor): Cluster labels for each sample.  centroids (torch.Tensor):
        Final cluster centroids.
    """
    if device is None:
        device = X.device

    X = X.to(device)
    num_samples = X.shape[0]

    # 1. Initialize Centroids (Random initialization)
    indices = torch.randperm(num_samples, device=device)[:num_clusters]
    centroids = X[indices]

    for i in range(max_iter):
        # 2. Compute Distances
        if distance == "euclidean":
            # Squared Euclidean distance: |x|^2 + |c|^2 - 2xc
            dists = torch.cdist(X, centroids, p=2)
        elif distance == "cosine":
            # Cosine distance: 1 - cosine_similarity
            X_norm = torch.nn.functional.normalize(X, dim=1)
            C_norm = torch.nn.functional.normalize(centroids, dim=1)
            dists = 1 - torch.mm(X_norm, C_norm.t())

        # 3. Assign Labels
        labels = torch.argmin(dists, dim=1)

        # 4. Update Centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(num_clusters):
            mask = labels == k
            if mask.sum() > 0:
                new_centroids[k] = X[mask].mean(dim=0)
            else:
                # Handle empty cluster by re-initializing randomly
                new_centroids[k] = X[torch.randint(0, num_samples, (1,)).item()]

        # 5. Check Convergence
        center_shift = torch.sum(
            torch.sqrt(torch.sum((new_centroids - centroids) ** 2, dim=1))
        )
        if center_shift < tol:
            break

        centroids = new_centroids

    return labels, centroids


def spectral_clustering(adj_matrix, num_clusters, normalize=True):
    """
    Performs Spectral Clustering on a graph adjacency matrix.

    Args:
        adj_matrix (torch.Tensor): Symmetric adjacency matrix (N, N).  num_clusters (int):
        Number of clusters (k).  normalize (bool): If True, uses Normalized Laplacian
        (L_sym).
                          If False, uses Unnormalized Laplacian (L).

    Returns:
        labels (torch.Tensor): Cluster assignments (0 to k-1).
    """
    # Ensure matrix is on the correct device and symmetric
    if not torch.allclose(adj_matrix, adj_matrix.T, atol=1e-6):
        print("Warning: Input matrix is not symmetric. Symmetrizing...")
        adj_matrix = (adj_matrix + adj_matrix.T) / 2

    device = adj_matrix.device
    N = adj_matrix.shape[0]

    # 1. Compute Degree Matrix D is a diagonal matrix where D_ii = sum(A_i)
    D_vec = torch.sum(adj_matrix, dim=1)

    # 2. Compute Laplacian
    if normalize:
        # Normalized Laplacian: L_sym = I - D^(-1/2) * A * D^(-1/2) Add epsilon to avoid
        # division by zero
        D_inv_sqrt = torch.pow(D_vec + 1e-8, -0.5)
        D_inv_sqrt_mat = torch.diag(D_inv_sqrt)

        # Pytorch matmul for efficiency
        L = torch.eye(N, device=device) - D_inv_sqrt_mat @ adj_matrix @ D_inv_sqrt_mat
    else:
        # Unnormalized Laplacian: L = D - A
        D_mat = torch.diag(D_vec)
        L = D_mat - adj_matrix

    # 3. Eigen Decomposition We use eigh because L is symmetric (Hermitian) This returns
    # eigenvalues in ascending order
    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    # 4. Select Top-k Eigenvectors The first eigenvalue is usually near 0.
    v_k = eigenvectors[:, :num_clusters]

    # 5. Normalize Rows (optional but recommended for Ng-Jordan-Weiss)
    if normalize:
        # Normalize rows to unit norm
        v_k = torch.nn.functional.normalize(v_k, p=2, dim=1)

    # 6. Clustering in Embedding Space Treat rows of v_k as points in R^k and cluster them
    # using K-Means
    labels, _ = kmeans(v_k, num_clusters=num_clusters, device=device)

    return labels
