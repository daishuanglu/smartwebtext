import torch

def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure that tensor is a 2d tensor

    In case of 1d tensor, let's expand the last dim
    """
    assert tensor.ndim in (1, 2)

    # Expand last dim in order to interpret this as (n, 1) points
    if tensor.ndim == 1:
        tensor = tensor[:, None]

    return tensor


def _radial_distance(X: torch.Tensor, control_points) -> torch.Tensor:
    """Compute the pairwise radial distances of the given points to the control points

        Input dimensions are not checked.

        Args:
            X (Tensor): N points in the source space
                Shape: (n, d_s)

        Returns:
            Tensor: The radial distance for each point to a control point (\\Phi(X))
                Shape: (n, n_c)
    """
    # Don't use mm for euclid dist, lots of imprecision comes from it (Will be a bit slower)
    dist = torch.cdist(X, control_points, compute_mode="donot_use_mm_for_euclid_dist")
    dist[dist == 0] = 1  # phi(r) = r^2 log(r) ->  (phi(0) = 0)
    return dist**2 * torch.log(dist)


def tps_transform_energy(X, Y, alpha=0.0, device='cpu'):
    X = X.to(device)
    Y = Y.to(device)
    X = _ensure_2d(X)
    Y = _ensure_2d(Y)
    assert X.shape[0] == Y.shape[0]

    n_c, d_s = X.shape

    phi = _radial_distance(X, X)

    # Build the linear system AP = Y
    X_p = torch.hstack([torch.ones((n_c, 1), device=device), X])

    A = torch.vstack(
        [
            torch.hstack([phi + alpha * torch.eye(n_c, device=device), X_p]),
            torch.hstack([X_p.T, torch.zeros((d_s + 1, d_s + 1), device=device)]),
        ]
    )

    Y = torch.vstack([Y, torch.zeros((d_s + 1, Y.shape[1]), device=device)])

    W = torch.linalg.solve(A, Y)
    Y_hat = torch.mm(A, W)
    return ((Y - Y_hat) ** 0.5).sum()