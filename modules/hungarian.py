"""The module implementing the Hungarian loss function."""

import torch
import torch.nn as nn

from modules import ops


def compute_pairwise_focal_bce(y_true, y_pred, alpha=0.5, gamma=2.0):
    # p - n*w*h
    # y - n*w*h
    # return: n*n pairwise cost matrix
    p = y_pred.view(y_pred.size(0), -1)
    y = y_true.view(y_true.size(0), -1)
    loss = -(torch.matmul(y, torch.log(p).T) + torch.matmul(1- y, torch.log(1 - p).T))
    pt = torch.exp(-loss)
    focal_dist = (alpha * (1 - pt) ** gamma * loss)
    return focal_dist


def compute_euclidean_distance(
        a: torch.tensor, b: torch.tensor):  # pylint: disable=invalid-name
    """
    Computes euclidean distance between two inputs `a` and `b`.

    The computation is performed using the following formula
    `dist = sqrt((a - b)^2) = sqrt(a^2 - 2ab.T - b^2)`

    Example:
    >>> a = torch.tensor(
    >>>     [[1., 2., 3., 4.],
    >>>      [5., 6., 7., 8.]]
    >>> )
    >>> b = torch.tensor(
    >>>     [[1., 1., 1., 1.],
    >>>      [2., 2., 2., 2.]]
    >>> )
    >>> compute_euclidean_distance(a,b)

    >>> torch.tensor(
    >>>     [[ 3.7416575,  2.4494898],
    >>>      [11.224972 ,  9.273619 ]] dtype=float32)

    Args:
        a:
            The 2D-tensor [num_entities, num_dimesions] of floats.
        b:
            The 2D-tensor [num_entities, num_dimesions] of floats.

    Result:
        The 2D tensor [num_entities, num_entities] with computed
        distances between entities.
    """
    a = a.view(a.size(0), -1)
    b = b.view(b.size(0), -1)
    a2 = torch.sum(a ** 2, dim=1).view(-1, 1)
    b2 = torch.sum(b ** 2, dim=1).view(1, -1)
    dist = (a2 - 2 * torch.matmul(a, b.T) + b2)
    return dist ** 0.5


def reduce_rows(matrix: torch.Tensor) -> torch.Tensor:
    """
    Subtracts the minimum value from each row.

    Example:
    >>> matrix = torch.Variable(
    >>>    [[ 30., 25., 10.],
    >>>     [ 15., 10., 20.],
    >>>     [ 25., 20., 15.]]
    >>> )
    >>> reduce_rows(matrix)

    >>> torch.Tensor(
    >>>     [[20. 15.  0.]
    >>>      [ 5.  0. 10.]
    >>>      [10.  5.  0.]], shape=(3, 3), dtype=float32)

    Args:
        matrix:
            The 2D-tensor [rows, columns] of floats to reduce.

    Returns:
        A new tensor with reduced values of the same shape as
        the input tensor.
    """
    return matrix - matrix.min(dim=1, keepdims=True).values


def reduce_cols(matrix: torch.Tensor) -> torch.Tensor:
    """
    Subtracts the minimum value from each column.

    Example:
    >>> matrix = torch.Variable(
    >>>    [[ 30., 25., 10.],
    >>>     [ 15., 10., 20.],
    >>>     [ 25., 20., 15.]]
    >>> )
    >>> reduce_cols(matrix)

    >>> torch.Tensor(
    >>>     [[15. 15.  0.]
    >>>      [ 0.  0. 10.]
    >>>      [10. 10.  5.]], shape=(3, 3), dtype=float32)

    Args:
        matrix:
            The 2D-tensor [rows, columns] of floats to reduce.

    Returns:
        A new tensor with reduced values of the same shape as
        the input tensor.
    """
    return matrix - matrix.min(dim=0, keepdims=True).values


def scratch_matrix(matrix):
    """
    Creates the mask for rows and columns which are covering all
    zeros in the matrix.

    Example:
    >>> matrix = torch.Variable(
    >>>    [[15., 15.,  0.],
    >>>     [ 0.,  0., 10.],
    >>>     [ 5.,  5.,  0.]]
    >>> )
    >>> scratch_matrix(matrix)

    >>> (<torch.Tensor: shape=(3, 1), dtype=bool, numpy=
    >>>     array([[False],
    >>>            [ True],
    >>>            [False]])>

    Args:
        matrix:
            The 2D-tensor [rows, columns] of floats to scrarch.

    Returns:
        scratched_rows_mask:
            The 2D-tensor row mask, where `True` values indicates the
            scratched rows and `False` intact rows accordingly.
        scratched_cols_mask:
            The 2D-tensor column mask, where `True` values indicates the
            scratched columns and `False` intact columns accordingly.
    """

    def scratch_row(zeros_mask, scratched_rows_mask, scratched_cols_mask):
        scratched_row_mask = ops.get_row_mask_with_max_zeros(zeros_mask)
        new_scratched_rows_mask = scratched_rows_mask.logical_or(scratched_row_mask)
        new_zeros_mask = zeros_mask.logical_and(
            torch.logical_not(ops.expand_item_mask(scratched_row_mask))
        )
        return new_zeros_mask, new_scratched_rows_mask, scratched_cols_mask

    def scratch_col(zeros_mask, scratched_rows_mask, scratched_cols_mask):
        scratched_col_mask = ops.get_col_mask_with_max_zeros(zeros_mask)
        new_scratched_cols_mask = scratched_cols_mask.logical_or(scratched_col_mask)
        new_zeros_mask = zeros_mask.logical_and(
            torch.logical_not(ops.expand_item_mask(scratched_col_mask))
        )
        return new_zeros_mask, scratched_rows_mask, new_scratched_cols_mask

    def body(zeros_mask, scratched_rows_mask, scratched_cols_mask):
        if ops.count_zeros_in_rows(zeros_mask).max() > ops.count_zeros_in_cols(zeros_mask).max():
            return scratch_row(zeros_mask, scratched_rows_mask, scratched_cols_mask)
        else:
            return scratch_col(zeros_mask, scratched_rows_mask, scratched_cols_mask)

    num_of_rows, num_of_cols = matrix.shape
    zeros_mask = matrix == 0
    scratched_rows_mask = torch.zeros(num_of_rows, 1).bool().to(matrix.device)
    scratched_cols_mask = torch.zeros(1, num_of_cols).bool().to(matrix.device)
    while zeros_mask.any():
        zeros_mask, scratched_rows_mask, scratched_cols_mask = body(
            zeros_mask, scratched_rows_mask, scratched_cols_mask)

    return scratched_rows_mask, scratched_cols_mask


def is_optimal_assignment(
    scratched_rows_mask: torch.Tensor, scratched_cols_mask: torch.Tensor
) -> torch.bool:
    """
    Test if we can achieve the optimal assignment.

    We can achieve the optimal assignment if the combined number of
    scratched columns and rows equals to the matrix dimensions (since
    matrix is square, dimension side does not matter.)

    Example:

        Optimal assignment:
        >>> scratched_rows_mask = torch.tensor(
        >>>    [[False], [True], [False]], torch.bool)
        >>> scratched_cols_mask = torch.tensor(
        >>>    [[True, False, True]])
        >>> is_optimal_assignment(scratched_rows_mask, scratched_cols_mask)

        >>> torch.Tensor(True, shape=(), dtype=bool)

        Not optimal assignment:
        >>> scratched_rows_mask = torch.tensor(
        >>>    [[False], [True], [False]], torch.bool)
        >>> scratched_cols_mask = torch.tensor(
        >>>    [[True, False, True]])
        >>> is_optimal_assignment(scratched_rows_mask, scratched_cols_mask)

        >>> torch.Tensor(False, shape=(), dtype=bool)

    Args:
        scratched_rows_mask:
            The 2D-tensor row mask, where `True` values indicates the
            scratched rows and `False` intact rows accordingly.
        scratched_cols_mask:
            The 2D-tensor column mask, where `True` values indicates the
            scratched columns and `False` intact columns accordingly.

    Returns:
        The boolean tensor, where `True` indicates the optimal assignment
        and `False` otherwise.
    """
    assert scratched_rows_mask.shape[0] == scratched_cols_mask.shape[1]
    n = scratched_rows_mask.shape[0]
    number_of_lines_covering_zeros = \
        scratched_rows_mask.long().sum() + scratched_cols_mask.long().sum()
    return number_of_lines_covering_zeros >= n


def shift_zeros(matrix, scratched_rows_mask, scratched_cols_mask):
    """
    Shifts zeros in not optimal mask.

    Example:

        Optimal assignment:
        >>> matrix = tf.constant(
        >>>    [[ 30., 25., 10.],
        >>>     [ 15., 10., 20.],
        >>>     [ 25., 20., 15.]], tf.float32
        >>> )
        >>> scratched_rows_mask = tf.constant(
        >>>    [[False], [True], [False]], tf.bool)
        >>> scratched_cols_mask = tf.constant(
        >>>    [[False, False, True]])
        >>> shift_zeros(matrix, scratched_rows_mask, scratched_cols_mask)

        >>> (<tf.Tensor:
        >>>       [[10., 10.,  0.],
        >>>        [ 0.,  0., 15.],
        >>>        [ 0.,  0.,  0.]], shape=(3, 3) dtype=float32)>,
        >>> <tf.Tensor:
        >>>       [[False],
        >>>        [ True],
        >>>        [False]], shape=(3, 1), dtype=bool>,
        >>> <tf.Tensor:
        >>>       [[False, False,  True]], shape=(1, 3), dtype=bool>)

    Args:
        matrix:
            The 2D-tensor [rows, columns] of floats with reduced
            values.
        scratched_rows_mask:
            The 2D-tensor row mask, where `True` values indicates the
            scratched rows and `False` intact rows accordingly.
        scratched_cols_mask:
            The 2D-tensor column mask, where `True` values indicates the
            scratched columns and `False` intact columns accordingly.

    Returns:
        matrix:
            The 3D-tensor [rows, columns] of floats with shifted
            zeros.
        scratched_rows_mask:
            The same as input.
        scratched_cols_mask:
            The same as input
    """
    cross_mask = scratched_rows_mask.logical_and(scratched_cols_mask).float()
    inline_mask = torch.logical_or(
        scratched_rows_mask.logical_and(scratched_cols_mask.logical_not()),
        scratched_rows_mask.logical_not().logical_and(scratched_cols_mask)
    ).float()
    outline_mask = torch.logical_not(
            torch.logical_or(scratched_rows_mask, scratched_cols_mask)).float()
    outline_min_value = (
            (1 - outline_mask) * ops.TF_FLOAT32_MAX + matrix * outline_mask).min()

    cross_matrix = matrix * cross_mask + outline_min_value * cross_mask
    inline_matrix = matrix * inline_mask
    outline_matrix = matrix * outline_mask - outline_min_value * outline_mask

    return [
        cross_matrix + inline_matrix + outline_matrix,
        scratched_rows_mask,
        scratched_cols_mask
    ]


def reduce_matrix(matrix):
    """
    Reduce matrix suitable to perform the optimal assignment.

    Example:
        >>> matrix = tf.constant(
        >>>    [[ 30., 25., 10.],
        >>>     [ 15., 10., 20.],
        >>>     [ 25., 20., 15.]], tf.float32
        >>> )
        >>> reduce_matrix(matrix)

        >>> tf.Tensor(
        >>>     [[10. 10.  0.]
        >>>      [ 0.  0. 15.]
        >>>      [ 0.  0.  0.]], shape=(3, 3), dtype=float32)

    Args:
        matrix:
            The 2D-tensor [rows, columns] of floats to reduce.

    Returns:
        A new tensor representing the reduced matrix of the same
        shape as the input tensor.
    """

    def body(matrix):
        new_matrix = reduce_rows(matrix)
        new_matrix = reduce_cols(new_matrix)
        scratched_rows_mask, scratched_cols_mask = scratch_matrix(new_matrix)
        if is_optimal_assignment(scratched_rows_mask, scratched_cols_mask):
            return [
                new_matrix,
                scratched_rows_mask,
                scratched_cols_mask,
            ]
        else:
            return shift_zeros(
                new_matrix, scratched_rows_mask, scratched_cols_mask
            )

    num_of_rows, num_of_cols = matrix.shape
    reduced_matrix = matrix
    scratched_rows_mask = torch.zeros(num_of_rows, 1).bool().to(matrix.device)
    scratched_cols_mask = torch.zeros(1, num_of_cols).bool().to(matrix.device)
    while is_optimal_assignment(
            scratched_rows_mask, scratched_cols_mask).logical_not():
        reduced_matrix, scratched_rows_mask, scratched_cols_mask = body(reduced_matrix)

    return reduced_matrix


def select_optimal_assignment_mask(reduced_matrix):
    """
    Selects the optimal solution based on the reduced matrix.

    Example:
        >>> reduced_matrix = tf.constant(
        >>>     [[10. 10.  0.]
        >>>      [ 0.  0. 15.]
        >>>      [ 0.  0.  0.]], shape=(3, 3), dtype=float32)
        >>> reduce_matrix(matrix)

        >>> tf.Tensor(
        >>>     [[False False  True]
        >>>      [ True False False]
        >>>      [False  True False]], shape=(3, 3), dtype=bool)

        Args:
            matrix:
                The 2D-tensor [rows, columns] of floats representing
                the reduced matrix and used for selecting the optimal
                solution.

        Returns:
            A new tensor representing the optimal assignment mask has
            the same dimension as the input.
    """

    def select_based_on_row(zeros_mask, selection_mask):
        best_row_mask = ops.expand_item_mask(
            ops.get_row_mask_with_min_zeros(zeros_mask)
        )
        best_col_mask = ops.expand_item_mask(
            ops.get_col_mask_with_max_zeros(
                torch.logical_and(best_row_mask, zeros_mask)
            )
        )
        new_selection_mask = torch.logical_or(
            selection_mask, torch.logical_and(best_row_mask, best_col_mask)
        )
        new_mask = torch.logical_and(
            zeros_mask,
            torch.logical_not(torch.logical_or(best_row_mask, best_col_mask)),
        )
        return new_mask, new_selection_mask

    def select_based_on_col(zeros_mask, selection_mask):
        best_col_mask = ops.expand_item_mask(
            ops.get_col_mask_with_min_zeros(zeros_mask)
        )
        best_row_mask = ops.expand_item_mask(
            ops.get_row_mask_with_max_zeros(
                torch.logical_and(best_col_mask, zeros_mask)
            )
        )
        new_selection_mask = torch.logical_or(
            selection_mask, torch.logical_and(best_col_mask, best_row_mask)
        )
        new_mask = torch.logical_and(
            zeros_mask,
            torch.logical_not(torch.logical_or(best_col_mask, best_row_mask)),
        )
        return new_mask, new_selection_mask

    def body(zeros_mask, selection_mask):
        zero_count_in_rows = torch.sum(zeros_mask, dim=1).float()
        zero_count_in_rows[zero_count_in_rows == 0] = ops.TF_FLOAT32_MAX
        min_zero_count_in_rows = zero_count_in_rows.min()
        zero_count_in_cols = torch.sum(zeros_mask, dim=0).float()
        zero_count_in_cols[zero_count_in_cols == 0] = ops.TF_FLOAT32_MAX
        min_zero_count_in_cols = zero_count_in_cols.min()
        if min_zero_count_in_rows < min_zero_count_in_cols:
            return select_based_on_row(zeros_mask, selection_mask)
        else:
            return select_based_on_col(zeros_mask, selection_mask)

    zeros_mask = reduced_matrix == 0
    selection_mask = torch.zeros_like(reduced_matrix).bool()
    while zeros_mask.any():
        zeros_mask, selection_mask = body(zeros_mask, selection_mask)
    return selection_mask


def hungarian_loss(
        y_true, y_pred, pairwise_cost_fn = compute_euclidean_distance):
    """
    Computes the Hungarian loss between `y_true` and `y_pred`.

    For example, if we are detecting 10  bounding boxes on a batch
    of 32 images, the  `y_true` and `y_pred` will represent 32 images
    where each image is represented by 10 bounding boxes and each
    bounding box is represented by 4  (x,y,w,h) coordinates. This
    gives us the final shape of  `y_true` and `y_pred` `(32, 10, 4)`.

    Example:
        >>> y_true = torch.tensor(
        >>>     [
        >>>         [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        >>>         [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        >>>     ]
        >>> )
        >>> y_pred = torch.tensor(
        >>>     [
        >>>         [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
        >>>         [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
        >>>     ]
        >>> )
        >>> hungarian_loss(y_true, y_pred)

        >>> torch.tensor([3.254 3.254], shape=(2,), dtype=float32)

    Args:
        y_true:
            The ground truth values, 3D `Tensor` of shape
            `[batch_size, num_of_entities, num_of_quantifiers]`.
        y_pred:
            The predicted values, 3D `Tensor` with of shape
            `[batch_size, num_of_entities, num_of_quantifiers]`.

    Returns:
        The predicted loss values 1D `tensor` with shape = `[batch_size]`.
    """
    def compute_sample_loss(v_true, v_pred):  # pragma: no cover
        cost = pairwise_cost_fn(v_true, v_pred)
        # We need to reshape the distance matrix by removing the
        # `None` dimension values.
        n = cost.shape[1]
        cost = cost.view(n, n)
        return (cost * select_optimal_assignment_mask(reduce_matrix(cost))).mean()

    losses = torch.zeros(y_true.shape[0]).to(y_true.device)
    # loop over batch elements pairs
    for i, (xt, xp) in enumerate(zip(y_true, y_pred)):
        losses[i] += compute_sample_loss(xt, xp)
    return losses


if __name__ == '__main__':

    def rmse(y_true, y_pred):
        return torch.sqrt(nn.MSELoss()(y_pred, y_true))

    y_true = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    y_pred = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
         [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]])
    h_loss = hungarian_loss(y_true, y_pred, pairwise_cost_fn=compute_euclidean_distance)
    print('function loss', h_loss)