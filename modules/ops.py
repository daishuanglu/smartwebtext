"""A collection of supporting operations."""
import torch

#TF_FLOAT32_MAX = 3.4028235e+38
TF_FLOAT32_MAX = 1e38

def count_zeros_in_rows(zeros_mask: torch.Tensor) -> torch.Tensor:
    """
    Counts a number of zero-values in each row using a zeros'-mask.

    Zeros' mask highlights the matrix cells with zero values.

    Example:
        >>> zeros_mask = torch.tensor(
        >>>     [[ True, False, False],
        >>>      [ True, True, False],
        >>>      [ True, True, True]]
        >>> )
        >>> count_zeros_in_rows(zeros_mask)

        >>>  torch.tensor(
        >>>    [[1.]
        >>>    [2.]
        >>>    [3.]]).float()
    Args:
        zeros_mask:
            A 2D boolean tensor mask [rows, columns] for highlighting
            zero cells. The `True` value indicates that the cell in
            the original matrix is 0 and `False` otherwise.

    Returns:
        A 2D tensor representing zeros count in each row.
    """
    return torch.sum(zeros_mask, dim=1, keepdim=True)


def count_zeros_in_cols(zeros_mask: torch.tensor) -> torch.Tensor:
    """
    Counts a number of zero-values in each column using a zeros'-mask.

    Zeros' mask highlights the matrix cells with zero values.

    Example:
        >>> zeros_mask = torch.tensor(
        >>>     [[ True, False, False],
        >>>      [ True, True, False],
        >>>      [ True, True, True]]
        >>> )
        >>> count_zeros_in_cols(zeros_mask)

        >>> torch.Tensor([[3. 2. 1.]], shape=(1, 3), dtype=float32)

    Args:
        zeros_mask:
            A 2D boolean tensor mask [rows, columns] for highlighting
            zero cells. The `True` value indicates that the cell in
            the original matrix is 0 and `False` otherwise.

    Returns:
        A 1D tensor representing zeros count in each column.
    """
    return torch.sum(zeros_mask, dim=0, keepdim=True)


def get_row_mask_with_min_zeros(zeros_mask: torch.Tensor) -> torch.Tensor:
    """
    Returns a row mask with minimum number of zeros.

    Note, rows containing all zeros are excluded from the computation
    of this mask.

    Example:

        1. Example with zeros in all rows.
        >>> zeros_mask = tf.constant(
        >>>     [[ True, False, False],
        >>>      [ True, True, False],
        >>>      [ True, True, True]]
        >>> )
        >>> get_row_mask_with_min_zeros(zeros_mask)

        >>> tf.Tensor(
        >>>     [[ True]
        >>>      [False]
        >>>      [False]], shape=(3, 1), dtype=bool)

        2. Example without zeros in one row.
        >>> zeros_mask = tf.constant(
        >>>     [[ False, False, False],
        >>>      [ True, True, False],
        >>>      [ True, True, True]]
        >>> )
        >>> get_row_mask_with_min_zeros(zeros_mask)

        >>> tf.Tensor(
        >>>     [[False]
        >>>      [ True]
        >>>      [False]], shape=(3, 1), dtype=bool)

    Args:
        zeros_mask:
            A 2D boolean tensor mask [rows, columns] for highlighting
            zero cells. The `True` value indicates that the cell in
            the original matrix is 0 and `False` otherwise.

    Returns:
        A 2D tensor represents a row mask with a minimum number of zeros.
    """
    counts = count_zeros_in_rows(zeros_mask).float()
    # In this step, we are replacing all zero counts with max floating
    # value, since we need this to eliminate rows filled with all zeros.
    counts[counts == 0] = TF_FLOAT32_MAX
    return torch.argsort(torch.argsort(counts, dim=0, descending=False), dim=0) == 0


def get_row_mask_with_max_zeros(zeros_mask: torch.Tensor) -> torch.Tensor:
    """
    Returns a row mask with maximum number of zeros.

    Example:
        >>> zeros_mask = torch.tensor(
        >>>     [[ True, False, False],
        >>>      [ True, True, False],
        >>>      [ True, True, True]]
        >>> )
        >>> get_row_mask_with_max_zeros(zeros_mask)

        >>> torch.Tensor(
        >>>     [[False]
        >>>      [False]
        >>>      [ True]], shape=(3, 1), dtype=bool)
    Args:
        zeros_mask:
            A 2D boolean tensor mask [rows, columns] for highlighting
            zero cells. The `True` value indicates that the cell in
            the original matrix is 0 and `False` otherwise.

    Returns:
        A 2D tensor represents a row mask with a maximum number of zeros.
    """
    counts = count_zeros_in_rows(zeros_mask)
    return torch.argsort(torch.argsort(counts, dim=0, descending=True), dim=0) == 0


def get_col_mask_with_min_zeros(zeros_mask) -> torch.Tensor:
    """
    Returns a column mask with minimum number of zeros.

    Example:

        1. Example with zeros in all columns.
        >>> zeros_mask = torch.tensor(
        >>>     [[ True, False, False],
        >>>      [ True, True, False],
        >>>      [ True, True, True]]
        >>> )
        >>> get_col_mask_with_min_zeros(zeros_mask)

        >>> torch.Tensor([[False False  True]], shape=(1, 3), dtype=bool)

        2. Example without zeros in one column.
        >>> zeros_mask = torch.tensor(
        >>>     [[ True, False, False],
        >>>      [ True,  True, False],
        >>>      [ True,  True, False]]
        >>> )
        >>> get_col_mask_with_min_zeros(zeros_mask)

        >>> torch.Tensor([[False True  False]], shape=(1, 3), dtype=bool)

    Args:
        zeros_mask:
            A 2D boolean tensor mask [rows, columns] for highlighting
            zero cells. The `True` value indicates that the cell in
            the original matrix is 0 and `False` otherwise.

    Returns:
        A 2D tensor represents a column mask with a minimum number of zeros.
    """
    counts = count_zeros_in_cols(zeros_mask).float()
    # In this step, we are replacing all zero counts with max floating
    # value, since we need this to eliminate columns filled with all zeros.
    counts[counts == 0] = TF_FLOAT32_MAX
    return torch.argsort(torch.argsort(counts, dim=1, descending=False), dim=1) == 0


def get_col_mask_with_max_zeros(zeros_mask: torch.Tensor) -> torch.Tensor:
    """
    Returns a column mask with maximum number of zeros.

    Example:
        >>> zeros_mask = tf.constant(
        >>>     [[ True, False, False],
        >>>      [ True, True, False],
        >>>      [ True, True, True]]
        >>> )
        >>> get_col_mask_with_max_zeros(zeros_mask)

        >>> tf.Tensor([[ True False False]], shape=(1, 3), dtype=bool)
    Args:
        zeros_mask:
            A 2D boolean tensor mask [rows, columns] for highlighting
            zero cells. The `True` value indicates that the cell in
            the original matrix is 0 and `False` otherwise.

    Returns:
        A 2D tensor represents a column mask with a maximum number of zeros.
    """
    counts = count_zeros_in_cols(zeros_mask)
    return torch.argsort(torch.argsort(counts, dim=1, descending=True), dim=1) == 0


def expand_item_mask(item_mask: torch.Tensor) -> torch.Tensor:
    """
    Expands row or column mask to for square shape

    Example:

        1. This example of expanding row mask.
        >>> row_mask = torch.tensor(
        >>>     [[ True],
        >>>      [False],
        >>>      [False]]]
        >>> )
        >>> expand_item_mask(row_mask)

        >>> torch.Tensor(
        >>>     [[ True  True  True]
        >>>      [False False False]
        >>>      [False False False]], shape=(3, 3), dtype=bool)

        2. This example of expanding column mask.
        >>> col_mask = tf.constant(
        >>>     [[ True, False, False]]]
        >>> )
        >>> expand_item_mask(col_mask)

        >>> torch.Tensor(
        >>>     [[ True False False]
        >>>      [ True False False]
        >>>      [ True False False]], shape=(3, 3), dtype=bool)

    Args:
        item_mask:
            A 2D boolean tensor mask [rows, 1] | [1, columns] for
            selected row or column.

    Returns:
        A 2D tensor [rows, columns] for the expanded row or column mask.
    """
    row_number, col_number = item_mask.shape
    ref_tensor = torch.zeros(col_number, col_number).to(item_mask.device) \
        if row_number == 1 else torch.zeros(row_number, row_number).to(item_mask.device)
    return (item_mask + ref_tensor).bool()


def pairwise_mask_iou(mask1, mask2, smooth=1e-10) -> torch.Tensor:
    """
    Inputs:
    mask1: NxHxW torch.float32. Consists of [0, 1]
    mask2: NxHxW torch.float32. Consists of [0, 1]
    Outputs:
    ret: NxM torch.float32. Consists of [0 - 1]
    """
    N, H, W = mask1.shape
    M, H, W = mask2.shape
    mask1 = mask1.view(N, H*W)
    mask2 = mask2.view(M, H*W)
    intersection = torch.matmul(mask1, mask2.t())
    area1 = mask1.sum(dim=1).view(1, -1)
    area2 = mask2.sum(dim=1).view(1, -1)
    union = (area1.t() + area2) - intersection
    ret = (intersection + smooth) / (union + smooth)
    return ret


def pairwise_mask_iou_dice(mask1, mask2, smooth=1e-8) -> torch.Tensor:
    """
    Inputs:
    mask1: NxHxW torch.float32. Consists of [0, 1]
    mask2: NxHxW torch.float32. Consists of [0, 1]
    Outputs:
    ret: NxM torch.float32. Consists of [0 - 1]
    """
    iou = pairwise_mask_iou(mask1, mask2, smooth)
    ret = 1 - iou
    return ret


def pairwise_bce(p, q, smooth=1e-8) -> torch.Tensor:
    """
    Inputs:
    p: MxHxW torch.float32. Consists of [0, 1]
    q: NxHxW torch.float32. Consists of [0, 1]
    Outputs:
    ret: NxM torch.float32. Consists of [0 - 1]
    """
    N, H, W = q.shape
    M, H, W = p.shape
    q_flat = q.view(N, H * W)
    p_flat = p.view(M, H * W)
    log_q_flat = torch.log(q_flat)
    log_q_neg_flat = torch.log(1 - q_flat)
    ent = - (torch.matmul(log_q_flat, p_flat.t())
             + torch.matmul(log_q_neg_flat, 1 - p_flat.t()))
    ent /= (H * W)
    return ent


def pairwise_cross_entropy(p, q, ignore_indices=[]) -> torch.Tensor:
    """
    Inputs:
    p: M * n_classes torch.float32. Consists of [0, 1]
    q: N * n_classes torch.float32. Consists of [0, 1]
    Outputs:
    ret: N * M torch.float32. Consists of [0 - 1]
    """
    n_cls = p.shape[-1]
    p = p.view(-1, n_cls)
    q = q.view(-1, n_cls)
    log_q = torch.log(q)
    p_ignored = p.clone()
    if len(ignore_indices) > 0:
        p_ignored[:, ignore_indices] = 0.0
    ent = - torch.matmul(log_q, p_ignored.t())
    ent /= (n_cls - len(ignore_indices))
    return ent


if __name__ == '__main__':
    zeros_mask = torch.tensor(
        [[True, False, False],[True, True, False], [True, True, True]])
    print(zeros_mask)
    print('count zeros in rows')
    print(count_zeros_in_rows(zeros_mask))
    print('count zeros in cols')
    print(count_zeros_in_cols(zeros_mask))
    print('Example with zero counts in all rows')
    zeros_mask = torch.tensor([[True, False, False], [True, True, False], [True, True, True]])
    print(get_row_mask_with_min_zeros(zeros_mask))
    print('Example without zero counts in one of the rows.')
    zeros_mask = torch.tensor([[False, False, False], [True, True, False], [True, True, True]])
    print(get_row_mask_with_min_zeros(zeros_mask))
    print('Row mask with maximum number of zeros.')
    print('input:')
    zeros_mask = torch.tensor([[True, False, False], [True, True, False], [True, True, True]])
    print(zeros_mask)
    print(get_row_mask_with_max_zeros(zeros_mask))
    row_mask = torch.tensor([[True],[False],[False]])
    print(expand_item_mask(row_mask))
    col_mask = torch.tensor([[True, False, False]])
    print(expand_item_mask(col_mask))
