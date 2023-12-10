import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            input_soft: torch.Tensor,
            target_one_hot: torch.Tensor) -> torch.Tensor:
        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)


def focal_loss(outputs, targets, alpha=1.0, gamma=2.0):
    # Input: shape N, (N,C), or(N, C ,d1,d2,...)
    # targets: If containing class indices, shape = (), N, (N, d1,d2,..dK)
    # with K>=1 in the case of K-dimensional loss where each value should be between
    # [0, C), If containing class probabilities, same shape as the input and each
    # value should be between [0,1].
    ce_loss = torch.nn.functional.cross_entropy(
        outputs,
        targets,
        reduction='none')  # important to add reduction='none' to keep per-batch-item loss
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()  # mean over the batch
    return focal_loss


def focal_loss_stable(y_real, y_pred, eps = 1e-8, gamma = 0):
    probabilities = torch.clamp(torch.sigmoid(y_pred), min=eps, max=1-eps)
    return (1 - probabilities)**gamma * (
        y_pred - y_real * y_pred + torch.log(1 + torch.exp(-y_pred)))


##
# version 1: use torch.autograd
class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                torch.nn.functional.softplus(logits, -1, 50),
                logits - torch.nn.functional.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + torch.nn.functional.softplus(logits, -1, 50),
                -torch.nn.functional.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


if __name__=='__main__':
    # end class MulticlassDiceLoss
    criterion = MulticlassDiceLoss(num_classes=3)
    y = torch.randn(10, 3, 4, 4)
    targets = torch.randint(0, 3, (10, 4, 4))
    print(y.shape)
    loss = criterion(y, targets)
    print('dice loss', loss)
    floss = focal_loss(y, targets, alpha=0.5, gamma=2)
    print('focal loss', floss)