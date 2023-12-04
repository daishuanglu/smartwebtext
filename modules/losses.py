import torch
import torch.nn as nn
from modules import ops


class MulticlassDiceLoss(nn.Module):
    """Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets, dim=-1, smooth=1e-6):
        """
        The "reduction" argument is ignored. This method computes the dice
        loss for all classes and provides an overall weighted loss.
        """
        probabilities = logits
        if dim is not None:
            probabilities = nn.Softmax(dim=dim)(logits)
        # end if
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
        # Multiply one-hot encoded ground truth labels with the probabilities to get the
        # prredicted probability for the actual class.
        intersection = targets_one_hot * probabilities
        mod_a = intersection.sum()
        mod_b = targets.numel()
        dice_coefficient = 2. * intersection / (mod_a + mod_b + smooth)
        dice_loss = -dice_coefficient.log()
        return dice_loss


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


class EMA():
    def __init__(self, alpha):
       super().__init__()
       self.alpha = alpha

    def update_average(self, old, new):
       if old is None:
           return new
       return old * self.alpha + (1 - self.alpha) * new

    def flow(self, student_model, teacher_model):
        for student_params, teacher_params in zip(student_model.parameters(), teacher_model.parameters()):
            old_weight, up_weight = teacher_params.data, student_params.data
            teacher_params.data = self.update_average(old_weight, up_weight)
        return teacher_model


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