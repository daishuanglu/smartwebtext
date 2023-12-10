"""
Using auxiliary labels in a general image segmentation neural network can provide several benefits
 during training. The main advantage is that it helps with the training convergence and stability 
 of deep neural networks. Here are some benefits:

Intermediate Supervision: By adding auxiliary classifiers at intermediate layers of the network, 
you introduce additional points for supervision. This helps in addressing the vanishing gradient 
problem and enables faster convergence during training.

Gradient Flow: Auxiliary classifiers allow gradients to flow through the network more easily, 
facilitating the learning process for both the main task (segmentation) and auxiliary tasks 
(classification at intermediate layers).

Regularization: The additional classification tasks act as a form of regularization, helping 
prevent overfitting and improving the generalization capability of the model.

Regarding the choice of classes for the auxiliary classification head, it depends on your specific 
segmentation task and dataset. In your case, where the final target is a label image with multiple 
classes like "person," "dog," and "street," you can use any subset of these classes for the 
auxiliary classifier.

Here are some considerations:

Similarity to Main Task: Choose classes that are relevant to the main segmentation task. For 
instance, if your main task is to segment people, you might consider using "person" as one of 
the classes for the auxiliary classification head.

Information Gain: Select classes that provide additional information to the network. If certain 
classes are harder to segment or are crucial for the application, including them in the auxiliary 
classification may be beneficial.

Balancing: Consider balancing the representation of classes in the auxiliary classifier to avoid 
biasing the training towards specific classes.

It's common to experiment with different configurations to see what works best for your specific 
segmentation problem. You can train and evaluate the model with different sets of classes in the 
auxiliary classifier to find the configuration that results in improved performance and training 
stability.
"""

from functools import partial
from typing import Any, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from torchvision.transforms._presets import SemanticSegmentation
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _VOC_CATEGORIES
from torchvision.models._utils import \
    _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from torchvision.models.mobilenetv3 import \
    mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
from torchvision.models.resnet import \
    ResNet, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.fcn import FCNHead


__all__ = [
    "DeepLabV3",
    "DeepLabV3_ResNet50_Weights",
    "DeepLabV3_ResNet101_Weights",
    "DeepLabV3_MobileNet_V3_Large_Weights",
    "deeplabv3_mobilenet_v3_large",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int, atrous_rates: Sequence[int] = (12, 24, 36)) -> None:
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


def _deeplabv3_resnet(
    backbone: ResNet,
    num_classes: int,
    aux: Optional[bool],
) -> DeepLabV3:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = DeepLabHead(2048, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)


_COMMON_META = {
    "categories": _VOC_CATEGORIES,
    "min_size": (1, 1),
    "_docs": """
        These weights were trained on a subset of COCO, using only the 20 categories that are present in the Pascal VOC
        dataset.
    """,
}


class DeepLabV3_ResNet50_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 42004074,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_resnet50",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 66.4,
                    "pixel_acc": 92.4,
                }
            },
            "_ops": 178.722,
            "_file_size": 160.515,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


class DeepLabV3_ResNet101_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 60996202,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#fcn_resnet101",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 67.4,
                    "pixel_acc": 92.4,
                }
            },
            "_ops": 258.743,
            "_file_size": 233.217,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


class DeepLabV3_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 11029328,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_mobilenet_v3_large",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 60.3,
                    "pixel_acc": 91.2,
                }
            },
            "_ops": 10.452,
            "_file_size": 42.301,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


def _deeplabv3_mobilenetv3(
    backbone: MobileNetV3,
    num_classes: int,
    aux: Optional[bool],
) -> DeepLabV3:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    aux_inplanes = backbone[aux_pos].out_channels
    return_layers = {str(out_pos): "out"}
    if aux:
        return_layers[str(aux_pos)] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(aux_inplanes, num_classes) if aux else None
    classifier = DeepLabHead(out_inplanes, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)


@register_model()
@handle_legacy_interface(
    weights=("pretrained", DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
)
def deeplabv3_resnet50(
    *,
    weights: Optional[DeepLabV3_ResNet50_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    .. betastatus:: segmentation module

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The pretrained weights for the
            backbone
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_ResNet50_Weights
        :members:
    """
    weights = DeepLabV3_ResNet50_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param("aux_loss", aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = resnet50(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


@register_model()
@handle_legacy_interface(
    weights=("pretrained", DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", ResNet101_Weights.IMAGENET1K_V1),
)
def deeplabv3_resnet101(
    *,
    weights: Optional[DeepLabV3_ResNet101_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet101_Weights] = ResNet101_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    .. betastatus:: segmentation module

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet101_Weights`, optional): The pretrained weights for the
            backbone
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_ResNet101_Weights
        :members:
    """
    weights = DeepLabV3_ResNet101_Weights.verify(weights)
    weights_backbone = ResNet101_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param("aux_loss", aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = resnet101(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


@register_model()
@handle_legacy_interface(
    weights=("pretrained", DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", MobileNet_V3_Large_Weights.IMAGENET1K_V1),
)
def deeplabv3_mobilenet_v3_large(
    *,
    weights: Optional[DeepLabV3_MobileNet_V3_Large_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[MobileNet_V3_Large_Weights] = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a MobileNetV3-Large backbone.

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The pretrained weights
            for the backbone
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights
        :members:
    """
    weights = DeepLabV3_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param("aux_loss", aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = mobilenet_v3_large(weights=weights_backbone, dilated=True)
    model = _deeplabv3_mobilenetv3(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model
