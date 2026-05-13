from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNet


def build_product_retinanet(
    num_classes: int = 1,
    pretrained_backbone: bool = False,
    min_size: int = 512,
    max_size: int = 768,
):
    """
    Build the Stage 2 product detector.

    The paper uses RetinaNet for product detection. This project treats
    products as foreground boxes; num_classes is the number of foreground
    product categories, with labels expected to be zero-based for RetinaNet.
    """
    if pretrained_backbone:
        # Torchvision may download weights the first time this is used.
        model = retinanet_resnet50_fpn_v2(
            weights=None,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
        )
    else:
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=None,
            trainable_layers=5,
        )
        model = RetinaNet(
            backbone,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
        )

    return model
