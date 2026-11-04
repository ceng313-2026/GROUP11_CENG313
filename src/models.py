"""Model definitions.

Owner: Can

- MIPNet (B2): pretrained 2D CNN on maximum intensity projections from 3 views, 14 outputs
- Planned: main model, 3D CNN with an auxiliary localizer head
"""
import torch
import torch.nn as nn
import torchvision

from src.utils import LABEL_COLS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MIPNet(nn.Module):
    """B2: one ResNet18 looks at the axial, coronal and sagittal MIP; the 3 feature vectors are joined
    and mapped to 14 logits (LABEL_COLS order).

    The same ResNet (shared weights) is used for all 3 views: the views look alike (vessels on a dark
    background), and sharing keeps the model small for about 3,500 training series.
    Inputs are uint8 tensors (batch, 3, H, W) from src.data.make_mips(); the 3 channels take the place
    of the RGB channels the ResNet was pretrained on.
    """

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = torchvision.models.resnet18(weights=weights)
        n_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(3 * n_features, len(LABEL_COLS)))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, axial, coronal, sagittal):
        features = [self.backbone((view.float() / 255 - self.mean) / self.std) for view in (axial, coronal, sagittal)]
        return self.head(torch.cat(features, dim=1))
