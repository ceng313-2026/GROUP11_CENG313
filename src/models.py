"""Model definitions.

Owner: Can

- MIPNet (B2): pretrained 2D CNN on maximum intensity projections from 3 views, 14 outputs
- B1-DICOM: LightGBM per label on DICOM header features (src.data.header_features), our Kaggle submission
- Planned: main model, 3D CNN with an auxiliary localizer head
"""
import lightgbm as lgb
import pandas as pd
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


# Same settings as B1-LGBM in notebooks/03_baselines.ipynb: small trees and a high minimum leaf size,
# because some labels have few positives.
B1_LGBM_PARAMS = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=50,
                      subsample=0.8, subsample_freq=1, random_state=0, verbose=-1)


def fit_b1_dicom(features, labels):
    """B1-DICOM: one LightGBM per label on the header features (columns of src.data.HEADER_FEATURES).

    B1 uses age, sex and modality from train.csv, which does not exist for the hidden test series.
    B1-DICOM reads the same information, and a few tags that tell the MR types apart, from the header.
    Missing tags are NaN, LightGBM learns which side of a split they go to.
    Returns {label: model}.
    """
    return {col: lgb.LGBMClassifier(**B1_LGBM_PARAMS).fit(features, labels[col]) for col in LABEL_COLS}


def predict_b1_dicom(models, features):
    """Probabilities of fit_b1_dicom() models, a DataFrame with LABEL_COLS columns."""
    return pd.DataFrame({col: models[col].predict_proba(features)[:, 1] for col in LABEL_COLS}, index=features.index)
