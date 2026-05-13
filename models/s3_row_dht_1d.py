import torch
import torch.nn as nn
import torch.nn.functional as F


class RowDHT1D(nn.Module):
    """
    Shelf-specific row detector.
    """

    def __init__(self, out_height=512):
        super().__init__()
        self.out_height = out_height

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.row_head = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        """
        x: B x 3 x H x W
        returns: B x out_height
        """
        feat = self.backbone(x)          # B x C x H' x W'

        feat_1d = feat.mean(dim=3)       # B x C x H'

        logits = self.row_head(feat_1d)  # B x 1 x H'
        logits = logits.squeeze(1)       # B x H'

        logits = F.interpolate(
            logits.unsqueeze(1),
            size=self.out_height,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

        return logits