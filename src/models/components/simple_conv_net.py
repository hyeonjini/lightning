import torch
import torch.nn as nn


class SimpleConvNet(nn.Module):
    def __init__(
        self,
        hparams: dict,
    ) -> None:

        super(SimpleConvNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, hparams["num_classes"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
