
from turtle import forward
import torch
import torch.nn as nn


class VGG(nn.Module):

    def __init__(
        self, 
        hparams: dict
    ) -> None:

        super(VGG, self).__init__()
        self.features = self._make_layers(hparams["feature"])
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, hparams["num_classes"]),
        )

        if hparams["init_weights"]:
            self._initialize_weights()
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg):
        layers = []
        in_channel = 3

        for layer in cfg:
            for _ in range(layer[0]): # layer[0] : repeat
                if layer[1] == "Conv":
                    args = [in_channel] + list(layer[2])
                    layers += [getattr(__import__("src.models.components.modules", fromlist=[""]), layer[1])(*args)]
                    in_channel = args[1]

                elif layer[1] == "MaxPool":
                    args = list(layer[2])
                    layers += [nn.MaxPool2d(2, 2)]
            
        return nn.Sequential(*layers)