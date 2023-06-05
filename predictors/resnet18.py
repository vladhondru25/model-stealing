import torch
import torch.nn as nn
from torchvision import models


# Pre-defined ResNet
class ResNet18(nn.Module):
    def __init__(self, name, n_outputs):
        super().__init__()
        self.name = name
        self.n_outputs = n_outputs

        self.model = models.resnet18(pretrained = False)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.n_outputs)
        input_size = 224

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs
    
# Cusotm ResNet
class ResidualBlock(nn.Module):
    def __init__(self, in_c=64, out_c=64, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels=in_c,  out_channels=out_c, kernel_size=3, padding=1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_c))
        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1, stride=1,      bias=False),
                        nn.BatchNorm2d(out_c))
        
        self.shortcut = nn.Identity()
        if stride > 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                                nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=stride, bias=False),
                                nn.BatchNorm2d(out_c))
            
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        z1 = self.activation(self.conv1(x))
        z2 = self.conv2(z1) + self.shortcut(x)
        return self.activation(z2)

class ResNet18Custom(nn.Module):
    def __init__(self, name = None, n_outputs=10, return_feature_domain=False):
        super(ResNet18Custom, self).__init__()

        self.return_feature_domain = return_feature_domain
        
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True))
        
        # Used to keep track of the previous numbers of channels
        self.prev_channels = 64
        
        self.res_layer1 = self._create_res_layer(out_c=128, stride=1, no_blocks=2)
        self.res_layer2 = self._create_res_layer(out_c=256, stride=2, no_blocks=2)
        self.res_layer3 = self._create_res_layer(out_c=512, stride=2, no_blocks=2)
        self.res_layer4 = self._create_res_layer(out_c=512, stride=2, no_blocks=2)
        
        self.pooling = nn.AvgPool2d(kernel_size=4)
        
        self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_features=512, out_features=n_outputs, bias=True))
        
        self.relu = nn.ReLU(inplace=True)
        
    def _create_res_layer(self, out_c, stride, no_blocks):
        interim_layers = []
        
        strides = [stride] + [1]*(no_blocks-1)
        for s in strides:
            interim_layers.append(ResidualBlock(in_c=self.prev_channels, out_c=out_c, stride=s))
            self.prev_channels = out_c
        
        return nn.Sequential(*interim_layers)
        
    def forward(self, x):
        
        x = self.conv1(x)
        
        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)
        x = self.res_layer4(x)

        if self.return_feature_domain:
            return self.fc(self.pooling(x)), x
        
        x = self.pooling(x)
        
        return self.fc(x)
    