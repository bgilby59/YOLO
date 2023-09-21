import torch
import torchvision
from global_constants import GRID_WIDTH, GRID_HEIGHT, BBS_PER_CELL, NUM_CLASSES


# Actual model goes here
class YOLO(torch.nn.Module):
    def __init__(self, batch_size):
        super().__init__()

        self.batch_size = batch_size

        # Use pretrained pytorch resnet 34 as backbone
        self.backbone = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.DEFAULT
        )

        # Add four convolutional layers and 2 fully-connected layers with randomly initialized weights
        self.conv1 = torch.nn.Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv2 = torch.nn.Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv3 = torch.nn.Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv4 = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        self.fc1 = torch.nn.Linear(self.batch_size * 20736, 4096 * 2)
        self.fc2 = torch.nn.Linear(
            4096 * 2,
            self.batch_size
            * GRID_HEIGHT
            * GRID_WIDTH
            * (BBS_PER_CELL * 5 + NUM_CLASSES),
        )

        self.act = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.backbone_forward(x)

        # Sigmoid activation function for final layer, leaky relu for all others
        x = self.conv1(x)

        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.act(x)

        x = torch.flatten(x)

        x = self.fc1(x)

        x = self.dropout(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.sig(x)

        # Reshape tensor:
        # Output should be 7 x 7 x 20 tensor (7x7 grid, 2 bounding boxes per grid cell, PASCAL VOC has 20 labeled classes -> (7 x 7 x (2*5+20)))
        x = torch.reshape(
            x,
            (self.batch_size, GRID_WIDTH, GRID_HEIGHT, BBS_PER_CELL * 5 + NUM_CLASSES),
        )

        return x

    def backbone_forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x
