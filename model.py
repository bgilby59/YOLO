import torch
import torchvision

GRID_HEIGHT = 7  # TODO: place these in a separate file somewhere
GRID_WIDTH = 7
BBS_PER_CELL = 2
NUM_CLASSES = 80


# Actual model goes here
class YOLO(torch.nn.Module):
    def __init__(self, batch_size):
        super().__init__()

        self.img_size = (320, 320)
        self.batch_size = batch_size

        # Use pretrained pytorch resnet 50 as backbone
        self.backbone = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.DEFAULT
        )

        # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # change classification layer from resnet34 to not shrink to 1000 classes
        # self.backbone.avgpool = Identity()
        # self.backbone.fc = Identity()

        # Add four convolutional layers and 2 fully-connected layers with randomly initialized weights
        # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        self.fc1 = torch.nn.Linear(
            self.batch_size * 57600, 4096 * 2
        )  # not too steep a drop off in size?
        self.fc2 = torch.nn.Linear(
            4096 * 2,
            self.batch_size
            * GRID_HEIGHT
            * GRID_WIDTH
            * (BBS_PER_CELL * 5 + NUM_CLASSES),
        )

        self.act = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        # TODO: resize images:
        # orig_img_sizes = get_img_sizes(x)  # track orig image sizes of imgs in batch
        # x = resize(x, self.img_size)

        x = self.backbone_forward(x)
        # print("input shape after backbone:", x.shape)

        # Linear activation function for final layer (no activation function), leaky relu for all others
        x = self.conv1(x)
        # print("input shape after first conv layer:", x.shape)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.act(x)
        # print("input shape after last conv layer:", x.shape)

        x = torch.flatten(x)
        # print("input shape after flatten:", x.shape)

        x = self.fc1(x)
        # print("input shape after first fully-connected:", x.shape)
        x = self.dropout(x)
        x = self.act(x)
        x = self.fc2(x)
        # print("output shape after first fully-connected:", x.shape)

        # Reshape tensor:
        # Output should be 7 x 7 x 90 tensor (7x7 grid, 2 bounding boxes per grid cell, COCO has 80 labeled classes -> (7 x 7 x (2*5+80)))
        x = torch.reshape(x, (self.batch_size, 7, 7, 90))

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


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(x):
        return x

    def forward(self, x):
        return x
