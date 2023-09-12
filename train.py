# Train script

# In paper, trained for about 135 epochs on the training and validation sets from pascal voc 2007 and 2012.
# Batch size = 64, momentum = .9, decay = .0005
# use dropout and data augmentation
"""NOTES"""
# Add data augmentation!
# How to perform data augmentation while preserving mapping of input data to labels??? Need to augment labels too?
# Answer: perform data augmentation that does not affect labels (e.g. contrast, brightness change, etc.)

import torch
import torchvision
from model import YOLO
from loss import yolo_loss
from torchvision import transforms
import numpy as np
import time

import gc

from PIL import Image

IMG_HEIGHT = 480
IMG_WIDTH = 480

# hyperparameters
batch_size = 1  # TODO: change this to 64, make sure loss function can handle that
momentum = 0.9
decay = 0.0005

# transform = transforms.Compose(
#     [
#         transforms.Resize((320, 320)),  # TODO: make sure bboxes are resized too!
#         transforms.ToTensor(),
#     ]
# )  # TODO: add data augmentation. Are these transformations applied each batch?
# target_transform = transforms.Compose([transforms.Resize((320, 320))])

lr = 0.001
dev = "cuda"


def transform(img, target):
    # transform bboxes to match image resizing
    width_scaling = IMG_WIDTH / img.width
    height_scaling = IMG_HEIGHT / img.height
    for obj in target:
        # print(obj)
        bbox = obj["bbox"]
        bbox[0] *= width_scaling  # x
        bbox[1] *= height_scaling  # y
        bbox[2] *= width_scaling  # width
        bbox[3] *= height_scaling  # height

    # resize image
    img_transforms = transforms.Compose(
        [
            transforms.Resize(
                (IMG_HEIGHT, IMG_WIDTH)
            ),  # TODO: make sure bboxes are resized too!
            transforms.ToTensor(),
        ]
    )
    img = img_transforms(img)

    return img, target


def coco_collate(batch):
    return tuple(zip(*batch))


def labels_to_dev(labels, dev="cuda"):
    return False


def train(
    model,
    dataloader,
    optimizer,
    epochs=30,
    criterion=yolo_loss,
    dev="cuda",
    batch_size=1,
):
    start = time.time()
    for i in range(epochs):
        running_loss = 0
        print("Starting epoch ", i)

        iteration = 1
        for imgs, labels in dataloader:
            batch = torch.stack(imgs)
            # print(labels)
            if torch.cuda.is_available():
                batch = batch.to(dev)
                # labels = labels_to_dev(labels, dev='cuda')

            optimizer.zero_grad()

            # print(type(imgs))
            # print(imgs.shape)

            # to avoid cuda out of memory error
            # print("emptying cache...")
            # gc.collect()
            # torch.cuda.empty_cache()

            predictions = model(batch)
            loss = 0
            for i in range(batch_size):
                loss += criterion(labels[i], predictions[i])
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if iteration % 50 == 1:
                print("iterations * batch size = ", iteration * batch_size)
                print("elapsed time: ", time.time() - start, "seconds")
                print("Running loss = ", running_loss)
                print("Average Loss = ", running_loss / iteration)
            iteration += 1

        print("Average Loss = ", running_loss / len(dataloader))


dataset = torchvision.datasets.coco.CocoDetection(
    "/home/benjamin-gilby/yolo_impl/COCO/train2017",
    "/home/benjamin-gilby/yolo_impl/COCO/annotations/instances_train2017.json",
    transforms=transform,
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, collate_fn=coco_collate
)

model = YOLO(batch_size=batch_size)
model.train()
if torch.cuda.is_available():
    model.to(dev)

optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)

train(model, dataloader, optimizer, batch_size=batch_size)

torch.save(model.state_dict(), "models/mobilenet_qat.pth")
