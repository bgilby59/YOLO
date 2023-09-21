# Train script

# In paper, trained for about 135 epochs on the training and validation sets from pascal voc 2007 and 2012.
# Batch size = 64, momentum = .9, decay = .0005

"""TODOs"""
# 1. Write mAP calculation function
# 2. Write evaluation function for VOC 2012 val set
# 3. Add data augmentation
# 4. If not satisfied with generalizability, consider adding 2007 VOC data to training (if 2012 isn't a superset of 2007)


import torch
import torchvision
from model import YOLO
from voc_loss import YOLO_LOSS
from torchvision import transforms
import time
from global_constants import IMG_WIDTH, IMG_HEIGHT, VOC_CLASSES
from visualize import *
from utils import voc_transform

import gc

# hyperparameters
batch_size = 1  # TODO: change this to 64, make sure loss function can handle that
momentum = 0.9
decay = 0.0005
lr = 0.000000001
dev = "cuda"


def train(
    model,
    dataloader,
    optimizer,
    epochs=100,
    criterion=YOLO_LOSS(),
    dev="cuda",
    batch_size=1,
):
    start = time.time()
    running_loss = 0
    print("Total number of epochs = ", epochs)
    for i in range(1, epochs + 1):
        running_loss = 0
        average_losses = []
        print("Starting epoch " + str(i))

        iteration = 0
        for imgs, labels in dataloader:
            iteration += 1

            # batch = torch.stack(imgs)

            if torch.cuda.is_available():
                imgs = imgs.to(dev)
                # labels = labels_to_dev(labels, dev='cuda')

            optimizer.zero_grad()

            # to avoid cuda out of memory error
            gc.collect()
            torch.cuda.empty_cache()

            predictions = model(imgs)
            # for i in range(batch_size):
            loss = criterion(predictions, labels)

            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            average_losses.append(loss.item())

        print("Average Loss = ", running_loss / len(dataloader))
        print(
            "elapsed time: ",
            time.time() - start,
            "seconds",
        )
        torch.save(
            model.state_dict(),
            "models/yolo_checkpoint_" + str(running_loss / len(dataloader)) + ".pth",
        )

    return running_loss / len(dataloader)


dataset = torchvision.datasets.voc.VOCDetection(
    "VOC2012", "2012", "train", transforms=voc_transform
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=False  # , collate_fn=coco_collate
)
print(len(dataloader))

model = YOLO(batch_size=batch_size)
model.train()
if torch.cuda.is_available():
    model.to(dev)

optimizer = torch.optim.SGD(
    model.parameters(), lr, momentum=momentum, weight_decay=decay
)

train(model, dataloader, optimizer, batch_size=batch_size)
