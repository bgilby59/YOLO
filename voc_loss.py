import math
import numpy as np
import torch

from utils import iou
from visualize import visualize_predictions
from global_constants import (
    GRID_WIDTH,
    GRID_HEIGHT,
    IMG_WIDTH,
    IMG_HEIGHT,
    BBS_PER_CELL,
    NUM_CLASSES,
)


CELL_HEIGHT = IMG_HEIGHT / GRID_HEIGHT
CELL_WIDTH = IMG_WIDTH / GRID_WIDTH

LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5


class YOLO_LOSS(torch.nn.Module):
    def __init__(self):
        super(YOLO_LOSS, self).__init__()

    def forward(self, pred, gt):
        xy_loss, wh_loss, neg_wh_loss, conf_loss, class_loss = [0.0] * 5

        pred = pred.to("cpu")  # TODO: Will this be very slow???
        pred = torch.squeeze(pred)

        responsible_bboxes = []

        noobj_cells = [0] * GRID_HEIGHT * GRID_WIDTH
        for obj in gt:
            row, col = get_obj_cell(obj)
            noobj_cells[row * GRID_WIDTH + col] = 1  # object found in this cell
            bbox_id = determine_responsible_bbox_id(
                row,
                col,
                torch.reshape(
                    pred[row, col, 0 : BBS_PER_CELL * 5], (-1, 5)
                ),  # reshape into list of bboxes
                obj["bbox"],
            )
            responsible_bboxes.append((row, col, bbox_id))

            bbox = output_box_to_pred(
                pred[row, col, bbox_id * 5 : bbox_id * 5 + 4], row, col
            )
            pred_x = bbox[0]

            pred_y = bbox[1]
            pred_w = bbox[2]
            pred_h = bbox[3]
            label_x = obj["bbox"][0]

            label_y = obj["bbox"][1]
            label_w = obj["bbox"][2]
            label_h = obj["bbox"][3]

            xy_loss += LAMBDA_COORD * (
                sum_squared_error(label_x, pred_x) + sum_squared_error(label_y, pred_y)
            )

            # loss part 2
            wh_loss += LAMBDA_COORD * (
                sum_squared_error(math.sqrt(label_w), math.sqrt(pred_w))
                + sum_squared_error(math.sqrt(label_h), math.sqrt(pred_h))
            )
            if (
                pred_w < 0
            ):  # I'll add this because negative height and width is a big problem right now, see if can remove later
                neg_wh_loss += pred_w
            if pred_h < 0:
                neg_wh_loss += pred_h

            # loss part 3
            conf_loss = sum_squared_error(1, pred[row, col, bbox_id * 5 + 4])

            # loss part 5 TODO: What if multiple objects are in the same cell? Multiple loss additions? That's the way it works now
            label_class = obj["category_id"]
            for class_id in range(NUM_CLASSES):
                pred_class_conf = pred[row, col, BBS_PER_CELL * 5 + class_id]
                if class_id == label_class:
                    class_loss += sum_squared_error(1, pred_class_conf)
                else:
                    class_loss += sum_squared_error(0, pred_class_conf)

        # for noobj_bbox in noobj_bboxes:
        for idx, cell in enumerate(noobj_cells):
            row = (idx - (idx % GRID_WIDTH)) // GRID_WIDTH  # TODO: double check these
            col = idx % GRID_WIDTH

            # loss part 4
            if cell == 0:
                for i in range(BBS_PER_CELL):
                    pred_c = pred[row, col, i * 5 + 4]
                    conf_loss += LAMBDA_NOOBJ * sum_squared_error(0, pred_c)

        # visualize_predictions(pred, responsible_bboxes, vis=False)

        total_loss = xy_loss + wh_loss + conf_loss + class_loss

        return total_loss


def sum_squared_error(gt, pred):
    return (gt - pred) ** 2


def determine_responsible_bbox_id(row, col, bboxes, obj_bbox):
    max_iou = 0
    responsible_bbox_id = 0  # if no bboxes have overlap, just punish the first box?
    for idx, bbox in enumerate(bboxes):
        pred_bbox = output_box_to_pred(bbox, row, col)
        if iou(pred_bbox, obj_bbox) > max_iou:
            responsible_bbox_id = idx
            max_iou = iou(pred_bbox, obj_bbox)
    return responsible_bbox_id


def get_obj_cell(obj):
    obj_center_x, obj_center_y = get_center_of_object(obj)

    cell_h = IMG_HEIGHT / GRID_HEIGHT
    cell_w = IMG_WIDTH / GRID_WIDTH

    col = int(torch.div(obj_center_x, cell_w, rounding_mode="floor"))
    row = int(torch.div(obj_center_y, cell_h, rounding_mode="floor"))

    return row, col


def get_center_of_object(obj):
    bbox = obj["bbox"]
    return (
        bbox[0] + (bbox[2] / 2),
        bbox[1] + (bbox[3] / 2),
    )


# Bounding boxes are outputted as values between [0,1] by model, need to convert back to image space
def output_box_to_pred(bbox, row, col):
    grid_x = col * CELL_WIDTH
    grid_y = row * CELL_HEIGHT
    pred_w = bbox[2] * IMG_WIDTH
    pred_h = bbox[3] * IMG_HEIGHT
    pred_center_x = grid_x + bbox[0] * CELL_WIDTH
    pred_center_y = grid_y + bbox[1] * CELL_HEIGHT
    # go from center of bbox to up left coordinate
    pred_x = pred_center_x - (pred_w / 2)
    pred_y = pred_center_y - (pred_h / 2)

    return [pred_x, pred_y, pred_w, pred_h]
