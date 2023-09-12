# Multi-part loss function goes here ＊＊＊DONEかもしれない＊＊＊

"""***NOTES***"""
# *** If stuff seems broken, might be useful to make unit tests for utils (iou, etc.)
# *** bbox is “responsible” for the ground truth box MEANS it has the highest IOU of any bbox in that grid cell
# How to determine object is in a grid cell? SEE BELOW
"""
Our system divides the input image into an S x S grid.
If the center of an object falls into a grid cell, that grid cell
is responsible for detecting that object.
"""
# SOLUTION for collecting all bboxes that are not responsible for any objects
# create list, noobj_bboxes
# put all bboxes in list
# when running determine responsible bbox, remove the returned box from list if in list
# at the end, only bboxes not responsible for any objects will be left, punish their confidence scores!

# !!! Lots of people are saying 1noobj_ij is if no obj is in the cell, but doesn't j pertain to bbox???
# I am acting as though it means no obj is in cell now

# !!! I think should not have to loop through rows and cols! I don't think it affects correctness, but this is slow! After pushing to github to save progress, change this!

import math
import numpy as np
import torch

GRID_HEIGHT = 7  # TODO: place these in a separate file somewhere
GRID_WIDTH = 7
BBS_PER_CELL = 2
NUM_CLASSES = 80
IMG_HEIGHT = 480  # TODO: what size should the image be?
IMG_WIDTH = 480

LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5


# TODO: there is probably a way to make this faster if training is too slow...
def yolo_loss(gt, pred):  # gt = list of objs, need to get which obj is in cell
    pred = pred.to("cpu")  # TODO: Will this be very slow???
    loss = 0

    noobj_cells = [0] * GRID_HEIGHT * GRID_WIDTH
    # TODO: can I remove grid loops?
    # noobj_bboxes = np.reshape(pred[row, col, 0 : BBS_PER_CELL * 5], (-1, 5))
    for obj in gt:
        # if object_in_cell(
        #     row, col, obj
        # ):  # TODO: Is OBJ_IN_CELL determined by obj x and y or h and w? How is BB_ASSIGNED_TO_OBJ determined?
        # print(obj)
        row, col = get_obj_cell(obj)
        noobj_cells[row * GRID_WIDTH + col] = 1  # object found in this cell
        bbox = (
            determine_responsible_bbox_id(  # TODO: GET LIST OF BBS! this works, right?
                torch.reshape(
                    pred[row, col, 0 : BBS_PER_CELL * 5], (-1, 5)
                ),  # reshape into list of bboxes
                obj["bbox"],
            )
        )
        # if bbox in noobj_bboxes:
        #     noobj_bboxes = noobj_bboxes.delete(bbox)

        pred_x = pred[row, col, bbox * 5].item()
        # print("PRED = ", pred)
        # print("PREX_X = ", pred_x)
        pred_y = pred[row, col, bbox * 5 + 1].item()
        pred_h = pred[row, col, bbox * 5 + 2].item()
        pred_w = pred[row, col, bbox * 5 + 3].item()
        label_x = obj["bbox"][0]  # TODO: CHECK THIS!
        # print("LABEL_X = ", label_x)
        label_y = obj["bbox"][1]  # TODO: CHECK THIS!
        label_h = obj["bbox"][2]  # TODO: CHECK THIS!
        label_w = obj["bbox"][3]  # TODO: CHECK THIS!

        # loss part 1
        # print("LAMBDA_COORD = ", LAMBDA_COORD)
        # print(
        #     "sum_squared_error(label_x, pred_x) = ",
        #     sum_squared_error(label_x, pred_x),
        # )
        # print(
        #     "sum_squared_error(label_y, pred_y) = ",
        #     sum_squared_error(label_y, pred_y),
        # )
        loss += LAMBDA_COORD * (
            sum_squared_error(label_x, pred_x) + sum_squared_error(label_y, pred_y)
        )

        # loss part 2
        loss += LAMBDA_COORD * (
            sum_squared_error(math.sqrt(label_w), math.sqrt(max(0, pred_w)))
            + sum_squared_error(math.sqrt(label_h), math.sqrt(max(0, pred_h)))
        )

        # loss part 3
        loss += sum_squared_error(1, pred[row, col, bbox * 5 + 4])

        # loss part 5 TODO: What if multiple objects are in the same cell? Multiple loss additions? That's the way it works now
        label_class = obj["category_id"]
        for class_id in range(NUM_CLASSES):
            pred_class_conf = pred[row, col, BBS_PER_CELL * 5 + class_id]
            if class_id == label_class:
                loss += sum_squared_error(1, pred_class_conf)
            else:
                loss += sum_squared_error(0, pred_class_conf)

    # for noobj_bbox in noobj_bboxes:
    for idx, cell in enumerate(noobj_cells):
        row = (idx - (idx % GRID_WIDTH)) // GRID_WIDTH  # TODO: double check these
        col = idx % GRID_WIDTH
        # loss part 4
        if cell == 0:
            for i in range(BBS_PER_CELL):
                pred_c = pred[row, col, i * 5 + 4]
                loss += LAMBDA_NOOBJ * sum_squared_error(0, pred_c)

    return loss

    """
    LOSS PART 1:
    PURPOSE: penalize poor bounding box placement
    CONSTANT1 = 5
    for each grid cell:
        for each bounding box assigned to that grid cell:
            if an object appears in that grid cell and that bounding box is responsible for that object:
                loss1 += CONSTANT1 * sum-squared error comparing location of that bounding box and ground truth bounding box for the object
    """
    """
    LOSS PART 2:
    PURPOSE: penalize poorly sized bounding boxes
    for each grid cell:
        for each bounding box assigned to that grid cell:
            if an object appears in that grid cell and that bounding box is responsible for that object:
                loss2 += CONSTANT1 * sum-squared error comparing square roots of width and height of that bounding box and ground truth bounding box for the object
    
    """
    """
    LOSS PART 3:
    PURPOSE: penalize poor confidence score
    for each grid cell:
        for each bounding box assigned to that grid cell:
            if an object appears in that grid cell and that bounding box is responsible for that object:
                loss3 += actual_confidence - predicted_confidence
    """
    """
    LOSS PART 4:
    PURPOSE: penalize poor confidence score
    CONSTANT2 = 0.5
    for each grid cell:
        for each bounding box assigned to that grid cell:
            if an object DOES NOT appears in that grid cell OR that bounding box IS NOT responsible for that object:
                loss4 += CONSTANT2 * (actual_confidence - predicted_confidence)
    """
    """
    LOSS PART 5:
    PURPOSE: penalize incorrect class prediction 
    for each grid cell:
        if an object appears in that grid cell:
            for each class:
                loss5 += actual_class_probability - predicted_class_probability
    """


def sum_squared_error(gt, pred):
    return (gt - pred) ** 2


# expect box format as [x of top left, y of top left, h, w]
def iou(box1, box2):
    return intersection(box1, box2) / union(box1, box2)


def intersection(box1, box2):
    # get intersection box coordinates
    # print("box1 = ", box1)
    # print("box2 = ", box2)
    min_x = max(box1[0], box2[0])
    min_y = max(box1[1], box2[1])
    max_x = min(box1[0] + box1[2], box2[0] + box2[2])
    max_y = min(box1[1] + box1[3], box2[1] + box2[3])

    # check if no overlap
    if min_x > max_x or min_y > max_y:
        return 0.0

    # return area of intersection box
    return (max_x - min_x) * (max_y - min_y)


def union(box1, box2):  # union = area of bbox1 + area of bbox2 - overlap
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    return area1 + area2 - intersection(box1, box2)


def determine_responsible_bbox_id(bboxes, obj_bbox):
    max_iou = 0
    responsible_bbox_id = 0  # if no bboxes have overlap, just punish the first box?
    for idx, bbox in enumerate(bboxes):
        if iou(bbox, obj_bbox) > max_iou:
            responsible_bbox_id = idx
            max_iou = iou(bbox, obj_bbox)
    return responsible_bbox_id


def object_in_cell(row, col, obj):
    obj_center_x, obj_center_y = get_center_of_object(obj)
    cell_h = IMG_HEIGHT / GRID_HEIGHT
    cell_w = IMG_WIDTH / GRID_WIDTH
    cell_x = cell_h * row
    cell_y = cell_w * col
    return (
        obj_center_x >= cell_x
        and obj_center_x < cell_x + cell_w
        and obj_center_y >= cell_y
        and obj_center_y < cell_y + cell_h
    )


def get_obj_cell(obj):
    obj_center_x, obj_center_y = get_center_of_object(obj)
    # print("obj_center_x = ", obj_center_x)
    # print("obj_center_y = ", obj_center_y)
    cell_h = IMG_HEIGHT / GRID_HEIGHT
    cell_w = IMG_WIDTH / GRID_WIDTH

    row = int(obj_center_x // cell_w)
    col = int(obj_center_y // cell_h)
    # print("row = ", row)
    # print("col = ", col)

    return row, col


def get_center_of_object(obj):
    bbox = obj["bbox"]
    # print("bbox: ", bbox)
    return (
        bbox[0] + (bbox[2] / 2),
        bbox[1] + (bbox[3] / 2),
    )


# def yolo_loss(gt, pred):  # gt = list of objs, need to get which obj is in cell
#     pred = pred.to("cpu")  # TODO: Will this be very slow???
#     loss = 0

#     noobj_cells = [0] * GRID_HEIGHT * GRID_WIDTH
#     # TODO: can I remove grid loops?
#     for row in range(GRID_HEIGHT):
#         for col in range(GRID_WIDTH):
#             # noobj_bboxes = np.reshape(pred[row, col, 0 : BBS_PER_CELL * 5], (-1, 5))
#             for obj in gt:
#                 if object_in_cell(
#                     row, col, obj
#                 ):  # TODO: Is OBJ_IN_CELL determined by obj x and y or h and w? How is BB_ASSIGNED_TO_OBJ determined?
#                     noobj_cells[row * GRID_WIDTH + col] = 1  # object found in this cell
#                     bbox = determine_responsible_bbox_id(  # TODO: GET LIST OF BBS! this works, right?
#                         torch.reshape(
#                             pred[row, col, 0 : BBS_PER_CELL * 5], (-1, 5)
#                         ),  # reshape into list of bboxes
#                         obj["bbox"],
#                     )
#                     # if bbox in noobj_bboxes:
#                     #     noobj_bboxes = noobj_bboxes.delete(bbox)

#                     pred_x = pred[row, col, bbox * 5].item()
#                     # print("PRED = ", pred)
#                     # print("PREX_X = ", pred_x)
#                     pred_y = pred[row, col, bbox * 5 + 1].item()
#                     pred_h = pred[row, col, bbox * 5 + 2].item()
#                     pred_w = pred[row, col, bbox * 5 + 3].item()
#                     label_x = obj["bbox"][0].item()  # TODO: CHECK THIS!
#                     # print("LABEL_X = ", label_x)
#                     label_y = obj["bbox"][1].item()  # TODO: CHECK THIS!
#                     label_h = obj["bbox"][2].item()  # TODO: CHECK THIS!
#                     label_w = obj["bbox"][3].item()  # TODO: CHECK THIS!

#                     # loss part 1
#                     # print("LAMBDA_COORD = ", LAMBDA_COORD)
#                     # print(
#                     #     "sum_squared_error(label_x, pred_x) = ",
#                     #     sum_squared_error(label_x, pred_x),
#                     # )
#                     # print(
#                     #     "sum_squared_error(label_y, pred_y) = ",
#                     #     sum_squared_error(label_y, pred_y),
#                     # )
#                     loss += LAMBDA_COORD * (
#                         sum_squared_error(label_x, pred_x)
#                         + sum_squared_error(label_y, pred_y)
#                     )

#                     # loss part 2
#                     loss += LAMBDA_COORD * (
#                         sum_squared_error(math.sqrt(label_w), math.sqrt(max(0, pred_w)))
#                         + sum_squared_error(
#                             math.sqrt(label_h), math.sqrt(max(0, pred_h))
#                         )
#                     )

#                     # loss part 3
#                     loss += sum_squared_error(1, pred[row, col, bbox * 5 + 4])

#                     # loss part 5 TODO: What if multiple objects are in the same cell? Multiple loss additions? That's the way it works now
#                     label_class = obj["category_id"]
#                     for class_id in range(NUM_CLASSES):
#                         pred_class_conf = pred[row, col, BBS_PER_CELL * 5 + class_id]
#                         if class_id == label_class:
#                             loss += sum_squared_error(1, pred_class_conf)
#                         else:
#                             loss += sum_squared_error(0, pred_class_conf)

#             # for noobj_bbox in noobj_bboxes:
#             for idx, cell in enumerate(noobj_cells):
#                 row = (
#                     idx - (idx % GRID_WIDTH)
#                 ) // GRID_WIDTH  # TODO: double check these
#                 col = idx % GRID_WIDTH
#                 # loss part 4
#                 if cell == 0:
#                     for i in range(BBS_PER_CELL):
#                         pred_c = pred[row, col, i * 5 + 4]
#                         loss += LAMBDA_NOOBJ * sum_squared_error(0, pred_c)

#     return loss
