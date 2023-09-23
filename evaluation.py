# mAP calculation:
# 1. compute the Average Precision (AP) for each class
# 2. compute the mean across all classes

import time
import torch
import torchvision
from model import YOLO
from utils import iou, voc_transform, output_box_to_pred
from global_constants import IOU_THRESHOLD, BBS_PER_CELL
import gc

"""
From guide: "taking the maximum precision whose recall value is greater or equal than recall-2"
"""
# AP calculation:
# 1. get total number of gt objects
# 2. make list of all detections above a threshold (0.5), indicate whether they are true positives (TP) or false positives (FP)
# 3. order predictions in order of confidence
# 4. iterate over each prediction, calculate precision and recall cumulatively (keep track of cumulative TP and FP)
#     - now you are left with list of precision-recall pairs
# 5. Calculate Area Under Curve (AUC) for precision-recall pairs (recall is x-axis, precision is y-axis)
#     - For each pair of points, p1 and p2, AP-per-point = (recall-2 - recall-1) * max(precisions[precision-2, :])
#     - AP = sum(AP-per-point-values)


def calculate_mean_average_precision(detections, num_gts):
    average_precisions = []
    for class_id in detections.keys():
        class_detections = detections[class_id]
        # 3. order predictions in order of confidence
        # print("class id = ", class_detections)
        # print("class detections = ", class_detections)
        class_detections = sorted(
            class_detections, key=lambda d: d["confidence"], reverse=True
        )

        # 4. iterate over each prediction, calculate precision and recall cumulatively (keep track of cumulative TP and FP)
        precisions = []
        recalls = []
        acc_tp, acc_fp = [0] * 2
        for detection in class_detections:
            if detection["true_positive"]:
                acc_tp += 1
            else:
                acc_fp += 1

            precisions.append(acc_tp / (acc_tp + acc_fp))
            recalls.append(acc_tp / num_gts)

        # 5. Calculate Area Under Curve (AUC) for precision-recall pairs (recall is x-axis, precision is y-axis)
        average_precision = 0
        for idx in range(1, len(recalls)):
            average_precision += (recalls[idx] - recalls[idx - 1]) * max(
                precisions[idx:]
            )  # AP-per-point = (recall-2 - recall-1) * max(precisions[precision-2, :])

        average_precisions.append(average_precision)

    return sum(average_precisions) / len(
        average_precisions
    )  # is this value mean average precision? or different?


def evaluation(dataloader, model, dev="cuda"):
    iteration = 0
    detections = {}
    total_num_gts = 0
    start = time.time()

    for img, label in dataloader:
        iteration += 1
        if torch.cuda.is_available():
            img = img.to(dev)
        total_num_gts += len(label)
        matched_objects = []

        # to avoid cuda out of memory error
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            predictions = model(img)

        predictions = torch.squeeze(predictions, 0)
        predictions = predictions.to(
            "cpu"
        )  # TODO: is this slow? struggling to put labels on cuda...
        # print(predictions.shape)

        for row in range(predictions.size(0)):
            for col in range(predictions.size(1)):
                pred_class = get_class_prediction(predictions[row, col, 10:].tolist())
                for bbox_id in range(BBS_PER_CELL):
                    pred_conf = predictions[row, col, bbox_id * 5 + 4]
                    pred_bbox = output_box_to_pred(
                        predictions[row, col, bbox_id * 5 : bbox_id * 5 + 4], row, col
                    )
                    true_positive = None

                    if pred_conf < IOU_THRESHOLD:
                        continue

                    matched_object, matched_obj_class = get_matched_object_class(
                        pred_bbox, label, matched_objects
                    )
                    if matched_obj_class is None or matched_obj_class != pred_class:
                        true_positive = False
                    else:
                        true_positive = True
                        matched_objects.append(matched_object)

                    if pred_class not in detections.keys():
                        detections[pred_class] = []
                    detections[pred_class].append(
                        {"confidence": pred_conf, "true_positive": true_positive}
                    )
        if iteration % 100 == 0:
            print("iteration " + str(iteration) + "...")
            print(
                "elapsed time: ",
                time.time() - start,
                "seconds",
            )

    mAP = calculate_mean_average_precision(detections=detections, num_gts=total_num_gts)
    print("Final PASCAL VOC 2012 mAP result = ", mAP)


def get_matched_object_class(pred_bbox, label, matched_objects):
    for obj in label:
        if (
            obj in matched_objects
        ):  # multiple preds on the same object should count as 1 TP and rest FP
            continue
        if iou(pred_bbox, obj["bbox"]) >= 0.5:
            return obj, obj["category_id"]
    return None, None


def get_class_prediction(class_probabilities):
    max_probability = max(class_probabilities)
    return class_probabilities.index(max_probability)


dev = "cuda"

dataset = torchvision.datasets.voc.VOCDetection(
    "VOC2012", "2012", "val", transforms=voc_transform
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False  # , collate_fn=coco_collate
)
print(len(dataloader))

model = YOLO(batch_size=1)
model.eval()
if torch.cuda.is_available():
    model.to(dev)

evaluation(dataloader, model)
