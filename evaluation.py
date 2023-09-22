# mAP calculation:
# 1. compute the Average Precision (AP) for each class
# 2. compute the mean across all classes

from utils import iou
from global_constants import IOU_THRESHOLD

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
        class_detections = sorted(class_detections, key="confidence", reverse=True)

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
    detections = {}
    total_num_gts = 0

    for img, label in dataloader:
        total_num_gts += len(label)

        predictions = model(img)

        for pred in predictions:
            pred_conf = None
            pred_bbox = None
            pred_class = None
            true_positive = None

            if pred_conf < IOU_THRESHOLD:
                continue

            if iou(pred_bbox, closest_object_bbox) < 0.5 or pred_class is wrong:
                true_positive = False
            else:  # TODO: deal with multiple predicted boxes on same object!
                true_positive = True

            detections["pred_class"].append(
                {"confidence": pred_conf, "true_positive": true_positive}
            )

    return calculate_mean_average_precision(
        detections=detections, num_gts=total_num_gts
    )
