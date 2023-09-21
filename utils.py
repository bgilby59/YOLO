from global_constants import IMG_WIDTH, IMG_HEIGHT, VOC_CLASSES
from torchvision import transforms
from visualize import visualize_imgs


# calculate iou
# expect box format as [x of top left, y of top left, w, h]
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


def voc_transform(img, target):
    width_scaling = IMG_WIDTH / img.width
    height_scaling = IMG_HEIGHT / img.height

    objects = []
    for obj in target["annotation"]["object"]:
        coco_bbox = []
        bbox = obj["bndbox"]
        coco_bbox.append(int(bbox["xmin"]) * width_scaling)  # x
        coco_bbox.append(int(bbox["ymin"]) * height_scaling)  # y
        coco_bbox.append(int(bbox["xmax"]) * width_scaling - coco_bbox[0])  # width
        coco_bbox.append(int(bbox["ymax"]) * height_scaling - coco_bbox[1])  # height
        objects.append({"bbox": coco_bbox, "category_id": VOC_CLASSES[obj["name"]]})
    target = objects

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

    pil_transform = transforms.ToPILImage()

    visualize_imgs(pil_transform(img), target, vis=False)

    return img, target


def coco_collate(batch):
    return tuple(zip(*batch))


def labels_to_dev(labels, dev="cuda"):
    return False
