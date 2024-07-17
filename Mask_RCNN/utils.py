import cv2
import numpy as np
import torch
import random

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward images
        outputs = model(image)
        # get scores
        scores = list(outputs[0]['scores'].detach().cpu().numpy())
        # index scores above threshold
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)
        # get masks
        masks = (outputs[0]['masks'] > .5).squeeze().detach().cpu().numpy()
        masks = masks[:thresholded_preds_count]

        # get bounding boxes in (x1, y1) (x2, y2) format
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in outputs[0]["boxes"].detach().cpu()]
        boxes = boxes[:thresholded_preds_count]

        # get labels
        labels = [coco_names[i] for i in outputs[0]["labels"]]

        return masks, boxes, labels


def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1
    # transparency param
    beta = .6
    gamma = 0

    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply rnd color to mask
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
        # combine masks into single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        # convert PIL to np array
        image = np.array(image)
        # RGB to BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask to img
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw bounding boxes
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, thickness=2)
        # add text to rechtangle
        cv2.putText(image, labels[i], (boxes[i][0][0], boxes[i][0][1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                 1, color, thickness=2, lineType=cv2.LINE_AA)
    return image
