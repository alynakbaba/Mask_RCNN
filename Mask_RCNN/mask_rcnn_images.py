import torch
import torchvision
import cv2
import argparse
import os

from PIL import Image
from utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import numpy as np

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='Path to the input data')
parser.add_argument('-t', '--threshold', default=0.965, type=float, help='Score threshold for discarding detection')
args = vars(parser.parse_args())

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1, progress=True)
model = model.to(device).eval()

# Image transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load and preprocess image
image_path = args['input']
image = Image.open(image_path).convert('RGB')
orig_image = image.copy()  # Keep original image for OpenCV operations

image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Perform inference
masks, boxes, labels = get_outputs(image, model, args['threshold'])
print(f"Masks: {masks}, Boxes: {boxes}, Labels: {labels}")

# Draw segmentation map
result = draw_segmentation_map(orig_image, masks, boxes, labels)
print(f"Result: {result}")

# Convert result to OpenCV format
result_cv2 = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

# Set save path
filename = os.path.basename(args['input']).split('.')[0]
save_path = os.path.join('..', 'Mask_RCNN\outputs', f"{filename}.jpg")

# Ensure directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the result
cv2.imwrite(save_path, result_cv2)
print(f"Result saved to {os.path.abspath(save_path)}")

#python mask_rcnn_images.py -i C:\Users\aleya\Masaüstü\Mask_RCNN\input\images33.jpg -t 0.965