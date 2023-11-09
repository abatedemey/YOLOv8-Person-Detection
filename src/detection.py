import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from src.utils import Timer
import logging
from matplotlib import pyplot as plt


DEVICE = "cpu"
YOLOV8_MODEL_PATH = "yolov8s.pt"
SAM_MODEL_PATH = "sam_vit_h_4b8939.pth"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class PersonDetector:
    def __init__(self, yolo_model_path, sam_model_path):
        self.people_class_id = 0

        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(device=DEVICE)

        self.sam_model = sam_model_registry["vit_h"](checkpoint=sam_model_path)
        self.sam_model.to(device=DEVICE)
        self.logger = logger

    def display_masks(self, masks):
        """
        Display the segmentation masks using matplotlib.
        """
        # Check if masks are on GPU, if so, move them to CPU
        if masks.is_cuda:
            masks = masks.cpu()

        for i, mask in enumerate(masks):
            # Convert the torch tensor to a numpy array
            mask_numpy = mask.squeeze().detach().numpy()

            # Normalize the mask to be in [0, 255] range if it's not binary
            mask_numpy = (mask_numpy / mask_numpy.max()) * 255
            mask_numpy = mask_numpy.astype(np.uint8)

            # Display the mask
            plt.figure(figsize=(8, 8))
            plt.imshow(mask_numpy, cmap="gray")
            plt.title(f"Mask {i}")
            plt.axis("off")
            plt.show()

    def detect_people(self, image):
        with torch.no_grad():
            results = self.yolo_model(image, stream=True)

        people_boxes = []
        people_class_id = 0
        for detection in results:
            for i, (box, cls_id) in enumerate(
                zip(detection.boxes.xyxy, detection.boxes.cls.long().tolist())
            ):
                if cls_id == people_class_id:
                    people_boxes.append(detection.boxes.xyxy[i].tolist())

        bbox = [[int(i) for i in box] for box in people_boxes]
        bbox_final = torch.tensor(bbox, device=DEVICE)
        return bbox_final

    def get_segmentation_masks(self, predictor, image, boxes):
        with torch.no_grad():
            transformed_boxes = predictor.transform.apply_boxes_torch(
                boxes.clone().detach(), image.shape[:2]
            )
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
        return masks

    def get_all_contours(self, masks, min_area=50):
        contour_list = []
        for mask in masks:
            contours, _ = cv2.findContours(
                mask.squeeze().cpu().numpy().astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    contour_list.append(contours)

        return contour_list

    def get_masks(self, image_path) -> list:
        timer = Timer(self.logger)
        # Read and process the image
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return []
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect people and get bounding boxes
        boxes = self.detect_people(image)
        timer.checkpoint("Detect people")

        # Check if any boxes were detected
        if boxes.nelement() == 0:
            self.logger.info(f"No people detected in image {image_path}.")
            return []

        # Get segmentation masks
        predictor = SamPredictor(self.sam_model)
        predictor.set_image(image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.get_segmentation_masks(predictor, image_rgb, boxes)
        timer.checkpoint("Get segmentation masks")
        return masks
