import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

DEVICE = "cpu"
YOLOV8_MODEL_PATH = "yolov8s.pt"
SAM_MODEL_PATH = "sam_vit_h_4b8939.pth"


class PersonDetector:
    def __init__(self, yolo_model_path, sam_model_path):
        self.people_class_id = 0
        self.yolo_model = YOLO(yolo_model_path)
        self.sam_model = sam_model_registry["vit_h"](checkpoint=sam_model_path)
        self.sam_model.to(device=DEVICE)

    def detect_people(self, image):
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

    def find_largest_contour(self, masks):
        contour_list = []
        for mask in masks:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour_list.append(max(contours, key=cv2.contourArea))

        return contour_list

    def find_person_contour(self, contours, image_shape):
        if not contours:
            return None

        # Assuming the person is central in the image
        # find the contour closest to the center
        image_center = np.array(image_shape[:2]) / 2

        def sort_key(cnt):
            M = cv2.moments(cnt)
            if M["m00"] == 0:  # avoid division by zero
                return float("inf")
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            distance_to_center = np.linalg.norm(np.array([cx, cy]) - image_center)
            return distance_to_center

        # Sort the contours by distance to center and choose the closest one
        contours.sort(key=sort_key)
        return contours[0]

    def process_image(self, image_path) -> list:
        # Load models

        # Get bounding boxes
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = self.detect_people(image)

        if boxes.nelement() == 0:
            return []

        # Get segmentation masks
        predictor = SamPredictor(self.sam_model)
        predictor.set_image(image)
        masks = self.get_segmentation_masks(predictor, image, boxes)

        # Get contours
        all_contours = []
        for mask in masks:
            contours, _ = cv2.findContours(
                mask.squeeze().cpu().numpy().astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            all_contours.extend(contours)

        # Now find the person contour from all contours
        person_contour = self.find_person_contour(all_contours, image.shape)

        return [person_contour] if person_contour is not None else []


def run():
    detector = PersonDetector()
    contours = detector.process_image(
        "images/groceries.jpg", YOLOV8_MODEL_PATH, SAM_MODEL_PATH
    )
    print(contours)


if __name__ == "__main__":
    run()
