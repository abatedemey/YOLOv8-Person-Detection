import os
from src.detection import PersonDetector
from src.visualize import ImageSegmentVisualizer

YOLO_MODEL_PATH = "models/yolov8s.pt"
SAM_MODEL_PATH = "models/sam_vit_h_4b8939.pth"


class Main:
    def __init__(self):
        self.people_detector = PersonDetector(YOLO_MODEL_PATH, SAM_MODEL_PATH)
        self.visualizer = ImageSegmentVisualizer()
        self.input_image_dir = "images/raw"
        self.output_image_dir = "images/labeled"
        self.output_prefix = "labeled"

    def process_directory(self):
        for filename in os.listdir(self.input_image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(self.input_image_dir, filename)
                output_image_path = os.path.join(
                    self.output_image_dir, f"{self.output_prefix}_{filename}"
                )

                print(f"Processing image: {image_path}")

                # Process image
                contours = self.people_detector.process_image(image_path)

                if len(contours) == 0:
                    print(f"No people detected in image {image_path}.")
                    continue

                # Visualize the contours
                self.visualizer.draw_contours(
                    image_path, contours, output_path=output_image_path
                )


if __name__ == "__main__":
    main_processor = Main()
    main_processor.process_directory()
