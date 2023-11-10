import logging
import os

from src.detection import PersonDetector
from src.visualize import ImageSegmentVisualizer

INPUT_DIR = "images/raw"
OUTPUT_DIR = "images/labeled"

YOLO_MODEL_PATH = "models/yolov8s.pt"
SAM_MODEL_PATH = "models/sam_vit_h_4b8939.pth"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Main:
    def __init__(self, input_dir="images/raw", output_dir="images/labeled"):
        self.people_detector = PersonDetector(YOLO_MODEL_PATH, SAM_MODEL_PATH)
        self.visualizer = ImageSegmentVisualizer()
        self.input_image_dir = input_dir
        self.output_image_dir = output_dir
        self.output_prefix = "labeled"

    def draw_contours(self, masks, image_path, output_image_path):
        # Process image
        contours = self.people_detector.get_all_contours(masks, min_area=50)

        # Check if any contours were detected
        if len(contours) == 0:
            logger.error(f"No contours detected in image {image_path}.")
            return

        # Visualize the contours
        self.visualizer.draw_contours(
            image_path, contours, output_path=output_image_path
        )

        return

    def generate_filenames(self, input_image_dir, output_image_dir, input_filename):
        filename_split_ext = os.path.splitext(input_filename)
        filename_no_ext = filename_split_ext[0]
        filename_ext = filename_split_ext[1]

        input_image_path = os.path.join(input_image_dir, input_filename)
        contour_image_path = os.path.join(
            output_image_dir, f"{filename_no_ext}.contour.{filename_ext}"
        )
        mask_path = os.path.join(
            output_image_dir, f"{filename_no_ext}.mask.{filename_ext}"
        )
        extracted_image_path = os.path.join(
            output_image_dir, f"{filename_no_ext}.extracted.{filename_ext}"
        )
        rembg_image_path = os.path.join(
            output_image_dir, f"{filename_no_ext}.rembg{filename_ext}"
        )
        dis_image_path = os.path.join(
            output_image_dir, f"{filename_no_ext}.dis{filename_ext}"
        )

        filenames = {
            "input": input_image_path,
            "contour": contour_image_path,
            "mask": mask_path,
            "extracted": extracted_image_path,
            "rembg": rembg_image_path,
            "dis": dis_image_path,
        }
        return filenames

    def process_directory(self):
        count_success = 0
        count_fail = 0
        images_failed = []
        for filename in os.listdir(self.input_image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                filenames = self.generate_filenames(
                    self.input_image_dir, self.output_image_dir, filename
                )

                logger.info(f"Processing image: {filenames['input']}")

                # Remove background with rembg
                # self.people_detector.rembg_image(filenames["input"], filenames["rembg"])

                # Remove background with Dichotomous Image Segmentation
                # self.people_detector.dis_image(filenames["input"], filenames["dis"])

                # Get segmentation masks
                masks = self.people_detector.get_masks(filenames["input"])
                if len(masks) > 0:
                    count_success += 1
                else:
                    count_fail += 1
                    images_failed.append(filenames["input"])
                    continue

                # Output masks
                self.visualizer.output_masks(masks, filenames["mask"])

                # Draw contours on original image
                self.draw_contours(masks, filenames["input"], filenames["contour"])

                # Output masked areas from original image
                self.visualizer.extract_masked_areas(
                    filenames["input"], masks, filenames["extracted"], min_area=50
                )

        logger.info(f"Successfully processed {count_success} images.")
        logger.info(f"Failed to process {count_fail} images.")
        logger.info(f"Failed images: {images_failed}")


if __name__ == "__main__":
    main_processor = Main(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
    main_processor.process_directory()
