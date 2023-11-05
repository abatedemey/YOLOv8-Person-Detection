import numpy as np
import cv2


class ImageSegmentVisualizer:
    def __init__(self):
        pass

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read the image at path: {image_path}")
        return image

    def draw_segments(
        self, image_path, segment_string, output_path="output_maskimg.jpg"
    ):
        segments = self.load_segments_from_string(segment_string)
        image = self.load_image(image_path)
        h, w = image.shape[:2]

        # Normalize the points by image width and height
        for segment in segments:
            segment[:, 0] *= w
            segment[:, 1] *= h

        cv2.drawContours(
            image,
            [segment.astype(np.int32) for segment in segments],
            -1,
            (0, 0, 255),
            1,
        )

        if cv2.imwrite(output_path, image):
            print(f"Image with contours saved to: {output_path}")
        else:
            print("Failed to save the image.")

    def draw_contours(
        self, image_path, contours_list, output_path="output_maskimg.jpg"
    ):
        image = self.load_image(image_path)

        # Draw the contours directly on the image
        cv2.drawContours(image, contours_list, -1, (0, 255, 0), 2)

        if cv2.imwrite(output_path, image):
            print(f"Image with contours saved to: {output_path}")
        else:
            print("Failed to save the image.")
