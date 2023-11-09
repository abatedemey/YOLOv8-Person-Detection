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

        cv2.imwrite(output_path, image)

    def draw_contours(
        self, image_path, contours_list, output_path="output_maskimg.jpg"
    ):
        image = self.load_image(image_path)

        # Draw the contours directly on the image
        for contours in contours_list:
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        cv2.imwrite(output_path, image)

    def output_masks(self, masks, output_image_path):
        for mask in masks:
            mask_np = mask.squeeze().detach().numpy().astype(np.uint8)
            cv2.imwrite(output_image_path, mask_np * 255)
            print(f"Mask saved to: {output_image_path}")

    def extract_masked_areas(self, image_path, masks, output_image_path, min_area=50):
        """
        Extract masked areas from the original image and paste them onto a white background.

        Args:
        - image: The original image.
        - masks: The segmentation masks, should be a list of binary masks.
        - min_area: Minimum area of the mask to be considered for extraction.

        Returns:
        - new_image: A new image with masked areas on a white background.
        """
        image = self.load_image(image_path)
        # Create a white background image of the same size as the original image
        new_image = np.ones_like(image, dtype=np.uint8) * 255

        # Check if masks are on GPU, if so, move them to CPU
        if masks.is_cuda:
            masks = masks.cpu()

        # Combine all the masks into a single mask
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for mask in masks:
            mask_np = mask.squeeze().detach().numpy().astype(np.uint8)

            # Find contours in the mask
            contours, _ = cv2.findContours(
                mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Draw the contours onto the combined mask
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    cv2.fillPoly(combined_mask, [cnt], color=(255))

        # Use the combined mask to copy the masked area from the original image
        masked_area = cv2.bitwise_and(image, image, mask=combined_mask)

        # Paste the masked area onto the white background
        new_image[combined_mask == 255] = masked_area[combined_mask == 255]

        # Save the new image
        cv2.imwrite(output_image_path, new_image)
        print(f"Image with masked areas saved to: {output_image_path}")

    def overlay_masks(
        self, image, masks, min_area=500, color=(255, 0, 0), transparency=0.5
    ):
        """
        Overlay masks on the image with transparency, preserving the original image colors where there is no mask.

        Args:
        - image: The original image.
        - masks: The segmentation masks, should be a list of binary masks.
        - min_area: Minimum area of the mask to be considered for overlay.
        - color: Color of the overlay (in BGR format).
        - transparency: Transparency of the overlay. 0 is fully transparent, 1 is opaque.

        Returns:
        - overlay_image: The original image with masks overlaid.
        """
        overlay_image = image.copy()

        # Check if masks are on GPU, if so, move them to CPU
        if masks.is_cuda:
            masks = masks.cpu()

        for mask in masks:
            mask_np = mask.squeeze().detach().numpy().astype(np.uint8)

            # Find contours in the mask
            contours, _ = cv2.findContours(
                mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Create an all zeros mask to draw our overlays onto
            contour_mask = np.zeros_like(mask_np)

            # Filter and draw contours on the contour_mask
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    cv2.fillPoly(
                        contour_mask, [cnt], color=(255, 255, 255)
                    )  # white color

            # Create an overlay that is the color we want the mask to be
            colored_overlay = np.zeros_like(image, dtype=np.uint8)
            colored_overlay[:] = color

            # Use the contour_mask to combine the color overlay with the original image
            overlay_where_masked = cv2.bitwise_and(
                colored_overlay, colored_overlay, mask=contour_mask
            )

            # Combine the original image with the colored overlay where masked
            cv2.addWeighted(
                overlay_where_masked,
                transparency,
                overlay_image,
                1 - transparency,
                0,
                overlay_image,
            )

        return overlay_image
