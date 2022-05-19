from typing import List, Tuple

import cv2
import numpy as np
import torch
from numpy.typing import NDArray


class FeatureExtractionROI:
    """The FeatureExtractionROI class is used to extract the region of interests for the turn signal classifier."""

    def __init__(self, intended_img_size=227):
        self.intended_img_size = intended_img_size
        self.roi_block = round(self.intended_img_size / 5)

    def extract_roi_sequence(self, sequence: List[NDArray]) -> List[Tuple[NDArray, NDArray]]:
        """Extract the region of interests features for a sequence"""
        gray_images = [
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in sequence
        ]

        all_outputs = []
        for i, j in range(len(gray_images) - 1), range(1, len(gray_images)):
            flow = self.calculate_flow(gray_images[i], gray_images[j])

            warped_image = self.warp_image(sequence[i], flow)
            difference_image = self.get_difference_image(warped_image, sequence[j])

            output = self.get_region_of_interests(difference_image)
            all_outputs.append(output)
        return all_outputs

    @staticmethod
    def calculate_flow(prev: NDArray, next: NDArray) -> NDArray:
        return cv2.calcOpticalFlowFarneback(prev=prev, next=next, flow=None, pyr_scale=0.5, levels=3,
                                            winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    @staticmethod
    def warp_image(image: NDArray, flow: NDArray) -> NDArray:
        h, w = flow.shape[:2]
        flow = -flow
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        return cv2.remap(src=image, map1=flow, map2=None, interpolation=cv2.INTER_LINEAR)

    def get_region_of_interests(self, image) -> Tuple[NDArray, NDArray]:
        x_start = 0
        x_end = 2 * self.roi_block
        y_start = self.roi_block
        y_end = 3 * self.roi_block
        left_roi = image[y_start:y_end, x_start:x_end]
        right_roi = image[y_start:y_end, -x_end:]
        return left_roi, right_roi

    @staticmethod
    def get_difference_image(image_a: NDArray, image_b: NDArray) -> NDArray:
        # Convert to int32 to allow subtraction
        return np.abs(image_a.astype(np.int32) - image_b.astype(np.int32)).astype(np.uint8)
