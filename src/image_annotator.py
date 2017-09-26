import numpy as np
import cv2


class ImageAnnotator():
    def __init__(self, perspective_transformer, line_detector):
        # Extract data from input classes
        left_fitx = line_detector.left.bestx
        right_fitx = line_detector.right.bestx
        ploty = line_detector.ploty
        crop_dict = perspective_transformer.crop_dict

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(perspective_transformer.dst_blank).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly() and convert to uncropped space
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))]) + [crop_dict["left"], crop_dict["top"]]
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]) + [crop_dict["left"], crop_dict["top"]]
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        cv2.polylines(color_warp, np.int_(pts_left), isClosed=0, color=(0, 0, 255), thickness=10)
        cv2.polylines(color_warp, np.int_(pts_right), isClosed=0, color=(255, 0, 0), thickness=10)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        self.unwarped_lines = perspective_transformer.unwarp(color_warp)
        self.radius = line_detector.calculate_radius()

    def calculate_distance(self, img):
        lane_bottom = np.argwhere((self.unwarped_lines[-1, :, 1] > 0) * 1)
        left_bottom = np.min(lane_bottom)
        right_bottom = np.max(lane_bottom)
        xm_per_pix = 3.7 / (right_bottom - left_bottom)  # meters per pixel in x dimension
        middle_of_image = img.shape[1] / 2
        middle_of_lane = (right_bottom + left_bottom) / 2
        distance_in_m = abs(middle_of_lane - middle_of_image) * xm_per_pix
        return distance_in_m

    def annotate_distance(self, img):
        distance = self.calculate_distance(img)
        direction = 'right' if distance > 0 else 'left'
        msg = "Vehicle is {}m {} of center".format(round(distance, 2), direction)
        return cv2.putText(np.copy(img), msg,
            (int(img.shape[1] / 4), int(img.shape[0] / 10 * 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2)

    def annotate_lines(self, img):
        return cv2.addWeighted(np.copy(img), 1, self.unwarped_lines, 0.3, 0)

    def annotate_radius(self, img):
        msg = "Radius of curvature = {} (m)".format(round(self.radius, 2))
        return cv2.putText(np.copy(img), msg,
            (int(img.shape[1] / 4), int(img.shape[0] / 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2)

    def annotate(self, img):
        lines = self.annotate_lines(img)
        radius = self.annotate_radius(lines)
        distance = self.annotate_distance(radius)
        return distance
