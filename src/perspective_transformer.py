import numpy as np
import cv2

WARP_SRC_POINTS = np.array([
    [555, 460],
    [740, 460],
    [190, 675],
    [1140, 675]])


class PerspectiveTransformer():
    def __init__(self, src=WARP_SRC_POINTS):
        self.M = None
        self.Minv = None
        self.img_size = None
        self.crop_dict = dict()
        self.src = np.float32(src)
        self.dst_blank = None

    def computePerspectiveTransform(self, img):
        self.img_size = (img.shape[0], img.shape[1])
        a = int(self.img_size[0] * (1 / 3))
        b = int(self.img_size[0] * (2 / 3))
        c = int(self.img_size[1] * (1 / 3))
        d = int(self.img_size[1] * (2 / 3))
        dst = np.float32([
            [a, c],
            [b, c],
            [a, d],
            [b, d]])
        self.crop_dict = {
            "left": a - 50,
            "right": b + 50,
            "top": c - 50,
            "bottom": d + 50}
        self.M = cv2.getPerspectiveTransform(self.src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, self.src)

    def warp(self, img):
        out_img = cv2.warpPerspective(img, self.M, self.img_size, flags=cv2.INTER_LINEAR)
        self.dst_blank = np.zeros_like(out_img)
        return out_img

    def unwarp(self, img):
        return cv2.warpPerspective(img, self.Minv, (self.img_size[1], self.img_size[0]))

    def crop(self, img):
        crop_dict = self.crop_dict
        return img[
            crop_dict["top"]: crop_dict["bottom"],
            crop_dict["left"]: crop_dict["right"]]

    def warp_and_crop(self, img):
        return self.crop(self.warp(img))
