import numpy as np
import cv2
import matplotlib.pyplot as plt


class CameraCalibrator():
    def __init__(self, nx=9, ny=6):
        self.nx = nx
        self.ny = ny
        self.images = None
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane.
        self.img_size = None
        self.dist = None
        self.mtx = None

    def searchChessboardCorners(self, plot=False):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.ny * self.nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(self.images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.img_size = (img.shape[1], img.shape[0])

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            # If found, add object points, image points
            if ret:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
                if plot:
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
                    # write_name = 'corners_found'+str(idx)+'.jpg'
                    # cv2.imwrite(write_name, img)
                    cv2.imshow('img', img)
                    cv2.waitKey(500)

    def calibrateCamera(self):
        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.img_size, None, None)
        self.mtx = mtx
        self.dist = dist

    def fit(self, images=[]):
        self.images = images
        self.searchChessboardCorners()
        self.calibrateCamera()

    def transform(self, img, plot=False):
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        if plot:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=15)
            ax2.imshow(dst)
            ax2.set_title('Undistorted Image', fontsize=15)
        return dst

    def write(self, img, loc):
        cv2.imwrite(loc, img)
