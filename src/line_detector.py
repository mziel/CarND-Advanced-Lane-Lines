import numpy as np
import cv2
import matplotlib.pyplot as plt


class Line():
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients of the last n fits of the line
        self.recent_fit = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def calculate_radius(self, ploty):
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.ally * self.ym_per_pix, self.allx * self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        curverad = ((1 + (2 * fit_cr[0] * y_eval * self.ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2 * fit_cr[0])

        return curverad


class LineDetector():
    def __init__(self, moving_avg_window=20, diff_max = 0.3):
        self.nwindows = 9  # Choose the number of sliding windows
        self.margin = 30  # Set the width of the windows +/- margin
        self.minpix = 10  # Set minimum number of pixels found to recenter window
        self.diff_max = diff_max
        self.moving_avg_window = moving_avg_window
        self.left = Line()
        self.right = Line()
        self.ploty = None

    def visualise(self, img, lefty, leftx, righty, rightx, left_fitx, right_fitx, ploty):
        img[lefty, leftx] = [255, 0, 0]
        img[righty, rightx] = [0, 0, 255]
        plt.imshow(img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

    def update_lines(self, side, is_first, fitx, fit, x, y):
        if side == "left":
            obj = self.left
        elif side == "right":
            obj = self.right
        else:
            raise ValueError("Need to choose left or right")

        if is_first:
            # Save to right Line class
            obj.detected = True
            obj.recent_xfitted.append(fitx)
            obj.bestx = fitx
            obj.recent_fit.append(fit)
            obj.best_fit = fit
            obj.allx = x
            obj.ally = y
        else:
            # TODO: Save to left Line class
            obj.detected = True
            obj.diffs = obj.best_fit - fit
            if np.sum(np.abs(obj.diffs)) < self.diff_max:
                obj.recent_xfitted.append(fitx)
                obj.recent_xfitted = obj.recent_xfitted[-self.moving_avg_window:]
                obj.bestx = np.mean(np.array(obj.recent_xfitted), axis=0)
                obj.recent_fit.append(fit)
                obj.recent_fit = obj.recent_fit[-self.moving_avg_window:]
                obj.best_fit = np.mean(np.array(obj.recent_fit), axis=0)
                obj.allx = x
                obj.ally = y
            else:
                obj.detected = False

    def detect(self, img, plot=False):
        if not self.left.detected:
            self.detect_first(img, plot)
        elif not self.right.detected:
            self.detect_first(img, plot)
        else:
            self.detect_next(img, plot)

    def detect_first(self, img, plot=False):
        # Assuming you have created a warped binary image called "img"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0] / 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 3)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(img.shape[0] / self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > self.minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        # Save to left and right Line class
        self.update_lines("left", True, left_fitx, left_fit, leftx, lefty)
        self.update_lines("right", True, right_fitx, right_fit, rightx, righty)
        self.ploty = ploty

        if plot:
            self.visualise(out_img, lefty, leftx, righty, rightx, left_fitx, right_fitx, ploty)

    def detect_next(self, img, plot=False):
        out_img = np.dstack((img, img, img)) * 255
        left_fit = self.left.best_fit
        right_fit = self.right.best_fit

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy +
            left_fit[2] - self.margin)) & (nonzerox < (left_fit[0] * (nonzeroy**2) +
            left_fit[1] * nonzeroy + left_fit[2] + self.margin)))

        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy +
            right_fit[2] - self.margin)) & (nonzerox < (right_fit[0] * (nonzeroy**2) +
            right_fit[1] * nonzeroy + right_fit[2] + self.margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        # Save to left and right Line class
        self.update_lines("left", False, left_fitx, left_fit, leftx, lefty)
        self.update_lines("right", False, right_fitx, right_fit, rightx, righty)
        self.ploty = ploty

        if plot:
            self.visualise(out_img, lefty, leftx, righty, rightx, left_fitx, right_fitx, ploty)

    def calculate_radius(self):
        left_curverad = self.left.calculate_radius(self.ploty)
        right_curverad = self.right.calculate_radius(self.ploty)
        return (left_curverad + right_curverad) / 2
