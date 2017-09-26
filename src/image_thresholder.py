import numpy as np
import cv2
import matplotlib.pyplot as plt


class ImageThresholder():
    def threshold(img, s_thresh=(170, 255), l_thresh=(30, 100), sx_thresh=(20, 100), h_thresh=(15, 100), plot=False):
        img = np.copy(img)

        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        h_channel = hls[:, :, 0]

        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        h_binary = np.zeros_like(h_channel)
        h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[((h_binary == 1) & (s_binary == 1)) | (sxbinary == 1)] = 1

        if plot:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
            f.tight_layout()
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=15)
            ax2.imshow(combined_binary, cmap='gray')
            ax2.set_title('Thresholded Image', fontsize=15)

        # optional_plot
        return color_binary, combined_binary
