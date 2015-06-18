__author__ = 'mataevs'

import cv2
import numpy as np


def optical_flow(img, prev_img):
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prev_img_bw = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

    height, width, _ = img.shape

    hsv = np.zeros_like(img)
    hsv[..., 1] = 255

    # Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_img_bw, img_bw, 0.5, 3, 15, 3, 7, 1.5, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180/ np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    flow_gray = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2GRAY)

    # Remove noise and very small blobs
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)

    (_, flow_bw) = cv2.threshold(flow_gray, 254, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    flow_blobs = cv2.dilate(cv2.erode(flow_bw, kernel3), kernel3)

    # Check if the largest blob is large enough
    contours_img = flow_blobs.copy()
    contours, hierarchy = cv2.findContours(contours_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contourAreas = [cv2.contourArea(cnt) for cnt in contours]
    maxContour = max(contourAreas)
    # print "max contour area", maxContour
    if len(contourAreas) == 0 or maxContour < 600:
        True
        # print "No moving person in image"
    else:
        personBlob = contours[contourAreas.index(maxContour)]

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(personBlob)
        boundingRect = (x, y, w, h)

        return flow_rgb, boundingRect

    return flow_rgb, None