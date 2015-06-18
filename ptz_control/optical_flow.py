__author__ = 'mataevs'

import cv2
import numpy as np
import imp

utils = imp.load_source('utils', '/home/mataevs/code/persondetector/detection/utils.py')


def count_move_percentage(image):
    movement = 0
    for row in image:
        for pixel in row:
            if pixel != 0:
                movement += 1

    return float(movement) / (image.shape[0] * image.shape[1]) * 100

def computeAverageFlow(flow, cnt, boundingRect):
    noPixels = 1
    flow_x = 0
    flow_y = 0

    x, y, w, h = boundingRect

    for j in range(y, y + h):
        for i in range(x, x + w):
            if cv2.pointPolygonTest(cnt, (i, j), False) >= 0:
                noPixels += 1
                flow_x += flow[j][i][0]
                flow_y += flow[j][i][1]

    flow_x /= float(noPixels)
    flow_y /= float(noPixels)

    return flow_x, flow_y

def boundaryMovementDetection(img, rect, centroid):
    height, width, _ = img.shape
    moveDirections = []
    if utils.rectOverlap(rect, (0, 0, width, 50)):
        moveDirections.append("up")
    if utils.rectOverlap(rect, (0, height - 50, width, 50)):
        moveDirections.append("down")
    if utils.rectOverlap(rect, (0, 0, 50, height)):
        moveDirections.append("left")
    if utils.rectOverlap(rect, (width - 50, 0, 50, height)):
        moveDirections.append("right")
    return moveDirections
#
# def moveDirectionsWithFlow(moveDirections, flow_x, flow_y):
#     newMoveDirections = []
#     if flow_x < 0 and 'right' in moveDirections:
#         newMoveDirections.append('right')
#     if flow_x > 0 and 'left' in moveDirections:
#         newMoveDirections.append('left')
#     if flow_y < 0 and 'up' in moveDirections:
#         newMoveDirections.append('up')
#     if flow_y > 0 and 'down' in moveDirections:
#         newMoveDirections.append('down')
#     return newMoveDirections

def optical_flow(img, prev_img):
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    prev_img = cv2.resize(prev_img, (0, 0), fx=0.5, fy=0.5)
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

    moveDirections = []
    centroid = None
    blob_flow = None

    # Check if the flow blobs cover a large area of the image
    flowPct = count_move_percentage(flow_blobs)
    # print "flow percentage", flowPct
    if flowPct > 15:
        True
        # print "No moving person in image"
    else:
        # Check if the largest blob is large enough
        contours_img = flow_blobs.copy()
        contours, hierarchy = cv2.findContours(contours_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

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
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness=3)

            # Centroid
            M = cv2.moments(personBlob)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(img, (cx, cy), 2, (0, 0, 255))

            flow_x, flow_y = computeAverageFlow(flow, personBlob, boundingRect)
            flow_cx, flow_cy = flow[cy][cx][0], flow[cy][cx][1]
            # print "Average flow:", flow_x, flow_y
            # print "Centroid flow:", flow_cx, flow_cy

            moveDirections = boundaryMovementDetection(img, boundingRect, (cx, cy))

            # moveDirections = moveDirectionsWithFlow(moveDirections, flow_x, flow_y)

            centroid = (cx, cy)

            blob_flow = (flow_x, flow_y)

    return img, flow_rgb, flow_blobs, centroid, blob_flow, moveDirections

def optical_flow_in_img(img, flowx, flowy):
    cv2.putText(
        img,
        "x=" + str(int(flowx * 100) / 100.0) + " y=" + str(int(flowy * 100) / 100.0),
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        thickness=2)