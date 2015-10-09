
import numpy as np
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import os
import utils
import numpy


def load_classifier(filepath):
    svm = joblib.load(filepath)
    return svm


def getWindowsAndDescriptors(img_path, scales, subwindow=None):
    base_img = cv2.imread(img_path)

    prev_img_path = utils.get_prev_img(img_path)
    base_prev_img = cv2.imread(prev_img_path)

    windows = []

    for scale in scales:
        img = cv2.resize(base_img, (0, 0), fx=scale, fy=scale)
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        prev_img = cv2.resize(base_prev_img, (0, 0), fx=scale, fy=scale)
        prev_img_bw = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

        height, width, _ = img.shape

        flow = cv2.calcOpticalFlowFarneback(prev_img_bw, img_bw, 0.5, 3, 15, 3, 5, 1.2, 0)

        hsv = np.zeros_like(img)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180/ np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flowRGB = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        flow_bw = cv2.cvtColor(flowRGB, cv2.COLOR_BGR2GRAY)

        if subwindow == None:
            nsx, nsy, nw, nh = 0, 0, width, height
        else:
            nsx, nsy, nw, nh = utils.getDetectionWindow(subwindow, width, height, scale)

        for x in range(nsx, nsx + nw - 64, 16):
            for y in range(nsy, nsy + nh - 128, 16):
                img_crop = img_bw[y:y + 128, x:x + 64]
                hog_gray = hog(img_crop, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), visualise=False)

                flow_crop = flow_bw[y:y + 128, x:x + 64]
                fd_flow = hog(flow_crop, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualise=False)

                fd = hog_gray + fd_flow

                windows.append([(int(x / scale), int(y / scale), int(64/scale), int(128/scale)), fd])
    return windows

def getDecisionFunction(svm, windows):
    return svm.decision_function([w[1] for w in windows])

def test_img_new(svm, img_path, scales, subwindow=None):
    base_img = cv2.imread(img_path)

    prev_img_path = utils.get_prev_img(img_path)
    base_prev_img = cv2.imread(prev_img_path)

    windows = []
    windows_features = []
    sc = []

    for scale in scales:
        img = cv2.resize(base_img, (0, 0), fx=scale, fy=scale)
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        prev_img = cv2.resize(base_prev_img, (0, 0), fx=scale, fy=scale)
        prev_img_bw = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

        height, width, _ = img.shape

        flow = cv2.calcOpticalFlowFarneback(prev_img_bw, img_bw, 0.5, 3, 15, 3, 5, 1.2, 0)

        flowx, flowy = flow[..., 0], flow[..., 1]

        if subwindow == None:
            nsx, nsy, nw, nh = 0, 0, width, height
        else:
            nsx, nsy, nw, nh = utils.getDetectionWindow(subwindow, width, height, scale)

        for x in range(nsx, nsx + nw - 64, 16):
            for y in range(nsy, nsy + nh - 128, 16):
                img_crop = img_bw[y:y + 128, x:x + 64]
                hog_gray = hog(img_crop, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), visualise=False)

                flowx_crop, flowy_crop = flowx[y:y+128, x:x+64], flowy[y:y+128, x:x+64]

                hog_flow_x = hog(flowx_crop, orientations=9, pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2), visualise=False)
                hog_flow_y = hog(flowy_crop, orientations=9, pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2), visualise=False)

                fd = numpy.concatenate((hog_gray, hog_flow_x, hog_flow_y))

                windows.append((x, y))
                windows_features.append(fd)
                sc.append(scale)

    classes = svm.predict(windows_features)

    results = []
    for i in range(0, len(windows)):
            if classes[i] == 1:
                scale = sc[i]
                results.append((int(windows[i][0] / scale), int(windows[i][1] / scale), int(64 / scale), int(128 / scale)))
    return results

def test_img(svm, img_path, scales, subwindow=None):
    base_img = cv2.imread(img_path)

    prev_img_path = utils.get_prev_img(img_path)
    base_prev_img = cv2.imread(prev_img_path)

    windows = []
    windows_features = []
    sc = []

    for scale in scales:
        img = cv2.resize(base_img, (0, 0), fx=scale, fy=scale)
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        prev_img = cv2.resize(base_prev_img, (0, 0), fx=scale, fy=scale)
        prev_img_bw = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

        height, width, _ = img.shape

        flow = cv2.calcOpticalFlowFarneback(prev_img_bw, img_bw, 0.5, 3, 15, 3, 5, 1.2, 0)

        hsv = np.zeros_like(img)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180/ np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flowRGB = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        flow_bw = cv2.cvtColor(flowRGB, cv2.COLOR_BGR2GRAY)

        if subwindow == None:
            nsx, nsy, nw, nh = 0, 0, width, height
        else:
            nsx, nsy, nw, nh = utils.getDetectionWindow(subwindow, width, height, scale)

        for x in range(nsx, nsx + nw - 64, 16):
            for y in range(nsy, nsy + nh - 128, 16):
                img_crop = img_bw[y:y + 128, x:x + 64]
                hog_gray = hog(img_crop, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), visualise=False)

                flow_crop = flow_bw[y:y + 128, x:x + 64]
                fd_flow = hog(flow_crop, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualise=False)
                fd = hog_gray + fd_flow

                windows.append((x, y))
                windows_features.append(fd)
                sc.append(scale)

    classes = svm.predict(windows_features)

    results = []
    for i in range(0, len(windows)):
            if classes[i] == 1:
                scale = sc[i]
                results.append((int(windows[i][0] / scale), int(windows[i][1] / scale), int(64 / scale), int(128 / scale)))
    return results
