
import numpy as np
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import os


def load_classifier(filepath):
    svm = joblib.load(filepath)
    return svm

def get_prev_img(img_path):
    p, ext = os.path.splitext(img_path)
    dir, f = os.path.split(p)

    prev_img = "%04d" % (int(f) + 1,) + ext
    prev_img_path = os.path.join(dir, prev_img)

    print img_path
    print prev_img_path

    return prev_img_path

def test_img(svm, img_path, scales):
    base_img = cv2.imread(img_path)

    prev_img_path = get_prev_img(img_path)
    base_prev_img = cv2.imread(prev_img_path)

    windows = []
    windows_features = []

    for scale in scales:
        img = cv2.resize(base_img, (0, 0), fx=scale, fy=scale)
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        prev_img = cv2.resize(base_prev_img, (0, 0), fx=scale, fy=scale)
        prev_img_bw = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

        height, width, _ = img.shape

        hsv = np.zeros_like(img)
        hsv[..., 1] = 255

        flow = cv2.calcOpticalFlowFarneback(prev_img_bw, img_bw, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180/ np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_bw = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        flow_bw = cv2.cvtColor(flow_bw, cv2.COLOR_BGR2GRAY)

        for x in range(0, width - 64, 16):
            for y in range(0, height - 128, 16):
                img_crop = img_bw[y:y + 128, x:x + 64]
                fd = hog(img_crop, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), visualise=False)

                flow_crop = flow_bw[y:y + 128, x:x + 64]
                fd_flow = hog(flow_crop, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualise=False)

                fd = fd + fd_flow

                windows.append((x, y, 64, 128, scale))
                windows_features.append(fd)

    probabilities = svm.predict_proba(windows_features)
    classes = svm.predict(windows_features)

    maxProb = -1
    bestWindowIndex = -1

    for i in range(len(windows_features)):
        if classes[i] == 1:
            if maxProb < probabilities[i][1]:
                maxProb = probabilities[i][1]
                bestWindowIndex = i

    if bestWindowIndex == -1:
        return None

    print "max prob hog = " + str(maxProb)

    return windows[bestWindowIndex]