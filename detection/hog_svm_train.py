from sklearn import svm
import random
from skimage.feature import hog
import cv2
from os.path import isfile, join
from os import listdir
import os
import numpy as np
from sklearn.externals import joblib
import sys
import uuid


def cmp(prefix):
    def cmp_file_names(a, b):
        a_n = int(a.replace(prefix, "").replace(".jpg", ""))
        b_n = int(b.replace(prefix, "").replace(".jpg", ""))

        if a_n > b_n:
            return 1
        if a_n < b_n:
            return -1
        return 0

    return cmp_file_names


def draw_detections(img, rects, thickness=1):
    for i in range(len(rects)):
        x, y, w, h = rects[i][1]
        prob = rects[i][0]
        pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)
        cv2.putText(img, str(int(prob * 100)), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), thickness=1)


def train_classifier(pos_path, neg_path, pos_flow_path, neg_flow_path):
    pos_train = [f for f in listdir(pos_path) if isfile(join(pos_path, f))]
    neg_train = [f for f in listdir(neg_path) if isfile(join(neg_path, f))]

    train_features = []
    train_classes = []

    for example in pos_train:
        img = cv2.imread(join(pos_path, example))
        img = cv2.resize(img, (64, 128))
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_flow = cv2.imread(join(pos_flow_path, example))
        img_flow = cv2.resize(img_flow, (64, 128))
        img_flow_bw = cv2.cvtColor(img_flow, cv2.COLOR_BGR2GRAY)

        fd = hog(img_bw, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualise=False)
        fd_flow = hog(img_flow_bw, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualise=False)
        fd = fd + fd_flow

        train_features.append(fd)
        train_classes.append(1)

    for example in neg_train:
        img = cv2.imread(join(neg_path, example))
        img = cv2.resize(img, (64, 128))
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_flow = cv2.imread(join(neg_flow_path, example))
        img_flow = cv2.resize(img_flow, (64, 128))
        img_flow_bw = cv2.cvtColor(img_flow, cv2.COLOR_BGR2GRAY)

        fd = hog(img_bw, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualise=False)
        fd_flow = hog(img_flow_bw, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualise=False)
        fd = fd + fd_flow

        train_features.append(fd)
        train_classes.append(0)

    svc = svm.SVC(C=1.0, kernel='linear', probability=True).fit(train_features, train_classes)
    return svc

svm = train_classifier(
    '/home/mataevs/ptz/positive/',
    '/home/mataevs/ptz/negative/',
    '/home/mataevs/ptz/positive_flow/',
    '/home/mataevs/ptz/negative_flow/')
joblib.dump(svm, 'svm.dump')