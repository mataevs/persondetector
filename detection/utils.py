__author__ = 'mataevs'

from os import listdir
from os.path import isfile, join, realpath
import random
import cv2
import csv
import datetime
import time
import os

def get_prev_img(img_path):
    p, ext = os.path.splitext(img_path)
    dir, f = os.path.split(p)

    prev_img = "%04d" % (int(f) + 1,) + ext
    prev_img_path = os.path.join(dir, prev_img)

    print img_path
    print prev_img_path

    return prev_img_path

def getFullImages(*folders):
    files = [realpath(join(dir, f)) for dir in folders for f in listdir(dir) if isfile(join(dir, f))]
    return files

def randomize(list):
    random.shuffle(list)
    return list

def draw_detections(img, rects, thickness = 1):
    for i in range(len(rects)):
        x, y, w, h = rects[i][0:4]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness)


def displayImageDetections(imgPath, windows, probTresh):
    img = cv2.imread(imgPath)

    for w in windows:
        startx, starty, prob = w

def parseMetadata(*dirpaths):
    metadata = {}

    for dirpath in dirpaths:
        filepath = dirpath + ".txt"

        with open(filepath) as metafile:
            metareader = csv.reader(metafile)
            for row in metareader:
                ts = datetime.datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S.%f")
                metadata[dirpath + "/" + row[0] + ".jpg"] = {
                    'time': time.mktime(ts.timetuple()) * 1000 + ts.microsecond/1000,
                    'pan': row[2],
                    'tilt': row[3],
                    'zoom': row[4]}

    return metadata

def rectOverlap(r1, r2):
    (x1, y1, w1, h1) = r1
    (x2, y2, w2, h2) = r2

    top1, top2 = y1, y2
    left1, left2 = x1, x2
    bottom1, bottom2 = y1 + h1, y2 + h2
    right1, right2 = x1 + w1, x2 + w2

    overlap = True
    if (left2 >= right1) or (left1 >= right2):
        overlap = False
    if (top2 >= bottom1) or (top1 >= bottom2):
        overlap = False
    return overlap


def getDetectionWindow(rectangle, width, height, scale):
    [sx, sy, w, h] = [i * scale for i in rectangle]

    cx = sx + w / 2
    cy = sy + h / 2

    nsx = 0 if min(sx, cx - 64) < 0 else min(sx, cx - 64)
    nsy = 0 if min(sy, cy - 128) < 0 else min(sy, cy - 128)
    nw = width - nsx if max(w, cx + 64) > width - nsx else max(w, cx + 64)
    nh = height - nsy if max(h, cy + 128) > height - nsy else max(h, cy + 128)
    # nsx = 0 if sx - (64 - w % 64) - 1 < 0 else sx - (64 - w % 64) - 1
    # nsy = 0 if sy - (128 - h % 128) - 1 < 0 else sy - (128 - h % 128) - 1
    # nw = width - nsx if w + (64 - w % 64) + 1 > width - nsx else w + (64 - w % 64) + 1
    # nh = height - nsy if h + (128 - h % 128) + 1 > height - nsy else h + (128 - h % 128) + 1
    return (int(nsx), int(nsy), int(nw), int(nh))
