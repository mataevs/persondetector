__author__ = 'mataevs'

from os import listdir
from os.path import isfile, join, realpath
import random
import cv2
import csv

def getFullImages(*folders):
    files = [realpath(join(dir, f)) for dir in folders for f in listdir(dir) if isfile(join(dir, f))]
    return files

def randomize(list):
    random.shuffle(list)
    return list

def draw_detections(img, rects, thickness = 1):
    for i in range(len(rects)):
        x, y, w, h = rects[i]
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
                metadata[dirpath + "/" + row[0] + ".jpg"] = {'time': row[1], 'pan': row[2], 'tilt': row[3], 'zoom': row[4]}

    return metadata