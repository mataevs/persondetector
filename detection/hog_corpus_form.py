__author__ = 'mataevs'


import utils
import random
import numpy
import cv2
from classifier import *

filepaths = [
    "/home/mataevs/captures/simple/set1",
    "/home/mataevs/captures/simple/set2",
    "/home/mataevs/captures/simple/set3",
    "/home/mataevs/captures/simple/set4",
    "/home/mataevs/captures/simple/ns1",
    "/home/mataevs/captures/simple/ns2",
    "/home/mataevs/captures/simple/ns3",
    "/home/mataevs/captures/simple/ns4",
    "/home/mataevs/captures/simple/dumpNew1",
    "/home/mataevs/captures/simple/dumpNew2",
    "/home/mataevs/captures/simple/dumpNew3",
]
images = utils.getFullImages(*filepaths)
random.shuffle(images)

c = loadCascadeClassifier("cascade_classifier_100_500_2k.dump")

resultsFile = open("hog_corpus_neg.txt", "w")

negatives = 0

imgIndex = 0
while negatives < 10000:
    imgPath = images[imgIndex]

    for scale in [0.3, 0.4, 0.5]:
        windows = c.getWindowsAndDescriptors(imgPath, scale)

        df = c.getDecisionFunctions(windows)

        sorted_df_idx = numpy.argsort(df)

        for i in range(0, 3):
            x, y, w, h = windows[sorted_df_idx[i]][0]
            resultsFile.write(imgPath + " " + str(x) + " " + str(y) + " " + str(scale) + "\n")
            img = cv2.imread(imgPath)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            crop = img[y:y+h, x:x+w]
            cv2.imwrite("hog_neg/" + str(negatives) + ".jpg", crop)
            print "try", imgIndex, "pos", negatives
            negatives += 1

    imgIndex += 1

resultsFile.close()