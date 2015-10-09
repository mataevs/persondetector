__author__ = 'mataevs'

import os
import utils
import cv2
import time
import classifier
from classifier import *
import detection_checker
from collections import Counter

def get_images(c, img_path, scales, subwindow=None):
    totalWindows = []
    for scale in scales:
        windows = c.getWindowsAndDescriptors(img_path, scale=scale, subwindow=subwindow)
        for i in range(0, len(windows)):
            # each window: (x,y,w,h), [featureDescriptor]
            x, y, w, h = windows[i][0]
            windows[i][0] = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
        totalWindows += windows
    return totalWindows

def get_decision_function(c, windows, thresh):
    return c.getDecisionFunctions(windows, threshold=thresh)

def test_img(c, img_path, scales, subwindow=None):
    results = []
    classes = []
    sc = []

    for scale in scales:
        res, cls = c.testImage(img_path, scale=scale, subwindow=subwindow)
        results = results + res
        classes = classes + cls
        sc = sc + [scale] * len(res)

    # def maxFunc(p):
    #     return p[2][1]
    #
    # if results == []:
    #     return []
    # maxRes = max(results, key=maxFunc)

    # for i in range(0, len(results)):
    #     result = results[i]
    #     if result[0] == maxRes[0] and result[1] == maxRes[1] and (result[2] == maxRes[2]).all():
    #         bestIndex = i
    #         break

    # return all positive windows detected at the same scale as the best window
    # bestWindowScale = sc[bestIndex]
    # bestWindows = []
    windows = []
    for i in range(0, len(classes)):
        if classes[i] == 1:
            scale = sc[i]
            windows.append((int(results[i][0] / scale), int(results[i][1] / scale), int(64 / scale), int(128 / scale)))

    if len(windows) == 0:
        return None
    return windows


def test_cascade(
        classifier_file,
        icf_result_dir,
        checker,
        resultsFile):

    cascade_detector = classifier.loadCascadeClassifier(classifier_file)

    filepaths = [
        "/home/mataevs/captures/metadata/dump_05_05_01_50",
        "/home/mataevs/captures/metadata/dump_05_06_13_10",
        "/home/mataevs/captures/metadata/dump_10_06_11_47",
        "/home/mataevs/captures/metadata/dump_05_05_01_51",
        "/home/mataevs/captures/metadata/dump_05_06_13_15",
        "/home/mataevs/captures/metadata/dump_07_05_11_40",
        "/home/mataevs/captures/metadata/dump_10_06_11_48",
        "/home/mataevs/captures/metadata/dump_05_05_11_54",
        "/home/mataevs/captures/metadata/dump_05_06_13_20",
        "/home/mataevs/captures/metadata/dump_07_05_11_46",
        "/home/mataevs/captures/metadata/dump_10_06_12_16",
        "/home/mataevs/captures/metadata/dump_05_06_12_57",
        "/home/mataevs/captures/metadata/dump_05_06_13_21",
        "/home/mataevs/captures/metadata/dump_05_06_13_24",
        "/home/mataevs/captures/metadata/dump_16_06_14_57",
        "/home/mataevs/captures/metadata/dump_05_06_13_25",
        "/home/mataevs/captures/metadata/dump_07_05_12_03",
        "/home/mataevs/captures/metadata/dump_16_06_15_26",
        "/home/mataevs/captures/metadata/dump_05_06_13_28",
        "/home/mataevs/captures/metadata/dump_07_05_12_05"
    ]

    if not os.path.exists(icf_result_dir):
        os.makedirs(icf_result_dir)

    metadata = utils.parseMetadata(*filepaths)

    scales = [
        [0.45, 0.5, 0.55],
        [0.4, 0.45, 0.5],
        [0.3, 0.35],
        [0.3]
    ]
    scaleSteps = [35, 45, 65, 90]

    resultsIcf = open(resultsFile + "cascade.txt", "w")

    sample = 0
    for imgPath in checker.getFileList():

        img = cv2.imread(imgPath)

        tilt = int(metadata[imgPath]['tilt'])
        if tilt > 90:
            tilt = 90 - (tilt - 90)

        imgScales = []
        for i in range(0, len(scaleSteps)):
            if tilt < scaleSteps[i]:
                imgScales = scales[i]
                break

        #prev_img_path = utils.get_prev_img(imgPath)
        # prev_img = cv2.imread(prev_img_path)

        # flow_rgb, boundingRect = optical_flow.optical_flow(img, prev_img)
        flow_rgb, boundingRect = None, None

        height, width, _ = img.shape

        before = time.time()
        pos_windows = test_img(cascade_detector, imgPath, imgScales, subwindow=boundingRect)
        after = time.time()

        print "Sample", sample, "time elapsed=", after-before

        if pos_windows != None and pos_windows != []:
            utils.draw_detections(img, pos_windows)

        cv2.imwrite(icf_result_dir + "/sample_2_" + str(sample) + ".jpg", img)

        # Check detections
        detections, truePositive = checker.checkDetections(imgPath, pos_windows)
        c = Counter(detections)
        truePositives = c[True]
        falsePositives = c[False]
        falseNegative = 0 if truePositive else 1
        resultsIcf.write(imgPath + " tp=" + str(truePositives) + " fp=" + str(falsePositives) + " fn=" + str(falseNegative) + "\n")

        sample += 1

    resultsIcf.close()

if __name__ == "__main__":
    checker = detection_checker.Checker("annotations.txt")
    test_cascade("cascade_classifier_100_500_2k.dump",
                 "cascade_5k_500",
                 checker,
                 "results_cascade_5k_500_")
