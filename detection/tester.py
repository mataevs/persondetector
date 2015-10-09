__author__ = 'mataevs'

import tester_hog
import tester_icf
import cv2
import random
import utils
import os
import detection_checker
import optical_flow
from collections import Counter
import time

def test_multiscale(
        hog_classifier_file,
        icf_classifier_file,
        hog_result_dir,
        icf_result_dir,
        no_samples):
    hog_classifier = tester_hog.load_classifier(hog_classifier_file)
    icf_classifier = tester_icf.load_classifier(icf_classifier_file)

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

    if not os.path.exists(hog_result_dir):
        os.makedirs(hog_result_dir)
    if not os.path.exists(icf_result_dir):
        os.makedirs(icf_result_dir)

    testImages = utils.getFullImages(*filepaths)

    metadata = utils.parseMetadata(*filepaths)

    scales = [
        [0.45, 0.5, 0.55],
        [0.4, 0.45, 0.5],
        [0.3, 0.35],
        [0.3]
    ]
    scaleSteps = [35, 45, 65, 90]

    for sample in range(0, no_samples):
        print "### Sample " + str(sample) + " ###"
        imgPath = random.choice(testImages)

        img = cv2.imread(imgPath)

        tilt = int(metadata[imgPath]['tilt'])
        if tilt > 90:
            tilt = 90 - (tilt - 90)

        imgScales = []
        for i in range(0, len(scaleSteps)):
            if tilt < scaleSteps[i]:
                imgScales = scales[i]
                break

        print imgScales

        prev_img_path = utils.get_prev_img(imgPath)
        prev_img = cv2.imread(prev_img_path)

        flow_rgb, boundingRect = optical_flow.optical_flow(img, prev_img)

        height, width, _ = img.shape

        bestWindowsHog = tester_hog.test_img(hog_classifier, imgPath, imgScales, allPositive=True, flow_rgb=flow_rgb, subwindow=boundingRect)
        if bestWindowsHog != None and bestWindowsHog != []:
            scale = bestWindowsHog[0][4]
            img_hog = cv2.resize(img, (0, 0), fx=scale, fy=scale)

            if boundingRect != None:
                x, y, w, h = utils.getDetectionWindow(boundingRect, img_hog.shape[1], img_hog.shape[0], scale)
                cv2.rectangle(img_hog, (x, y), (x+w, y+h), (0, 0, 255), thickness=2, lineType=8)

            utils.draw_detections(img_hog, bestWindowsHog)
        else:
            scale = 0.5
            img_hog = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            if boundingRect != None:
                x, y, w, h = utils.getDetectionWindow(boundingRect, img_hog.shape[1], img_hog.shape[0], scale)
                cv2.rectangle(img_hog, (x, y), (x+w, y+h), (0, 0, 255), thickness=2, lineType=8)

        cv2.imwrite(hog_result_dir + "/sample_2_" + str(sample) + ".jpg", img_hog)

        bestWindowsIcf = tester_icf.test_img(icf_classifier, imgPath, imgScales, allPositive=True, subwindow=boundingRect)
        if bestWindowsIcf != None and bestWindowsIcf != []:
            scale = bestWindowsIcf[0][4]
            img_icf = cv2.resize(img, (0, 0), fx=scale, fy=scale)

            if boundingRect != None:
                x, y, w, h = utils.getDetectionWindow(boundingRect, img_icf.shape[1], img_icf.shape[0], scale)
                cv2.rectangle(img_icf, (x, y), (x+w, y+h), (0, 0, 255), thickness=2, lineType=8)

            utils.draw_detections(img_icf, bestWindowsIcf)
        else:
            scale = 0.5
            img_icf = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            if boundingRect != None:
                x, y, w, h = utils.getDetectionWindow(boundingRect, img_icf.shape[1], img_icf.shape[0], scale)
                cv2.rectangle(img_icf, (x, y), (x+w, y+h), (0, 0, 255), thickness=2, lineType=8)

        cv2.imwrite(icf_result_dir + "/sample_2_" + str(sample) + ".jpg", img_icf)

def test_multiscale_checker(
        hog_classifier_file,
        icf_classifier_file,
        hog_result_dir,
        icf_result_dir,
        checker,
        resultsFile):
    hog_classifier = tester_hog.load_classifier(hog_classifier_file)
    # icf_classifier = tester_icf.load_classifier(icf_classifier_file)

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

    if not os.path.exists(hog_result_dir):
        os.makedirs(hog_result_dir)
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

    resultsHog = open(resultsFile + "hog.txt", "w")
    # resultsIcf = open(resultsFile + "icf.txt", "w")

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

        posWindowsHog = tester_hog.test_img(hog_classifier, imgPath, imgScales, subwindow=boundingRect)
        if posWindowsHog != None and posWindowsHog != []:
            utils.draw_detections(img, posWindowsHog)

        cv2.imwrite(hog_result_dir + "/sample_2_" + str(sample) + ".jpg", img)

        # Check detections
        detections, truePositive = checker.checkDetections(imgPath, posWindowsHog)
        c = Counter(detections)
        truePositives = c[True]
        falsePositives = c[False]
        falseNegative = 0 if truePositive else 1
        resultsHog.write(imgPath + " tp=" + str(truePositives) + " fp=" + str(falsePositives) + " fn=" + str(falseNegative) + "\n")

        # beforeIcf = time.time()
        # windowsIcf = tester_icf.test_img(icf_classifier, imgPath, imgScales, allPositive=True, subwindow=boundingRect)
        # afterIcf = time.time()
        #
        # print "Sample", sample, "time elapsed=", afterIcf-beforeIcf
        #
        # if windowsIcf != None and windowsIcf != []:
        #     scale = windowsIcf[0][4]
        #     img_icf = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        #     utils.draw_detections(img_icf, windowsIcf)
        # else:
        #     scale = 0.5
        #     img_icf = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        # cv2.imwrite(icf_result_dir + "/sample_2_" + str(sample) + ".jpg", img_icf)
        #
        # # Check detections
        # detections, truePositive = checker.checkDetections(imgPath, windowsIcf)
        # c = Counter(detections)
        # truePositives = c[True]
        # falsePositives = c[False]
        # falseNegative = 0 if truePositive else 1
        # resultsIcf.write(imgPath + " tp=" + str(truePositives) + " fp=" + str(falsePositives) + " fn=" + str(falseNegative) + "\n")

        sample += 1

    # resultsHog.close()
    # resultsIcf.close()

# test_multiscale("hog/svm.dump",
#                 "icf_new_5000_2000_1k.dump",
#                 "./hog_result_all_flow",
#                 "./icf_result_all_flow",
#                 100)

checker = detection_checker.Checker("annotations.txt")
test_multiscale_checker("hog/svm.dump",
                "icf_classifiers/icf_new_5000f_2000e.dump",
                "./hog_result_dir",
                "./icf_5000_2000",
                checker,
                "results_hog_")

