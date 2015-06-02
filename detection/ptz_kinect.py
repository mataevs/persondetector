
import tester_hog
import tester_icf
import cv2
import utils
import os
import math

def read_kinect_metadata(file):
    with open(file) as kmf:
        entries = kmf.readlines()

    ts = []
    for entry in entries:
        comma = entry.index(',')
        ts.append(float(entry[0:comma]))

    return ts


def test_multiscale(
        hog_classifier_file,
        icf_classifier_file,
        imgpaths,
        kinect_detections,
        hog_result_dir,
        icf_result_dir):
    hog_classifier = tester_hog.load_classifier(hog_classifier_file)
    icf_classifier = tester_icf.load_classifier(icf_classifier_file)

    if not os.path.exists(hog_result_dir):
        os.makedirs(hog_result_dir)
    if not os.path.exists(icf_result_dir):
        os.makedirs(icf_result_dir)

    testImages = utils.getFullImages(*imgpaths)
    metadata = utils.parseMetadata(*imgpaths)

    kinect_ts = read_kinect_metadata(kinect_detections)

    print metadata
    print kinect_ts

    scales = [
        [0.45, 0.5, 0.55],
        [0.4, 0.45, 0.5],
        [0.3, 0.35],
        [0.3]
    ]
    scaleSteps = [35, 45, 65, 90]

    for kinectts in kinect_ts:
        sel = min(metadata.items(), key=lambda mt: abs(kinectts - mt[1]['time']))
        print str(kinectts) + " " + str(sel[1]['time'])

        imgPath = sel[0]
        imgTs = sel[1]['time']

        print "### Sample " + str(imgPath) + " ###"

        img = cv2.imread(imgPath)

        tilt = int(metadata[imgPath]['tilt'])

        print tilt

        imgScales = []
        for i in range(0, len(scaleSteps)):
            if tilt < scaleSteps[i]:
                imgScales = scales[i]
                break

        print imgScales

        bestWindowHog = tester_hog.test_img(hog_classifier, imgPath, imgScales)
        if bestWindowHog != None:
            scale = bestWindowHog[4]
            print "best scale hog = " + str(scale)
            img_hog = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            utils.draw_detections(img_hog, [bestWindowHog[0:4]])
        else:
            scale = 0.5
            img_hog = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        cv2.imwrite(hog_result_dir + "/" + str(imgTs) + ".jpg", img_hog)

        bestWindowIcf = tester_icf.test_img(icf_classifier, imgPath, imgScales)
        if bestWindowIcf != None:
            scale = bestWindowIcf[4]
            print "best scale icf = " + str(scale)
            img_icf = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            utils.draw_detections(img_icf, [bestWindowIcf[0:4]])
            cv2.imshow("icf", img_icf)
        else:
            scale = 0.5
            img_icf = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            cv2.imshow("icf", img_icf)
        cv2.imwrite(icf_result_dir + "/" + str(imgTs) + ".jpg", img_icf)

test_multiscale("hog/svm.dump",
                "icf/classifier_230.dump",
                [
                    "/home/mataevs/ptz/testsets/dump_07_05_12_03",
                    "/home/mataevs/ptz/testsets/dump_07_05_12_05",
                ],
                "metadata.csv",
                "./kinect_hog_result",
                "./kinect_icf_result")
