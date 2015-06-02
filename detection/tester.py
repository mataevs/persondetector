__author__ = 'mataevs'

import tester_hog
import tester_icf
import cv2
import random
import utils
import time
import os


def test_all(icf_classifier_file, hog_classifier_file, scale=0.6):
    print "foo"
    icf_classifier = tester_icf.load_classifier(icf_classifier_file)
    hog_classifier = tester_hog.load_classifier(hog_classifier_file)

    testImages = utils.getFullImages(
        "/home/mataevs/ptz/dumpNew1",
        "/home/mataevs/ptz/dumpNew2",
        "/home/mataevs/ptz/dumpNew3")
    #     # "/home/mataevs/ptz/ns1",
    #     # "/home/mataevs/ptz/ns2",
    #     # "/home/mataevs/ptz/ns3",
    #     # "/home/mataevs/ptz/ns4")

    # testImages = utils.getFullImages(
    #     "/home/mataevs/ptz/ptz_code/dump_05_05_01_50",
    #     "/home/mataevs/ptz/ptz_code/dump_05_05_01_51"
    #     #"/home/mataevs/ptz/ptz_code/dump_05_05_11_54"
    # )

    while True:
        imgPath = random.choice(testImages)

        before = time.time()

        bestWindowIcf = tester_icf.test_img(icf_classifier, imgPath, scale)

        afterIcf = time.time() - before

        before = time.time()

        bestWindowHog = tester_hog.test_img(hog_classifier, imgPath, scale)

        afterHog = time.time() - before

        print time.time()

        print "icf=" + str(afterIcf) + " hog=" + str(afterHog)

        img = cv2.imread(imgPath)

        img_icf = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        utils.draw_detections(img_icf, [bestWindowIcf])
        cv2.imshow("icf", img_icf)

        img_hog = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        utils.draw_detections(img_hog, [bestWindowHog])
        cv2.imshow("hog", img_hog)

        key = cv2.waitKey(0)

        if key == 27:
            exit(1)

def test_hog_pyramid(hog_classifier_file):

    hog_classifier = tester_hog.load_classifier(hog_classifier_file)

    testImages = utils.getFullImages(
        "/home/mataevs/ptz/ptz_code/dump_05_05_01_50")

    while True:
        imgPath = random.choice(testImages)

        img = cv2.imread(imgPath)

        scale = 0.6
        while scale >= 0.3:
            print scale
            bestWindowHog = tester_hog.test_img(hog_classifier, imgPath, scale)

            img_hog = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            utils.draw_detections(img_hog, [bestWindowHog])
            cv2.imshow("hog_" + str(scale), img_hog)

            scale -= 0.05

        key = cv2.waitKey(0)

        if key == 27:
            exit(1)


def test_multiscale(
        hog_classifier_file,
        icf_classifier_file,
        hog_result_dir,
        icf_result_dir,
        no_samples):
    hog_classifier = tester_hog.load_classifier(hog_classifier_file)
    icf_classifier = tester_icf.load_classifier(icf_classifier_file)

    filepaths = [
        "/home/mataevs/ptz/testsets/dump_05_05_01_50",
        "/home/mataevs/ptz/testsets/dump_05_05_01_51",
        "/home/mataevs/ptz/testsets/dump_05_05_11_54",
        "/home/mataevs/ptz/testsets/dump_07_05_11_07",
        "/home/mataevs/ptz/testsets/dump_07_05_11_40",
        "/home/mataevs/ptz/testsets/dump_07_05_11_46",
        "/home/mataevs/ptz/testsets/dump_07_05_11_49",
        "/home/mataevs/ptz/testsets/dump_07_05_12_02",
        "/home/mataevs/ptz/testsets/dump_07_05_12_03",
        "/home/mataevs/ptz/testsets/dump_07_05_12_05",
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
        cv2.imwrite(hog_result_dir + "/sample_2_" + str(sample) + ".jpg", img_hog)

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
        cv2.imwrite(icf_result_dir + "/sample_2_" + str(sample) + ".jpg", img_icf)

        # key = cv2.waitKey(0)
        #
        # if key == 27:
        #     exit(1)



# test_all("icf/classifier_230.dump", "hog/svm.dump", 0.4)
test_multiscale("hog/svm.dump",
                "icf/classifier_230.dump",
                "./hog_result",
                "./icf_result",
                300)