__author__ = 'mataevs'

from classifier import Classifier
import utils
import pickle
import random
import cv2
import numpy

def train():
    posImages = utils.getFullImages(
        "/home/mataevs/ptz/INRIAPerson/train/pos",
        "/home/mataevs/ptz/positive")

    posImages = utils.randomize(posImages)

    negImages = utils.getFullImages(
        "/home/mataevs/ptz/INRIAPerson/train/neg",
        "/home/mataevs/ptz/negative")
    negImages = utils.randomize(negImages)

    print len(posImages)
    print len(negImages)

    c = Classifier()
    print "Starting training"
    c.train(posImages, negImages, 200, 300)
    print "Finished training"
    c.saveClassifier("classifier_all_200.dump")

def load_classifier(path):
    with open(path) as input:
        c = pickle.load(input)
    return c

# def test_img(c, img_path, scale):
#     results, classes = c.testImage(img_path, scale=scale)
#
#     def maxFunc(p):
#             return p[2][1]
#
#     bestIndex = results.index(max(results, key=maxFunc))
#
#     return (results[bestIndex][0], results[bestIndex][1], 64, 128)

def test_img(c, img_path, scales):
    results = []
    classes = []
    sc = []

    for scale in scales:
        res, cls = c.testImage(img_path, scale=scale)

        results = results + res
        classes = numpy.concatenate((classes, cls))
        for i in range(0, len(res)):
            sc.append(scale)


    def maxFunc(p):
        return p[2][1]

    # print len(results)

    maxRes = max(results, key=maxFunc)

    for i in range(0, len(results)):
        result = results[i]
        if result[0] == maxRes[0] and result[1] == maxRes[1] and (result[2] == maxRes[2]).all():
            bestIndex = i
            break

    if classes[bestIndex] != 1:
        return None

    print "max prob icf = " + str(results[bestIndex][2][1])

    return (results[bestIndex][0], results[bestIndex][1], 64, 128, sc[bestIndex])


def test(scale=0.6):
    c = load_classifier("icf/classifier_230.dump")

    # testImages = utils.getFullImages(
    #     "/home/mataevs/ptz/dumpNew1",
    #     "/home/mataevs/ptz/dumpNew2",
    #     "/home/mataevs/ptz/dumpNew3",
    #     "/home/mataevs/ptz/ns1",
    #     "/home/mataevs/ptz/ns2",
    #     "/home/mataevs/ptz/ns3",
    #     "/home/mataevs/ptz/ns4")

    testImages = utils.getFullImages(
        "/home/mataevs/ptz/INRIAPerson/Test/pos"
    )

    while True:
        imgPath = random.choice(testImages)

        bestWindow = test_img(c, imgPath, scale)

        img = cv2.imread(imgPath)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        utils.draw_detections(img, [bestWindow])

        cv2.imshow("detection", img)
        key = cv2.waitKey(0)

        if key == 27:
            exit(1)