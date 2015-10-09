__author__ = 'mataevs'

from classifier import Classifier
import utils
import pickle
import random
import cv2
import numpy

def train(classifier_out_name, noInitialFeatures, noWantedFeatures, noEstimators):
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
    c.train(posImages, negImages, initialFeatures=noInitialFeatures, wantedFeatures=noWantedFeatures, noEstimators=noEstimators)
    print "Finished training"
    c.saveClassifier(classifier_out_name)

def load_classifier(path):
    with open(path) as input:
        c = pickle.load(input)
    return c

def test_img(c, img_path, scales, allPositive=False, subwindow=None):
    results = []
    classes = []
    sc = []

    for scale in scales:
        res, cls = c.testImage(img_path, scale=scale, subwindow=subwindow)

        results = results + res
        classes = numpy.concatenate((classes, cls))
        for i in range(0, len(res)):
            sc.append(scale)

    def maxFunc(p):
        return p[2][1]
    print results
    maxRes = max(results, key=maxFunc)

    for i in range(0, len(results)):
        result = results[i]
        if result[0] == maxRes[0] and result[1] == maxRes[1] and (result[2] == maxRes[2]).all():
            bestIndex = i
            break

    if classes[bestIndex] != 1:
        return None

    if not allPositive:
        # return the best window and the scale it was detected at
        return (results[bestIndex][0], results[bestIndex][1], 64, 128, sc[bestIndex])
    else:
        # return all positive windows detected at the same scale as the best window
        bestWindowScale = sc[bestIndex]
        bestWindows = []
        for i in range(0, len(classes)):
            if classes[i] == 1 and sc[i] == bestWindowScale:
                bestWindows.append((results[i][0], results[i][1], 64, 128, sc[i]))
        return bestWindows


def test(input_classifier_name, scale=0.6):
    c = load_classifier(input_classifier_name)

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

if __name__ == "__main__":
    train("icf_new_100f_30e.dump", 100, 100, 30)