__author__ = 'mataevs'

from classifier import Classifier
import utils
import pickle
import random
import cv2

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

def test(scale=0.6):
    with open("classifier.dump") as input:
        c = pickle.load(input)

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

        results, classes = c.testImage(imgPath, scale=scale)

        img = cv2.imread(imgPath)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        # rects = []
        # for i in xrange(0, len(results)):
        #     if classes[i] == 1:
        #         rects.append((results[i][0], results[i][1], 64, 128))
        # draw_detections(img, rects)

        def maxFunc(p):
            return p[2][1]

        bestIndex = results.index(max(results, key=maxFunc))

        print imgPath
        print results[bestIndex]
        print classes[bestIndex]

        draw_detections(img, [(results[bestIndex][0], results[bestIndex][1], 64, 128)])


        cv2.imshow("detection", img)
        key = cv2.waitKey(0)

        if key == 27:
            exit(1)


def draw_detections(img, rects, thickness = 1):
    for i in range(len(rects)):
        x, y, w, h = rects[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness)

test(0.4)