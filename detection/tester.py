__author__ = 'mataevs'

from os import listdir
from os.path import isfile, join
from PIL import Image
from icf import ImageProcessor
import classifier
import cv2
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import random


def getValidImagesList(dir):
    files = [ join(dir, f) for f in listdir(dir) if isfile(join(dir, f)) and (Image.open(join(dir, f)).size == (64, 128)) ]
    return files

def getImgProcessor(imgPath, scale=1.0):
    imgProc = ImageProcessor(imgPath, scale)
    imgProc.getIntegralChannels(6)
    return imgProc

def getImgFeature(imgProc, featureCoords, startx=0, starty=0):
    return imgProc.getFeature(startx, starty, featureCoords)

def gettestset(dir, label, featureCoords):
    set = []
    for img in getValidImagesList(dir):
        set.append(getImgFeature(getImgProcessor(img), featureCoords))
    labels = []
    for i in xrange(0, len(set)):
        labels.append(label)
    return set, labels

def draw_detections(img, rects, thickness = 1):
    for i in range(len(rects)):
        x, y, w, h = rects[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness)


def train(posDir, negDir, noEstimators, featureCoordsInput, classifierOutput):
    featureCoords = pickle.load(open(featureCoordsInput, 'r'))

    posset, cp = gettestset(posDir, 1, featureCoords)
    negset, cn = gettestset(negDir, -1, featureCoords)

    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=noEstimators)

    features = posset + negset
    classes = cp + cn

    print "Starting classification\n"
    classifier.fit(features, classes)

    with open(classifierOutput, 'w') as output:
        pickle.dump(classifier, output)

    print "Finished classification\n"

    return classifier, featureCoords

def test(posDir, negDir, classifierInput, featureCoordsInput):
    cls, featureCoords = classifier.loadClassifier(classifierInput, featureCoordsInput)

    print "Loading positive test images"
    posset, cp = gettestset(posDir, 1, featureCoords)

    print "Loading negative test images"
    negset, cn = gettestset(negDir, -1, featureCoords)

    print "Positive test images:" + str(len(posset))
    print "Positive test images accuracy: " + str(cls.score(posset, cp))

    print "Negative test images:" + str(len(negset))
    print "Negative test images accuracy: " + str(cls.score(negset, cn))

def testImage(imgPath, classifier, featureCoords):

    scale = 1.1
    while scale >= 0.6:
        scale -= 0.1
        imgProcessor = getImgProcessor(imgPath, scale)

        windows = []
        for startx in range(0, imgProcessor.getWidth() - 64, 16):
            for starty in range(0, imgProcessor.getHeight() - 128, 16):
                windows.append((startx, starty, getImgFeature(imgProcessor, featureCoords, startx, starty)))

        img = imgProcessor.getImage()

        classes = classifier.predict([w[2] for w in windows])

        rects = []
        for i in xrange(0, len(classes)):
            if classes[i] == 1:
                rects.append((windows[i][0], windows[i][1], 64, 128))
        print rects
        draw_detections(img, rects)

        cv2.imshow("detections", img)
        key = cv2.waitKey(0)
        if key == 27:
            exit(1)
        elif key == ord('s'):
            return

def testRandomImages(files, classifier, featureCoords):
    while True:
        file = random.choice(files)

        testImage(file, classifier, featureCoords)


def getFullImages(*folders):
    files = [ join(dir, f) for dir in folders for f in listdir(dir) if isfile(join(dir, f)) ]
    return files

images = getFullImages(
    "/home/mataevs/ptz/dumpNew1",
    "/home/mataevs/ptz/dumpNew2",
    "/home/mataevs/ptz/dumpNew3",
    "/home/mataevs/ptz/ns1",
    "/home/mataevs/ptz/ns2",
    "/home/mataevs/ptz/ns3",
    "/home/mataevs/ptz/ns4")

# classifier, featureCoords = train("/home/mataevs/ptz/positive", "/home/mataevs/ptz/negative", 200, "features.txt", "classifierMySet.tmp")

# test("/home/mataevs/ptz/positive", "/home/mataevs/ptz/negative", "adaboost.tmp", "features.txt")

classifier, featureCoords = classifier.loadClassifier("classifierMySet.tmp", "features.txt")

testRandomImages(images, classifier, featureCoords)