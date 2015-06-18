__author__ = 'mataevs'

from icf import ImageProcessor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import random
import gc
import datetime
import utils
import numpy

class Classifier:
    def __init__(self, classifierFile=None, featureCoordsFile=None):
        if classifierFile == None or featureCoordsFile == None:
            self.classifier = None
            self.featureCoords = None
        else:
            self.classifier = pickle.load(open(classifierFile, 'r'))
            self.featureCoords = pickle.load(open(featureCoordsFile, 'r'))

    def featureSelection(self, positiveImages, negativeImages, noInitialFeatures, noWantedFeatures):
        print "------ Starting feature selection -------"

        initialFeatures = []
        for i in range(noInitialFeatures):
            startx = random.randrange(0, 64 - 16, 4)
            starty = random.randrange(0, 128 - 16, 4)
            endx = random.randrange(startx + 16, 64, 4)
            endy = random.randrange(starty + 16, 128, 4)
            channel = random.randrange(0, 10)
            initialFeatures.append((startx, starty, endx, endy, channel))

        return initialFeatures

        # positiveFeatures = [self.getFeature(img, initialFeatures) for img in positiveImages]
        # negativeFeatures = [self.getFeature(img, initialFeatures) for img in negativeImages]
        # features = positiveFeatures + negativeFeatures
        # classes = ([1] * len(positiveImages)) + ([-1] * len(negativeImages))

        # print "Starting AdaBoost for feature selection", datetime.datetime.now()
        # featureSelector = AdaBoostClassifier(n_estimators=noWantedFeatures)
        # featureSelector.fit(features, classes)
        # print "Finished AdaBoost feature selection", datetime.datetime.now()
        #
        # importances = featureSelector.feature_importances_
        # importances = enumerate(importances)
        # sortedImportances = sorted(importances, key=lambda x:x[1])
        #
        # selectedFeatures = []
        # for i in range(0, noWantedFeatures):
        #     selectedFeatures.append(initialFeatures[sortedImportances[i][0]])
        #
        # with open("selectedFeatures_" + str(noInitialFeatures) + "_" + str(noWantedFeatures) + ".dump", 'w') as output:
        #     pickle.dump(selectedFeatures, output)
        #
        # print "------- Finished feature selection --------"
        #
        # return selectedFeatures

    def train(self, positiveImages, negativeImages, initialFeatures=30000, wantedFeatures=5000, noEstimators=100):
        print "positive images no=", len(positiveImages)
        print "negative images no=", len(negativeImages)

        selectedFeatures = self.featureSelection(positiveImages[:1000], negativeImages[:4000], initialFeatures, wantedFeatures)
        self.featureCoords = selectedFeatures

        gc.collect()

        print "Collecting positive features"
        positiveFeatures = [self.getFeature(img, selectedFeatures) for img in positiveImages]
        print "Collecting negative features"
        negativeFeatures = [self.getFeature(img, selectedFeatures) for img in negativeImages]

        features = positiveFeatures + negativeFeatures
        classes = ([1] * len(positiveImages)) + ([-1] * len(negativeImages))

        print "Starting AdaBoost classification", datetime.datetime.now()
        self.classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=noEstimators)
        self.classifier.fit(features, classes)
        print "Finished AdaBoost classification", datetime.datetime.now()

        print self.classifier.feature_importances_

    def testImage(self, image, scale=1, subwindow=None):
        imgProcessor = ImageProcessor(image, scale=scale)
        imgProcessor.getIntegralChannels(6)

        if subwindow != None:
            nsx, nsy, nw, nh = utils.getDetectionWindow(subwindow, imgProcessor.getWidth(), imgProcessor.getHeight(), scale)
        else:
            nsx, nsy, nw, nh = 0, 0, imgProcessor.getWidth(), imgProcessor.getHeight()

        windows = []
        for startx in xrange(nsx, nsx + nw - 64, 16):
            for starty in xrange(nsy, nsy + nh - 128, 16):
                windows.append((startx, starty, imgProcessor.getFeature(startx, starty, self.featureCoords)))

        probs = self.classifier.predict_proba([w[2] for w in windows])
        classes = self.classifier.predict([w[2] for w in windows])

        results = []
        for i in xrange(0, len(windows)):
            results.append((windows[i][0], windows[i][1], probs[i]))
        return results, classes

    def saveClassifier(self, classifierFile):
        with open(classifierFile, 'w') as output:
            pickle.dump(self, output)

    def getFeature(self, img, featureCoords):
        imgProcessor = ImageProcessor(img)
        imgProcessor.getIntegralChannels(6)

        # needed for centering the 64x128 image
        startx = int((imgProcessor.getWidth() - 64) / 2)
        starty = int((imgProcessor.getHeight() - 128) / 2)

        feature = imgProcessor.getFeature(startx, starty, featureCoords)

        del imgProcessor

        return feature

class CascadeClassifier:
    classifiers = []

    def __init__(self, classifierList):
        for classifier in classifierList:
            self.classifiers.append(classifier)

    def testImage(self, image, scale, subwindow=None):
        imgProcessor = ImageProcessor(image, scale=scale)
        imgProcessor.getIntegralChannels(6)

        if subwindow != None:
            nsx, nsy, nw, nh = utils.getDetectionWindow(subwindow, imgProcessor.getWidth(), imgProcessor.getHeight(), scale)
        else:
            nsx, nsy, nw, nh = 0, 0, imgProcessor.getWidth(), imgProcessor.getHeight()

        results = []
        classes = []
        for startx in xrange(nsx, nsx + nw - 64, 16):
            for starty in xrange(nsy, nsy + nh - 128, 16):
                c = None
                feature = None
                for classifier in self.classifiers:
                    feature = [imgProcessor.getFeature(startx, starty, classifier.featureCoords)]
                    c = classifier.classifier.predict(feature)
                    if c == -1:
                        results.append((startx, starty, classifier.classifier.predict_proba(feature)[0]))
                        break
                if c == 1:
                    results.append((startx, starty, self.classifiers[-1].classifier.predict_proba(feature)[0]))
                classes.append(c)

        return results, classes