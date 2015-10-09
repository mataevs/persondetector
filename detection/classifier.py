__author__ = 'mataevs'

from icf import ImageProcessor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import cPickle
import pickle
import random
import gc
import datetime
import utils
import numpy
import math

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
    featureCoords = []

    def __init__(self, featuresFilePath, noFeatures=0):
        if noFeatures == 0:
            with open(featuresFilePath) as f:
                featureList = pickle.load(f)
                self.featureCoords = featureList
        else:
            self.featureCoords = self.featureGeneration(noFeatures)

    def saveClassifier(self, classifierFile):
        with open(classifierFile + ".cls", "w") as output:
            pickle.dump(self.classifiers, output)
        with open(classifierFile, 'w') as output:
            pickle.dump(self, output)

    def getFeature(self, img):
        imgProcessor = ImageProcessor(img)
        imgProcessor.getIntegralChannels(6)

        # needed for centering the 64x128 image
        startx = int((imgProcessor.getWidth() - 64) / 2)
        starty = int((imgProcessor.getHeight() - 128) / 2)

        feature = imgProcessor.getFeature(startx, starty, self.featureCoords)

        del imgProcessor

        return feature

    def featureGeneration(self, noFeatures):
        print "------ Starting feature generation -------"

        features = []
        for i in range(noFeatures):
            startx = random.randrange(0, 64 - 4, 4)
            starty = random.randrange(0, 128 - 4, 4)
            endx = random.randrange(startx + 5, 64, 4)
            endy = random.randrange(starty + 5, 128, 4)
            channel = random.randrange(0, 10)
            features.append((startx, starty, endx, endy, channel))

        print "Saving features"
        p = cPickle.Pickler(open("generatedFeatures_" + str(noFeatures) + ".dump", "wb"), protocol=2)
        p.fast = True
        p.dump(features)

        return features

    def generateAndSaveFeatures(self, positiveImages, negativeImages):
        # print "Collecting positive features"
        # positiveFeatures = [self.getFeature(img) for img in positiveImages]
        # p = cPickle.Pickler(open("positive_features_5000.dump", "wb"), protocol=2)
        # p.fast = True
        # p.dump(positiveFeatures)

        print "Collecting negative features"
        negativeFeatures = [self.getFeature(img) for img in negativeImages]
        p = cPickle.Pickler(open("negative_features_5000.dump", "wb"), protocol=2)
        p.fast = True
        p.dump(negativeFeatures)

    def train(self, no_steps, no_weak):
        positiveFeatures = cPickle.load(open("positive_features_5000.dump", "rb"))
        negativeFeatures = cPickle.load(open("negative_features_5000.dump", "rb"))

        features = positiveFeatures + negativeFeatures
        classes = ([1] * len(positiveFeatures)) + ([-1] * len(negativeFeatures))

        print "Starting AdaBoost classification", datetime.datetime.now()

        for step in range(0, no_steps):
            print "Training classifier", step, "no_estimators=", no_weak[step]
            self.classifiers.append(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=no_weak[step]))
            print len(self.classifiers)
            self.classifiers[step].fit(features, classes)
        print "Finished AdaBoost classification", datetime.datetime.now()


    def getWindowsAndDescriptors(self, image, scale, subwindow=None):
        imgProcessor = ImageProcessor(image, scale=scale)
        imgProcessor.getIntegralChannels(6)

        if subwindow != None:
            nsx, nsy, nw, nh = utils.getDetectionWindow(subwindow, imgProcessor.getWidth(), imgProcessor.getHeight(), scale)
        else:
            nsx, nsy, nw, nh = 0, 0, imgProcessor.getWidth(), imgProcessor.getHeight()

        windows = []
        for startx in xrange(nsx, nsx + nw - 64, 16):
            for starty in xrange(nsy, nsy + nh - 128, 16):
                feature = imgProcessor.getFeature(startx, starty, self.featureCoords)
                windows.append([(startx, starty, 64, 128), feature])

        return windows

    def testImage(self, image, scale, subwindow=None):
        windows = self.getWindowsAndDescriptors(image, scale, subwindow=subwindow)

        results = []
        classes = []

        for win in windows:
            c = None
            x, y, w, h = win[0]
            feature = win[1]

            for classifier in self.classifiers[1:2]:
                c = classifier.predict(feature)
                if c == -1:
                    break
                results.append((x, y))
                classes.append(c)

        return results, classes

    def getDecisionFunctions(self, windows):
        # df = []
        #
        # print windows[0]
        #
        features = [w[1] for w in windows]
        #
        # print len(features)
        # print len(features[0])
        #
        # for c in self.classifiers:
        #     df.append(c.decision_function(features))
        # return df
        return self.classifiers[1].decision_function(features)


def loadCascadeClassifier(filePath):
    with open(filePath) as f:
        classifier = pickle.load(f)
    with open(filePath + ".cls") as f:
        classifier.classifiers = pickle.load(f)
    return classifier


class SoftCascadeClassifier:
    classifier = None
    featureCoords = []

    def __init__(self, classifierFilePath, new=False):
        if not new:
            with open(classifierFilePath) as f:
                self.classifier = pickle.load(f)

    def saveClassifier(self, classifierFile):
        with open(classifierFile + ".cls", "w") as output:
            pickle.dump(self.classifier, output)
        with open(classifierFile, 'w') as output:
            pickle.dump(self, output)

    def getFeature(self, img):
        imgProcessor = ImageProcessor(img)
        imgProcessor.getIntegralChannels(6)

        # needed for centering the 64x128 image
        startx = int((imgProcessor.getWidth() - 64) / 2)
        starty = int((imgProcessor.getHeight() - 128) / 2)

        feature = imgProcessor.getFeature(startx, starty, self.featureCoords)

        return feature

    def featureGeneration(self, noFeatures):
        print "------ Starting feature generation -------"

        features = []
        for i in range(noFeatures):
            startx = random.randrange(0, 64 - 4, 4)
            starty = random.randrange(0, 128 - 4, 4)
            endx = random.randrange(startx + 5, 64, 4)
            endy = random.randrange(starty + 5, 128, 4)
            channel = random.randrange(0, 10)
            features.append((startx, starty, endx, endy, channel))

        print "Saving features"
        p = cPickle.Pickler(open("generatedFeatures_" + str(noFeatures) + ".dump", "wb"), protocol=2)
        p.fast = True
        p.dump(features)

        return features

    def generateAndSaveFeatures(self, positiveImages, negativeImages):
        print "Collecting positive features"
        positiveFeatures = [self.getFeature(img) for img in positiveImages]
        p = cPickle.Pickler(open("positive_features_5000.dump", "wb"), protocol=2)
        p.fast = True
        p.dump(positiveFeatures)

        print "Collecting negative features"
        negativeFeatures = [self.getFeature(img) for img in negativeImages]
        p = cPickle.Pickler(open("negative_features_5000.dump", "wb"), protocol=2)
        p.fast = True
        p.dump(negativeFeatures)

    def train(self, no_estimators):
        positiveFeatures = cPickle.load(open("positive_features_5000.dump", "rb"))
        negativeFeatures = cPickle.load(open("negative_features_5000.dump", "rb"))

        features = positiveFeatures + negativeFeatures
        classes = ([1] * len(positiveFeatures)) + ([-1] * len(negativeFeatures))

        print "Starting AdaBoost classification", datetime.datetime.now()

        self.classifier = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=no_estimators,
            algorithm='SAMME')
        self.classifier.fit(features, classes)

        print "Finished AdaBoost classification", datetime.datetime.now()


    def getWindowsAndDescriptors(self, image, scale, subwindow=None):
        imgProcessor = ImageProcessor(image, scale=scale)
        imgProcessor.getIntegralChannels(6)

        if subwindow != None:
            nsx, nsy, nw, nh = utils.getDetectionWindow(subwindow, imgProcessor.getWidth(), imgProcessor.getHeight(), scale)
        else:
            nsx, nsy, nw, nh = 0, 0, imgProcessor.getWidth(), imgProcessor.getHeight()

        windows = []
        for startx in xrange(nsx, nsx + nw - 64, 16):
            for starty in xrange(nsy, nsy + nh - 128, 16):
                feature = imgProcessor.getFeature(startx, starty, self.featureCoords)
                windows.append([(startx, starty, 64, 128), feature])

        return windows

    def testImage(self, image, scale, subwindow=None, threshold=0):
        windows = self.getWindowsAndDescriptors(image, scale, subwindow=subwindow)

        results = []
        classes = []

        for win in windows:
            x, y, w, h = win[0]
            feature = win[1]

            c = None
            i, current = 2, 0
            tempEstimation = 0.0
            weights = 0.0
            while current < len(self.classifier.estimators_):
                while current < min(i, len(self.classifier.estimators_)):
                    tempEstimation += \
                        self.classifier.estimator_weights_[current] *\
                        self.classifier.estimators_[current].predict(feature)[0]
                    weights += self.classifier.estimator_weights_[current]
                    current += 1

                if tempEstimation < threshold:
                    c = -1
                    break
                i *= 2
            if c == None:
                if tempEstimation <= 0:
                    c = -1
                else:
                    c = 1
            results.append((x, y))
            classes.append(c)

        return results, classes

    def getDecisionFunctions(self, windows, threshold=0):
        df = numpy.zeros(len(windows))

        noEstimators = len(self.classifier.estimators_)

        df_index = 0
        for win in windows:
            feature = win[1]

            i, current = 2, 0
            tempEstimation = 0.0
            weights = 0.0
            while current < noEstimators:
                while current < min(i, noEstimators):
                    tempEstimation += \
                        self.classifier.estimator_weights_[current] *\
                        self.classifier.estimators_[current].predict(feature)[0]
                    weights += self.classifier.estimator_weights_[current]
                    current += 1
                if tempEstimation < threshold:
                    break
                i *= 2

            val = tempEstimation / weights - 0.1 * (noEstimators - current)
            df[df_index] = 1 / (1 + (math.exp(-val)))
            # if current < len(self.classifier.estimators_):
            #     df[df_index] = -1
            # else:
            #     df[df_index] = 1 / (1 + (math.exp(-val))) - 0.5
                # if val < 0:
                #     df[df_index] = -1 / (1 + (math.exp(-val)))
                # else:
                #     df[df_index] = 1 / (1 + (math.exp(-val)))

            # if current == len(self.classifier.estimators_):
            #     df[df_index] = 0.5 + 1 / (1 + math.exp(val))
            # else:
            #     df[df_index] = 0

            df_index += 1

        # cdf = self.classifier.decision_function([w[1] for w in windows])
        # print "adaboost", cdf

        return df


def loadCascadeClassifier(filePath):
    with open(filePath) as f:
        classifier = pickle.load(f)
    with open(filePath + ".cls") as f:
        classifier.classifiers = pickle.load(f)
    return classifier

def loadSoftCascadeClassifier(filePath):
    print "load soft cascade classifier"
    with open(filePath)as f:
        classifier = pickle.load(f)
    with open(filePath + ".cls") as f:
        classifier.classifier = pickle.load(f)
    return classifier

if __name__ == "__main__":
    # posImages = utils.getFullImages(
    #     "/home/mataevs/ptz/INRIAPerson/train/pos",
    #     "/home/mataevs/ptz/positive")
    #
    # posImages = utils.randomize(posImages)
    # posImages = posImages
    #
    # negImages = utils.getFullImages(
    #     "/home/mataevs/ptz/INRIAPerson/train/neg",
    #     "/home/mataevs/ptz/negative")
    # negImages = utils.randomize(negImages)
    # negImages = negImages
    #
    # cascade = CascadeClassifier("generatedFeatures_5000.dump")
    # # cascade.generateAndSaveFeatures(posImages, negImages)
    # cascade.train(3, [10, 50, 100])
    #
    # cascade.saveClassifier("cascade_classifier_10_50_100.dump")

    softCascade = loadSoftCascadeClassifier("soft_cascade.dump")
    softCascade.train(2048)
    softCascade.saveClassifier("soft_cascade.dump")