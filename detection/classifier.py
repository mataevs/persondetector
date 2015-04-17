__author__ = 'mataevs'

from icf import ImageProcessor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import random
from memory_profiler import profile

class Classifier:
    def __init__(self, classifierFile=None, featureCoordsFile=None):
        if classifierFile == None or featureCoordsFile == None:
            self.classifier = None
            self.featureCoords = None
        else:
            self.classifier = pickle.load(open(classifierFile, 'r'))
            self.featureCoords = pickle.load(open(featureCoordsFile, 'r'))

    def train(self, positiveImages, negativeImages, noEstimators=100, noFeatureCoords=500):
        self.featureCoords = []
        for i in range(noFeatureCoords):
            startx = random.randrange(0, 64 - 16, 4)
            starty = random.randrange(0, 128 - 16, 4)
            endx = random.randrange(startx + 16, 64, 4)
            endy = random.randrange(starty + 16, 128, 4)
            self.featureCoords.append((startx, starty, endx, endy))

        positiveFeatures = [self.getFeature(img) for img in positiveImages]
        negativeFeatures = [self.getFeature(img) for img in negativeImages]

        features = positiveFeatures + negativeFeatures
        classes = ([1] * len(positiveImages)) + ([-1] * len(negativeImages))

        self.classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=noEstimators)
        self.classifier.fit(features, classes)

    def testImage(self, image, scale=1):
        imgProcessor = ImageProcessor(image, scale=scale)
        imgProcessor.getIntegralChannels(6)

        windows = []
        for startx in xrange(16, imgProcessor.getWidth() - 64 - 16, 16):
            for starty in xrange(16, imgProcessor.getHeight() - 128 - 16, 16):
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

    def getFeature(self, img):
        print img
        imgProcessor = ImageProcessor(img)
        imgProcessor.getIntegralChannels(6)

        startx = int((imgProcessor.getWidth() - 64) / 2)
        starty = int((imgProcessor.getHeight() - 128) / 2)

        feature = imgProcessor.getFeature(startx, starty, self.featureCoords)

        del imgProcessor

        return feature