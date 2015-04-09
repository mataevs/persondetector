__author__ = 'mataevs'

from inria import InriaLoader
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle


def trainClassifier(datasetDir, noFeatures, noNegSamplesPerImg, noEstimators, classifierOutput, featureCoordsOutput):
    inria = InriaLoader(datasetDir, noFeatures)

    features, classes = inria.gettraindataset(noNegSamplesPerImg)

    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=noEstimators)

    print "Starting classification\n"
    classifier.fit(features, classes)

    print "Starting object serialization"
    with open(classifierOutput, 'w') as output:
        pickle.dump(classifier, output)

    with open(featureCoordsOutput, 'w') as output:
        pickle.dump(inria.getFeatureCoords(), output)

def loadClassifier(classifierInput, featureCoordsInput):
    print "Loading classifier"

    classifier = pickle.load(open(classifierInput, 'r'))
    featureCoords = pickle.load(open(featureCoordsInput, 'r'))

    return classifier, featureCoords


def testInria(datasetDir, classifierInput, featureCoordsInput):
    classifier, featureCoords = loadClassifier(classifierInput, featureCoordsInput)

    inria = InriaLoader(datasetDir)
    inria.setFeatureCoords(featureCoords)

    features, classes = inria.getpositivetestset()

    print "Starting classifier test"
    score = classifier.score(features, classes)

    print score

# trainClassifier("/home/mataevs/ptz/INRIAPerson", 500, 5, 100, "adaboost.tmp", "features.txt")

# loadClassifier("/home/mataevs/ptz/INRIAPerson", "adaboost.tmp", "features.txt")
