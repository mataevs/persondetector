__author__ = 'mataevs'

from icf import ImageProcessor

import os.path as path
import random


class InriaLoader:
    def __init__(self, dirPath, featureCoordsNo=None):
        self.dirPath = dirPath
        random.seed()

        if featureCoordsNo != None:
            self.featureCoords = []
            for i in range(featureCoordsNo):
                startx = random.randrange(0, 64 - 16, 4)
                starty = random.randrange(0, 128 - 16, 4)
                endx = random.randrange(startx + 16, 64, 4)
                endy = random.randrange(starty + 16, 128, 4)
                self.featureCoords.append((startx, starty, endx, endy))

    def setFeatureCoords(self, featureCoords):
        self.featureCoords = featureCoords

    def getFeatureCoords(self):
        return self.featureCoords

    def getpositivetrainlist(self):
        with open(path.join(self.dirPath, "train/pos.lst")) as f:
            poslist = f.readlines()
        for i in xrange(0, len(poslist)):
            poslist[i] = poslist[i].rstrip("\n")
        return poslist

    def getnegativetrainlist(self):
        with open(path.join(self.dirPath, "train/neg.lst")) as f:
            neglist = f.readlines()
        for i in xrange(0, len(neglist)):
            neglist[i] = neglist[i].rstrip("\n")
        return neglist

    def getpositivetestlist(self):
        with open(path.join(self.dirPath, "test/pos.lst")) as f:
            poslist = f.readlines()
        for i in xrange(0, len(poslist)):
            poslist[i] = poslist[i].rstrip("\n")
        return poslist

    def getnegativetestlist(self):
        with open(path.join(self.dirPath, "test/neg.lst")) as f:
            neglist = f.readlines()
        for i in xrange(0, len(neglist)):
            neglist[i] = neglist[i].rstrip("\n")
        return neglist

    def getImgFeature(self, imgPath, test=False):
        imgProc = ImageProcessor(path.realpath(path.join(self.dirPath, imgPath)))
        imgProc.getIntegralChannels(6)
        if test:
            return imgProc.getFeature(3, 3, self.featureCoords)
        return imgProc.getFeature(16, 16, self.featureCoords)

    def getImgFeatures(self, imgPath, noSamples=1, test=False):
        imgProc = ImageProcessor(path.realpath(path.join(self.dirPath, imgPath)))
        imgProc.getIntegralChannels(6)

        features = []
        for i in xrange(0, noSamples):
            if test:
                startx = random.randrange(3, imgProc.getWidth() - 3 - 64, 16)
                starty = random.randrange(3, imgProc.getHeight() - 3 - 128, 16)
            else:
                startx = random.randrange(3, imgProc.getWidth() - 3 - 64, 16)
                starty = random.randrange(3, imgProc.getHeight() - 3 - 128, 16)

            features.append(imgProc.getFeature(startx, starty, self.featureCoords))
        return features

    def getpositivedataset(self):
        posset = []
        for img in self.getpositivetrainlist():
            print img
            posset.append(self.getImgFeature(img))
        poslabel = []
        for i in xrange(0, len(posset)):
            poslabel.append(1)
        return posset, poslabel

    def getnegativedataset(self, noSamples=1):
        negset = []
        for img in self.getnegativetrainlist():
            print img
            negset.extend(self.getImgFeatures(img, noSamples))
        neglabel = []
        for i in xrange(0, len(negset)):
            neglabel.append(-1)
        return negset, neglabel

    def getpositivetestset(self):
        posset = []
        for img in self.getpositivetestlist():
            print img
            posset.append(self.getImgFeature(img, True))
        poslabel = []
        for i in xrange(0, len(posset)):
            poslabel.append(1)
        return posset, poslabel

    def getnegativetestset(self):
        negset = []
        for img in self.getnegativetestlist():
            print img
            negset.extend(self.getImgFeatures(img, 1, True))
        neglabel = []
        for i in xrange(0, len(negset)):
            neglabel.append(-1)
        return negset, neglabel

    def gettraindataset(self, noNegSamplesPerImg):
        posset, poslabel = self.getpositivedataset()
        negset, neglabel = self.getnegativedataset(noNegSamplesPerImg)
        features = posset + negset
        labels = poslabel + neglabel
        return features, labels

    def gettestdataset(self):
        posset, poslabel = self.getpositivetestset()
        negset, neglabel = self.getnegativetestset()
        features = posset + negset
        labels = poslabel + neglabel
        return features, labels