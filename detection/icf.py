__author__ = 'mataevs'

import cv2
import numpy
import math


class ImageProcessor:
    def __init__(self, path, scale=1.0):
        self.path = path
        self.bgr = cv2.imread(self.path)

        if scale != 1.0:
            self.bgr = cv2.resize(self.bgr, (0, 0), fx=scale, fy=scale)

    def smooth(self):
        self.bgr = cv2.GaussianBlur(self.bgr, (1, 1), 1)

    def getWidth(self):
        return self.bgr.shape[1]

    def getHeight(self):
        return self.bgr.shape[0]

    def getImage(self):
        return self.bgr

    def getLUVIntegralChannels(self):
        self.luv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2LUV)
        self.l = cv2.integral(self.luv[:, :, 0])
        self.u = cv2.integral(self.luv[:, :, 1])
        self.v = cv2.integral(self.luv[:, :, 2])
        return [self.l, self.u, self.v]

    def getGrayIntegralChannel(self):
        self.gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        self.g = cv2.integral(self.gray)
        return self.g

    def getIntegralHistogram(self, nbins):
        if hasattr(self, 'gray'):
            gray = self.gray
        else:
            gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)

        cv2.equalizeHist(gray, gray)

        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        bins = []
        for i in xrange(0, nbins):
            bins.append(numpy.zeros(gray.shape, dtype=float))

        self.integrals = []
        for i in xrange(0, nbins):
            self.integrals.append(numpy.zeros((gray.shape[0] + 1, gray.shape[1] + 1), float))

        binStep = 180 / nbins

        for y in xrange(0, gray.shape[0]):
            for x in xrange(0, gray.shape[1]):
                if sx[y, x] == 0:
                    temp_gradient = ((math.atan(sy[y, x] / (sx[y, x] + 0.00001))) * (180 / math.pi)) + 90
                else:
                    temp_gradient = ((math.atan(sy[y, x] / sx[y, x])) * (180 / math.pi)) + 90
                temp_magnitude = math.sqrt((sx[y, x] * sx[y, x]) + (sy[y, x] * sy[y, x]))

                for i in xrange(0, nbins):
                    if temp_gradient <= binStep * i:
                        bins[i-1][y, x] = temp_magnitude
                        break

        for i in xrange(0, nbins):
            self.integrals[i] = cv2.integral(bins[i])

        return self.integrals

    def getIntegralChannels(self, nbins):
        luv = self.getLUVIntegralChannels()
        g = self.getGrayIntegralChannel()
        hist = self.getIntegralHistogram(nbins)
        return luv + [g] + hist

    def getFeature(self, startx, starty, featureCoords):
        feature = []

        for coords in featureCoords:
            feature.append(getIntegralValue(self.l, startx + coords[0], starty + coords[1], startx + coords[2], starty + coords[3]))
            feature.append(getIntegralValue(self.u, startx + coords[0], starty + coords[1], startx + coords[2], starty + coords[3]))
            feature.append(getIntegralValue(self.v, startx + coords[0], starty + coords[1], startx + coords[2], starty + coords[3]))
            feature.append(getIntegralValue(self.g, startx + coords[0], starty + coords[1], startx + coords[2], starty + coords[3]))

            for bin in self.integrals:
                feature.append(getIntegralValue(bin, startx + coords[0], starty + coords[1], startx + coords[2], starty + coords[3]))

        return feature

def getIntegralValue(array, sx, sy, ex, ey):
    return array[ey + 1, ex + 1] - array[ey + 1, sx] - array[sy, ex + 1] + array[sy, sx]

