__author__ = 'mataevs'

import cv2
import numpy
import math


class ImageProcessor:
    def __init__(self, path, scale=1.0):
        self.path = path
        self.bgr = cv2.imread(self.path)
        self.luv = None
        self.l = None
        self.u = None
        self.v = None
        self.magnitudes = None
        self.integrals = []

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

    # def getGrayIntegralChannel(self):
    #     self.gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
    #     self.g = cv2.integral(self.gray)
    #     return self.g

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

        magnitudes = numpy.zeros(gray.shape, dtype=float)

        for i in xrange(0, nbins):
            self.integrals.append(numpy.zeros((gray.shape[0] + 1, gray.shape[1] + 1), float))

        binStep = 180 / nbins

        for y in xrange(0, gray.shape[0]):
            for x in xrange(0, gray.shape[1]):
                if sx[y, x] == 0:
                    temp_gradient = ((math.atan(sy[y, x] / (sx[y, x] + 0.00001))) * (180 / math.pi)) + 90
                else:
                    temp_gradient = ((math.atan(sy[y, x] / sx[y, x])) * (180 / math.pi)) + 90
                magnitudes[y, x] = math.sqrt((sx[y, x] * sx[y, x]) + (sy[y, x] * sy[y, x]))

                for i in xrange(0, nbins):
                    if temp_gradient <= binStep * i:
                        bins[i-1][y, x] = magnitudes[y, x]
                        break

        for i in xrange(0, nbins):
            self.integrals[i] = cv2.integral(bins[i])
        self.magnitudes = cv2.integral(magnitudes)

        # todo - create and return integral image for gradient magnitude

        return self.integrals, self.magnitudes

    def getIntegralChannels(self, nbins):
        luv = self.getLUVIntegralChannels()
        # g = self.getGrayIntegralChannel()
        hist, magnitudes = self.getIntegralHistogram(nbins)
        return luv + [magnitudes] + hist

    def getFeature(self, startx, starty, featureCoords):
        feature = []

        for coords in featureCoords:
            rsx, rsy, rex, rey, channelNo = coords

            if channelNo == 0:
                channel = self.l
            elif channelNo == 1:
                channel = self.u
            elif channelNo == 2:
                channel = self.v
            elif channelNo == 3:
                channel = self.magnitudes
            else:
                channel = self.integrals[channelNo-4]
            feature.append(getIntegralValue(channel, startx+rsx, starty+rsy, startx+rex, starty+rey))

        return feature

def getIntegralValue(array, sx, sy, ex, ey):
    return array[ey + 1, ex + 1] - array[ey + 1, sx] - array[sy, ex + 1] + array[sy, sx]

