__author__ = 'mataevs'

import utils

class Checker:
    entries = {}
    def __init__(self, metadataFilePath):

        with open(metadataFilePath, "r") as metaFile:
            lines = metaFile.readlines()

        for line in lines:
            [filePath, sx, sy, ex, ey] = line.split()

            sx, sy, ex, ey = int(sx), int(sy), int(ex), int(ey)

            if sx == -1 or sy == -1 or ex == -1 or ey == -1:
                rect = None
            else:
                rect = (sx, sy, ex-sx, ey-sy)
            self.entries[filePath] = rect

    def getFileList(self):
        return self.entries.keys()

    def checkDetections(self, imgPath, rects):
        legitRect = self.entries[imgPath]

        if rects != None:
            if legitRect == None:
                return [False] * len(rects), True

            trueDetections = False
            results = []

            for rect in rects:
                scale, rect = rect[4], rect[0:4]

                legitRectScaled = tuple(i * scale for i in legitRect)

                if utils.rectOverlap(rect, legitRectScaled):
                    xs1, ys1, xe1, ye1 = rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]
                    xs2, ys2, xe2, ye2 = legitRectScaled[0], legitRectScaled[1], legitRectScaled[0] + legitRectScaled[2], legitRectScaled[1] + legitRectScaled[3]

                    areaIntersection = max(0, max(xe1, xe2) - min(xs1, xs2)) * max(0, max(ye1, ye2) - min(ys1, ys2))
                    areaUnion = rect[2] * rect[3] + legitRectScaled[2] * legitRectScaled[3] - areaIntersection
                    ratio = areaIntersection / areaUnion

                    if ratio < 0.5:
                        results.append(False)
                    else:
                        results.append(True)
                        trueDetections = True
                else:
                    results.append(False)

            return results, trueDetections
        else:
            if legitRect == None:
                return [], True
            return [], False