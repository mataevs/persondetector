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

    def getWindowsClasses(self, imgPath, rects):
        legitRect = self.entries[imgPath]

        if rects != None:
            if legitRect == None:
                return [-1] * len(rects)

            xs2, ys2, xe2, ye2 = legitRect[0], legitRect[1], legitRect[0] + legitRect[2], legitRect[1] + legitRect[3]

            results = []

            for rect in rects:
                if utils.rectOverlap(rect, legitRect):
                    xs1, ys1, xe1, ye1 = rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]

                    areaIntersection = max(0, max(xe1, xe2) - min(xs1, xs2)) * max(0, max(ye1, ye2) - min(ys1, ys2))
                    areaUnion = rect[2] * rect[3] + legitRect[2] * legitRect[3] - areaIntersection
                    ratio = areaIntersection / areaUnion

                    if ratio < 0.5:
                        results.append(-1)
                    else:
                        results.append(1)
                else:
                    results.append(-1)
            return results
        return []

    def checkDetections(self, imgPath, rects):
        legitRect = self.entries[imgPath]

        if rects != None:
            if legitRect == None:
                return [False] * len(rects), True

            xs2, ys2, xe2, ye2 = legitRect[0], legitRect[1], legitRect[0] + legitRect[2], legitRect[1] + legitRect[3]

            trueDetections = False
            results = []

            for rect in rects:
                if utils.rectOverlap(rect, legitRect):
                    xs1, ys1, xe1, ye1 = rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]

                    areaIntersection = max(0, max(xe1, xe2) - min(xs1, xs2)) * max(0, max(ye1, ye2) - min(ys1, ys2))
                    areaUnion = rect[2] * rect[3] + legitRect[2] * legitRect[3] - areaIntersection
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