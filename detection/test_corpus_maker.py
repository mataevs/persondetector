__author__ = 'mataevs'

import cv2
import utils
import random

drawing = False
finishedRectangle = False
sx, sy = -1, -1
ex, ey = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global sx, sy, ex, ey, drawing, finishedRectangle

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        sx, sy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img, (sx, sy), (x, y), (0, 255, 0), thickness=1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        finishedRectangle = True
        cv2.rectangle(img, (sx, sy), (x, y), (0, 255, 0), thickness=1)
        ex, ey = x, y

window = cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_rectangle)


def annotateImage(imgPath, annotationFile):
    global img, finishedRectangle
    global sx, sy, ex, ey

    finishedRectangle = False

    img = cv2.imread(imgPath)

    while not finishedRectangle:
        cv2.imshow("image", img)
        k = cv2.waitKey(50) & 0xFF
        if k == 27:
            annotationFile.close()
            exit()
        elif k == ord('n'):
            sx = -1
            sy = -1
            ex = -1
            ey = -1
            break
        elif k == ord('m'):
            return

    annotationFile.write(imgPath + " " + str(sx) + " " + str(sy) + " " + str(ex) + " " + str(ey) + "\n")

def test():
    filepaths = [
        "/home/mataevs/captures/metadata/dump_05_05_01_50",
        "/home/mataevs/captures/metadata/dump_05_06_13_10",
        # "/home/mataevs/captures/metadata/dump_07_05_11_07",
        "/home/mataevs/captures/metadata/dump_10_06_11_47",
        "/home/mataevs/captures/metadata/dump_05_05_01_51",
        "/home/mataevs/captures/metadata/dump_05_06_13_15",
        "/home/mataevs/captures/metadata/dump_07_05_11_40",
        "/home/mataevs/captures/metadata/dump_10_06_11_48",
        "/home/mataevs/captures/metadata/dump_05_05_11_54",
        "/home/mataevs/captures/metadata/dump_05_06_13_20",
        "/home/mataevs/captures/metadata/dump_07_05_11_46",
        "/home/mataevs/captures/metadata/dump_10_06_12_16",
        "/home/mataevs/captures/metadata/dump_05_06_12_57",
        "/home/mataevs/captures/metadata/dump_05_06_13_21",
        "/home/mataevs/captures/metadata/dump_05_06_13_24",
        # "/home/mataevs/captures/metadata/dump_07_05_12_02",
        "/home/mataevs/captures/metadata/dump_16_06_14_57",
        "/home/mataevs/captures/metadata/dump_05_06_13_25",
        "/home/mataevs/captures/metadata/dump_07_05_12_03",
        "/home/mataevs/captures/metadata/dump_16_06_15_26",
        "/home/mataevs/captures/metadata/dump_05_06_13_28",
        "/home/mataevs/captures/metadata/dump_07_05_12_05"
    ]

    testImages = utils.getFullImages(*filepaths)

    with open("annotations.txt", "a") as annotationFile:
        while True:
            imgPath = random.choice(testImages)
            testImages.remove(imgPath)
            annotateImage(imgPath, annotationFile)

test()