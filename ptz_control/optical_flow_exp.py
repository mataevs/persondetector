'''
Script that displays frames from a PTZ camera (Samsung SNP-3120V)
'''

import ptz_acq
import ptz_control
import Tkinter as tk
import ImageTk
from StreamViewer import StreamViewer
import os
import datetime
from collections import deque
import threading
import cv2
import numpy as np
import optical_flow
import math
import time

save_experiment = True

stop_moving = 0

fixed_positions = [(155, 23), (140, 38), (140, 53), (230, 73), (310, 38), (275, 33)]

def tiltMove(p, t, step):
    global stop_moving
    t = (t + step) % 180
    if t > 90:
        stop_moving = 3
        p = (pan + 180) % 360
        t = 90 - (t - 90)
    return p, t

def directionImportances(flowx, flowy):
    dirs = [None] * 4
    if abs(flowx) > abs(flowy):
        ximp1, yimp1, ximp2, yimp2 = 0, 1, 3, 2
    else:
        ximp1, yimp1, ximp2, yimp2 = 1, 0, 2, 3

    if flowx > 0:
        dirs[ximp1], dirs[ximp2] = 'left', 'right'
    else:
        dirs[ximp1], dirs[ximp2] = 'right', 'left'

    if flowy > 0:
        dirs[yimp1], dirs[yimp2] = 'up', 'down'
    else:
        dirs[yimp1], dirs[yimp2] = 'down', 'up'
    return dirs

def moveCamera(directions, givenStep=4, flow=(0, 0)):
    global pan, tilt, zoom
    global stop_moving

    p, t, z = pan, tilt, zoom

    flowx, flowy = flow

    # flowDirections = directionImportances(flowx, flowy)[:2]

    # If in cone, move only one direction
    if t > 60:
        bestDirections = directionImportances(flow[0], flow[1])

        print 'bestDirections', bestDirections
        print 'receivedDirections', directions

        for dir in bestDirections:
            if dir in directions:
                directions = [dir]
                break

    print directions

    print "move from: pan=", p, "tilt=", t
    for direction in directions:
        if givenStep == 'dynamic':
            if tilt < 40:
                step = 10
            elif tilt < 60:
                step = 12
            elif tilt < 70:
                step = 15
            else:
                stop_moving = 1
                step = 20
        else:
            step = givenStep

        stepx, stepy = step, step

        # if direction in flowDirections:
        #     if abs(flowx) < 5:
        #         stepx *= 0.7
        #     elif abs(flowx) < 10:
        #         stepx = stepx
        #     else:
        #         stepx *= 1.3
        #
        #     if abs(flowy) < 5:
        #         stepy *= 0.7
        #     elif abs(flowy) < 10:
        #         stepy = stepy
        #     else:
        #         stepy *= 1.3
        # else:
        #     stepx *= 0.5
        #     stepy *= 0.5
        #
        # print "stepx=",stepx," stepy=",stepy

        if direction == 'left':
            if t > 75:
                p, t = tiltMove(p, t, -stepx)
                p = (p - 90) % 360
                stop_moving = 2
            else:
                p = (p - stepx) % 360
        if direction == 'right':
            if t > 75:
                p, t = tiltMove(p, t, -stepx)
                p = (p + 90) % 360
                stop_moving = 2
            else:
                p = (p + stepx) % 360
        if direction == 'up':
            p, t = tiltMove(p, t, -stepy)
        if direction == 'down':
            p, t = tiltMove(p, t, +stepy)

    # if t > 80:
    #     stop_moving = True

    print "move to: pan=", p, "tilt=", t

    queueCommand(p, t, z)

def keystroke(event):
    global pan, tilt, zoom

    global position
    global fixed_positions
    global stop_moving

    print event.keysym, event.keycode

    p, t, z = pan, tilt, zoom

    if event.keycode == 57: # key 'n'
        position = (position + 1) % len(fixed_positions)
        p, t = fixed_positions[position]
        queueCommand(p, t, z)
    elif event.keycode == 33: # key 'p' ?
        position = (position - 1) % len(fixed_positions)
        p, t = fixed_positions[position]
        queueCommand(p, t, z)
    elif event.keycode == 10: # key 1
        p = 200
        t = 58
        z = 1
        queueCommand(p, t, z)
    elif event.keycode == 11: # key 2
        p = 135
        t = 55
        z = 1
        queueCommand(p, t, z)
    elif event.keycode == 14: # key 5
        p = 300
        t = 48
        z = 1
        queueCommand(p, t, z)
    if event.keycode == 113:
        moveCamera(['left'], givenStep=5)
    elif event.keycode == 114:
        moveCamera(['right'], givenStep=5)
    elif event.keycode == 111:
        moveCamera(['up'], givenStep=5)
    elif event.keycode == 116:
        moveCamera(['down'], givenStep=5)
    elif event.keycode == 38:
        z += 1
        queueCommand(p, t, z)
    elif event.keycode == 52:
        z -= 1
        queueCommand(p, t, z)
    elif event.keycode == 9:
        if save_experiment:
            metadataFile.close()
        exit()
    elif event.keycode == 27:
        stop_moving = False

# Initialize viewer
root = tk.Tk()
root.bind('<Key>', keystroke)
root.geometry("%dx%d+0+0" % (640, 480))
root.resizable(False, False)
tbk = StreamViewer(root)

# Initialize camera pan and tilt
pan, tilt = fixed_positions[0]
position = 0
zoom = 1
ptz_control.setPTZ(pan, tilt, zoom)

# Create dump folder
folderName = "dump_" + datetime.datetime.now().strftime("%d_%m_%H_%M")
if not os.path.exists(folderName) and save_experiment:
    os.makedirs(folderName)

# Create metadata text file
if save_experiment:
    metadataFile = open(folderName + ".txt", "w")

# Create log file
log = open("movements.log", "w")

# Create deque of images
d = deque(maxlen=10)
# Create condition
nonEmptyCondition = threading.Condition()
accessSemaphore = threading.Semaphore(1)

# Create getImages thread
def getImages(d, nonEmptyCondition, accessSemaphore):
    for img, imgBase64 in ptz_acq.getFrame(enableViewer=True):
        accessSemaphore.acquire()
        d.append(img)
        with nonEmptyCondition:
            nonEmptyCondition.notifyAll()
        accessSemaphore.release()

getterThread = threading.Thread(target=getImages, args=(d, nonEmptyCondition, accessSemaphore))
getterThread.setDaemon(True)
getterThread.start()


# Create deque of moving commands
commands = deque(maxlen=10)
# Create commands condition
nonEmptyConditionCommands = threading.Condition()

# Create sendCommands thread
def sendCommands(commands, nonEmptyConditionCommands):
    global pan, tilt, zoom
    global frameNo

    while True:
        with nonEmptyConditionCommands:
            nonEmptyConditionCommands.wait()

        print 'received command'
        coords = commands.pop()
        p, t, z = coords
        ptz_control.setPTZ(p, t, z)
        log.write("frameNo=" + str(frameNo) + " pan=" + str(p) + " tilt=" + str(t) + "\n")

        pan, tilt, zoom = p, t, z
        time.sleep(0.3)

def queueCommand(p, t, z):
    global commands
    global nonEmptyConditionCommands

    commands.append((p, t, z))
    print 'appended command'
    with nonEmptyConditionCommands:
        nonEmptyConditionCommands.notifyAll()

commandSenderThread = threading.Thread(target=sendCommands, args=(commands, nonEmptyConditionCommands))
commandSenderThread.setDaemon(True)
commandSenderThread.start()

frameNo = 0

prevImg = None
currentImg = None

cv2.namedWindow('img')
cv2.moveWindow('img', 1000, 200)

prevCentroid = None
framesToDrop = 0

while True:

    while len(d) < 2:
        with nonEmptyCondition:
            nonEmptyCondition.wait()

    accessSemaphore.acquire()
    img = d.pop()
    pimg = d.pop()
    accessSemaphore.release()

    frameName = "%04d" % (frameNo,)
    frameNo += 1

    currentImg = np.array(img)
    currentImg = currentImg[:, :, ::-1]

    prevImg = np.array(pimg)
    prevImg = prevImg[:, :, ::-1]

    if prevImg != None:
        res_img, flow_rgb, flow_blobs, centroid, flow, move_dirs = optical_flow.optical_flow(currentImg, prevImg)

        moved = False
        if len(move_dirs) != 0 and centroid != None and stop_moving == 0:
            if prevCentroid != None:
                x_dist = prevCentroid[0] - centroid[0]
                y_dist = prevCentroid[1] - centroid[1]
                dist = math.sqrt(x_dist * x_dist + y_dist * y_dist)

                if dist < 100 and framesToDrop == 0:
                    print "--------------------------------------------------------"
                    print "moving", frameNo
                    moveCamera(move_dirs, givenStep='dynamic', flow=flow)
                    moved = True

        stop_moving = stop_moving - 1 if stop_moving > 0 else 0

        if moved:
            prevCentroid = None
        else:
            prevCentroid = centroid

        if save_experiment:
            metadataFile.write(str(frameName) + "," + str(datetime.datetime.now()) + "," + str(pan) + "," + str(tilt) + "," + str(zoom) + "\n")
            # if flow != None:
            #     optical_flow.optical_flow_in_img(currentImg, flow[0], flow[1])
            cv2.imwrite(folderName + "/" + frameName + ".jpg", currentImg)

        if flow != None:
            optical_flow.optical_flow_in_img(res_img, flow[0], flow[1])
        cv2.imshow('img', res_img)

        key = cv2.waitKey(1)
        if key == 27:
            exit()

    # Show image
    imagetk = ImageTk.PhotoImage(img)
    tbk.addImage(imagetk)
    root.update()