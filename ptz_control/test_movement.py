__author__ = 'mataevs'

'''
Script that displays frames from a PTZ camera (Samsung SNP-3120V)
'''

import ptz_acq
import ptz_control
import Tkinter as tk
import ImageTk
from StreamViewer import StreamViewer
import random


def keystroke(event):
    print event.keysym, event.keycode
    pan, tilt, zoom = ptz_control.getPTZ()
    if event.keycode == 113:
        ptz_control.setPTZ(pan - 5, tilt, zoom)
    elif event.keycode == 114:
        ptz_control.setPTZ(pan + 5, tilt, zoom)
    elif event.keycode == 111:
        ptz_control.setPTZ(pan, tilt - 5, zoom)
    elif event.keycode == 116:
        ptz_control.setPTZ(pan, tilt + 5, zoom)
    elif event.keycode == 38:
        ptz_control.setPTZ(pan, tilt, zoom + 1)
    elif event.keycode == 52:
        ptz_control.setPTZ(pan, tilt, zoom - 1)


root = tk.Tk()
root.bind('<Key>', keystroke)
root.geometry("%dx%d+0+0" % (640, 480))
root.resizable(False, False)
tbk = StreamViewer(root)

pan = 0
tilt = 40

ptz_control.setPTZ(pan, tilt, 1)

frameNo = 0
for img, imgBase64 in ptz_acq.getFrame(enableViewer=True):
    frameName = "%04d" % (frameNo,)
    frameNo += 1

    if frameNo % 5 == 0:
        ptz_control.setPTZ(pan, tilt, 1)
        signCtrl = random.randint(0,2)
        sign = 1
        if signCtrl == 0:
            sign = -1
        pan = (pan + sign * 10) % 360

    img.save("dump/" + frameName + ".jpg")
    imagetk = ImageTk.PhotoImage(img)
    tbk.addImage(imagetk)
    root.update()