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

def keystroke(event):
    global pan
    global tilt
    global zoom

    print event.keysym, event.keycode
    if event.keycode == 113:
        pan -= 5
        ptz_control.setPTZ(pan, tilt, zoom)
    elif event.keycode == 114:
        pan += 5
        ptz_control.setPTZ(pan, tilt, zoom)
    elif event.keycode == 111:
        tilt -= 5
        ptz_control.setPTZ(pan, tilt, zoom)
    elif event.keycode == 116:
        tilt += 5
        ptz_control.setPTZ(pan, tilt, zoom)
    elif event.keycode == 38:
        zoom += 1
        ptz_control.setPTZ(pan, tilt, zoom)
    elif event.keycode == 52:
        zoom -= 1
        ptz_control.setPTZ(pan, tilt, zoom)
    elif event.keycode == 9:
        metadataFile.close()
        exit(1)

# Initialize viewer
root = tk.Tk()
root.bind('<Key>', keystroke)
root.geometry("%dx%d+0+0" % (640, 480))
root.resizable(False, False)
tbk = StreamViewer(root)

# Initialize camera pan and tilt
pan = 156
tilt = 48
zoom = 1
ptz_control.setPTZ(pan, tilt, zoom)

# Create dump folder
folderName = "dump_" + datetime.datetime.now().strftime("%d_%m_%H_%M")
if not os.path.exists(folderName):
    os.makedirs(folderName)

# Create metadata text file
metadataFile = open(folderName + ".txt", "w")

frameNo = 0
for img, imgBase64 in ptz_acq.getFrame(enableViewer=True):
    frameName = "%04d" % (frameNo,)
    frameNo += 1

    # Save image
    img.save(folderName + "/" + frameName + ".jpg")

    # Save image metadata
    # Format is CSV: frameNo,timestamp,pan,tilt,zoom
    metadataFile.write(str(frameName) + "," + str(datetime.datetime.now()) + "," + str(pan) + "," + str(tilt) + "," + str(zoom) + "\n")

    # Show image
    imagetk = ImageTk.PhotoImage(img)
    tbk.addImage(imagetk)
    root.update()