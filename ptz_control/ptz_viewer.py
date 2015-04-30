'''
Script that displays frames from a PTZ camera (Samsung SNP-3120V)
'''

import ptz_acq
import ptz_control
import Tkinter as tk
import ImageTk
from StreamViewer import StreamViewer


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


frameNo = 0
for img, imgBase64 in ptz_acq.getFrame(enableViewer=True):
	frameName = "%04d" % (frameNo,)
	frameNo += 1
	img.save("dump/" + frameName + ".jpg")
	imagetk = ImageTk.PhotoImage(img)
	tbk.addImage(imagetk)
	root.update()