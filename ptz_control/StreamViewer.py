from Tkinter import *


class StreamViewer(Frame):
    def __init__(self, root):
        root.title("Samsung SNP-3120V viewer")
        self.addFrame(root)

    def addFrame(self, root):
        frame = Frame(root, background="#FFFFFF")
        self.addCanvas(frame)
        frame.pack(fill=BOTH, expand=YES)

    def addCanvas(self, frame):
        self.canvas = Canvas(frame, background='#000000')
        self.canvas.pack(fill=BOTH, expand=YES)
        self.canvas.pack()

    def addImage(self, photoimage):
        self.canvas.create_image(640, 480, image=photoimage, anchor=SE)

    def exit(self, root):
        root.quit()
        sys.exit()
