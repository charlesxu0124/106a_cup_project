from calendar import c
import sys
import numpy as np
from scipy import interpolate
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

if 1:
    import cv2
    # sys.path.insert(0, '/opt/ros/melodic/lib/python2.7/dist-packages')
    # import cv2
    import queue
    import threading
    import time

# bufferless VideoCapture
class VideoCapture:
  streams = {}

  def __init__(self, name):
    print("starting video stream", name)
    self.name = name
    if name in self.streams:
        print("already started, returning stream")
        return

    cap = cv2.VideoCapture(name)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 90)
    # cap.set(cv2.CAP_PROP_FOCUS, 20)

    if not cap.isOpened():
        print("Error opening resource: " + str(name))
        print("Maybe opencv VideoCapture can't open it")
        exit(0)
    q = queue.Queue()
    self.streams[name] = (cap, q)
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      cap, q = self.streams[self.name]
      # cap.set(cv2.CAP_PROP_EXPOSURE, -10)
      ret, frame = cap.read()
      # cv2.imshow("frame", frame)
      # key = cv2.waitKey(1)
      if not ret:
        break
      if not q.empty():
        try:
          q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      q.put(frame)

  def read(self):
    cap, q = self.streams[self.name]
    return q.get()


if __name__ == "__main__":
  import sys
  resource = int(sys.argv[1])
  # resource = 4 # "/dev/video1"
  cap = VideoCapture(resource)
  frame = cap.read()
  cv2.imwrite("frame.png", frame)
