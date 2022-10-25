import cv2
import numpy as np
import os
import shutil
from pynput.mouse import Listener
from pathlib import Path



cascade = cv2.CascadeClassifier("haar_cascade.xml") #cascade object
vid_capture = cv2.VideoCapture(0) #activate web cam
eye_imgs = Path(os.getcwd() + "/eye_imgs")

def normalize(x): #normalizing function
  return (x - x.min()) / (x.max() - x.min())

def scan(image_size=(32, 32)):
  _, frame = vid_capture.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #-> grayscalse
  bounds = cascade.detectMultiScale(gray, 1.3, 10) #set bounds
  if len(bounds) == 2:
    eyes = []
    for box in bounds:
      x, y, w, h = box #params for rect
      eye = frame[y:y + h, x:x + w] #crop out bounds
      eye = cv2.resize(eye, image_size)
      eye = normalize(eye) #normalize
      eye = eye[10:-10, 5:-5] #crop around eyeball
      eyes.append(eye)
    return (np.hstack(eyes) * 255).astype(np.uint8)
  else:
    return None

def on_click(x, y, button, pressed):
  if pressed:
    eyes = scan()
    if not eyes is None:
      filename = str(eye_imgs) + "/" + "{} {} {}.jpeg".format(x, y, button)
      cv2.imwrite(filename, eyes)
      
with Listener(on_click = on_click) as listener:
  listener.join()