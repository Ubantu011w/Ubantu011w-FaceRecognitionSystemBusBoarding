from mtcnn.mtcnn import MTCNN
import timeit
import cv2 as cv
import os
import numpy as np

path = "dataset/4/"

def mtcnnAlgo():
  imgs = os.listdir(path)
  i = 0
  detector = MTCNN()
  for file in os.listdir(path):
    img = cv.imread(path + file)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = detector.detect_faces(img)
    if (len(result) > 0):
      i+=1
  print(str(i) + '/' + str(len(imgs)))
    

def haarcascade():
  imgs = os.listdir(path)
  i = 0
  haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
  for file in os.listdir(path):
    img = cv.imread(path + file)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    if len(faces) > 0:
      i += 1
  print(str(i) + '/' + str(len(imgs)))


execution_time = timeit.timeit('haarcascade()', globals=globals(), number=1)
print(str(execution_time) + 's')
# 240.41304770000002s, 304/304 