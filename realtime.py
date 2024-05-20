# face recognition part II
#IMPORT
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

#INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("LR_model_160x160_tests.pkl", 'rb'))

Names = ["Abdulla", "Ahmad", "Ahmad_gh", "Jenna", "Robert", "Taylor"]
#cap = cv.VideoCapture("tcp://192.168.87.163:8494")
cap = cv.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    #frame = cv.rotate(frame, cv.ROTATE_180) # rotate
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for x,y,w,h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160,160)) # 1x160x160x3
        img = np.expand_dims(img,axis=0)
        ypred = facenet.embeddings(img)
        prob_array = model.predict_proba(ypred)
        #final_name = encoder.inverse_transform(face_name)[0]
        final_name =  Names[np.argmax(prob_array)]
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 5)
        cv.putText(frame, str(final_name) + ": " + str(np.round(prob_array.max(), 3)), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,255), 3, cv.LINE_AA)

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & ord('q') ==27:
        break

cap.release()
cv.destroyAllWindows