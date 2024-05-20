from PIL import Image, ImageShow
from sanic import Sanic
import cv2 as cv
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from sanic.response import text, json, file
import time

facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_15classes.npz") # to get classes names..
faces_embeddings_cacheMemory = np.load("faces_embeddings_done_3classes.npz") # to get classes names..
Y = faces_embeddings['arr_1']
Yhat = faces_embeddings_cacheMemory['arr_1']
encoder = LabelEncoder()
encoderHat = LabelEncoder()
encoder.fit(Y)
encoderHat.fit(Yhat)
names = encoder.classes_.tolist()
namesCache = encoderHat.classes_.tolist()

model = pickle.load(open("LR_model_160x160_ALLL_1.pkl", 'rb'))
cacheMemoryModel = pickle.load(open("LR_model_160x160_Cache.pkl", 'rb'))

app = Sanic("FaceRecognitionSystem")

def getToken(): 
    return open("token.env", "r").read()

@app.post("/PostAnImage")
async def getImage(request):
    try:
        # img = {'file': open('image.jpg', 'rb')}
        file = request.files.get('image')
        recievedToken = request.form.get('token')
        previousTime = float(request.form.get('time'))
        tm =  time.time() - previousTime
        if not (recievedToken == getToken()):
            return text("Failed to authorize")
        #result = getJson(img)
        nparr = np.frombuffer(file.body, np.uint8) # convert binary data to numpy array
        img = cv.imdecode(nparr, cv.IMREAD_COLOR) # convert numpy array to image

        face = getFace(img)
        img = np.expand_dims(img,axis=0)
        result = getJson(face,tm)
        tokenJson = {"Token": getToken()}

        return json(result)
    except Exception as e:
        print(e)
        return json("Failed to upload.")


def getJson(image, time):
    prob_array = cacheMemoryModel.predict_proba(facenet.embeddings(image))
    probability = np.round(prob_array.max(), 3)
    if (probability >= 0.75):
        final_name =  namesCache[np.argmax(prob_array)]
        return {"Name": final_name, "Probability": str(probability), "time": time}
    else:
        prob_array = model.predict_proba(facenet.embeddings(image))
        print(prob_array)
        probability = np.round(prob_array.max(), 3)
        if (probability >= 0.75):
            return {"Name": final_name, "Probability": str(probability), "time": time}
    


def getFace(image): # for testing purposes
    haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for x,y,w,h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160,160)) # 1x160x160x3
        cv.imwrite("testFace2.jpg", img)
        img = np.expand_dims(img,axis=0)
        return img


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8443)