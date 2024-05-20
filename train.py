import numpy as np
from mtcnn.mtcnn import MTCNN
import cv2 as cv
import os
from keras_facenet import FaceNet
import pickle
from faceLoader import faceLoader

embedder = FaceNet()

def get_embedding(face_img):
    face_img = face_img.astype('float32') # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0) 
    # 4D (Nonex160x160x3)
    yhat= embedder.embeddings(face_img)
    return yhat[0] # 512D image (1x1x512)

# faceLoader = faceLoader("dataset/cache")
# detector = faceLoader.detector
# X, Y = faceLoader.load_classes()
# EMBEDDED_X = []

# for img in X:
#     EMBEDDED_X.append(get_embedding(img))

# EMBEDDED_X = np.asarray(EMBEDDED_X)
# np.savez_compressed('faces_embeddings_done_3classes.npz', EMBEDDED_X, Y)

with np.load('faces_embeddings_done_3classes.npz') as data:
    kza = data['arr_0']
    mza = data['arr_1']


## TESTING
# t_im = cv.imread("testing/ubantu.jpg")
# t_im = cv.cvtColor(t_im, cv.COLOR_BGR2RGB)
# x,y,w,h = detector.detect_faces(t_im)[0]['box']

# t_im = t_im[y:y+h, x:x+w]
# t_im = cv.resize(t_im, (160,160))
# test_im = get_embedding(t_im)


# from sklearn.svm import SVC # Support Vector Machine 
# from sklearn.model_selection import train_test_split

# X_train, X_test, Y_train, Y_test = train_test_split(kza, mza, shuffle=True, random_state=17)
# model = SVC(kernel='linear', probability=True)
# model.fit(X_train, Y_train)

# test_im = [test_im]
# ypreds = model.predict(test_im)

# print(ypreds)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Assuming X and Y are your features and labels respectively
X_train, X_test, Y_train, Y_test = train_test_split(kza, mza, test_size=0.25, random_state=0)
# Create a Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(kza, mza)

# Test the model by making predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(Y_test, y_pred)
# print(f"Accuracy: {accuracy}")

# # Display predictions
# print("Predictions:", y_pred)
# print(classification_report(Y_test, y_pred, target_names=X_test))
# probabilities = model.predict_proba(X_test)


# Create a Logistic Regression model
# model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
# model.fit(kza, mza)

# # Example of a test image wrapped in a list (assuming 'test_im' is a predefined variable holding your test image features)
# test_im = [test_im]
# ypreds = model.predict(test_im)
# print("Predicted class:", ypreds)

# To get the probabilities (confidence scores) for each class
# yprobs = model.predict_proba(test_im)
# print("Probabilities:", yprobs)

# from sklearn.linear_model import SGDClassifier

# # Configure SGDClassifier for logistic regression
# model = SGDClassifier(loss='log_loss')  # 'log' specifies logistic regression

# # Initial training
# model.fit(kza, mza)

# When new data arrives
#model.partial_fit(X_new, y_new)

with open('LR_model_160x160_Cache.pkl','wb') as f:
    pickle.dump(model,f)