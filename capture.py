import cv2 as cv
import PIL

#INITIALIZE
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv.VideoCapture("tcp://127.0.0.1:8494")
name = ""
# WHILE LOOP

i = 7
while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for x,y,w,h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160,160)) # 1x160x160x3
        newframe = frame
        #cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
    if (cv.waitKey(25) & 0xFF) == ord('q'):
        cv.imwrite("dataset/" + name + '/' + str(i) + ".jpg", newframe)
        print("reaction")
        i += 1
    cv.imshow("Face Recognition:", frame)

cap.release()
cv.destroyAllWindows