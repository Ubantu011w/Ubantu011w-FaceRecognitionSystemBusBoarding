import cv2
import time
import requests
from datetime import datetime, timedelta
import threading
import RPi.GPIO as GPIO

accuracy = 0.82
expiration_time = 5
timeToFail = 2
shared_container = {} # to return value from threading.timing

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(14,GPIO.OUT) # blue
GPIO.setup(15,GPIO.OUT) # green
GPIO.setup(18,GPIO.OUT) # red
lights = {"blue": 14, "green": 15, "red": 18}
def resetLights():
    GPIO.output(14,GPIO.LOW)
    GPIO.output(15,GPIO.LOW)
    GPIO.output(18,GPIO.LOW)
    
def turnLight(i):
    GPIO.output(i, GPIO.HIGH)
    
resetLights()
def getToken(): 
    return open("token.env", "r").read()

def isNewFace(recent_faces, x, y, w, h, expiration_time, isTimed):
    current_time = datetime.now()
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    face_radius = max(w, h) // 2


    for face in recent_faces:
        last_seen, old_center_x, old_center_y, old_radius = face
        distance = ((old_center_x - face_center_x) ** 2 + (old_center_y - face_center_y) ** 2) ** 0.5
        radius_sum = old_radius + face_radius

        if ((current_time - last_seen) < timedelta(seconds=expiration_time) and distance < radius_sum) and not isTimed:
            face[0] = current_time  # Update last seen time
            return False
    return True

def process_frame(frame, face_cascade, recent_faces, face_id):
    height, width = frame.shape[:2]
    roi_x0, roi_y0, roi_x1, roi_y1 = int(width * 0.3), int(height * 0.15), int(width * 0.7), int(height * 0.85)
    roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(roi_gray, 1.3, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        full_x, full_y = x + roi_x0, y + roi_y0
        if isNewFace(recent_faces, full_x, full_y, w, h, expiration_time, False):
            #face_detected_actions(frame, full_x, full_y, w, h, recent_faces, face_id)
            recent_faces.append([datetime.now(), full_x + w // 2, full_y + h // 2, max(w, h) // 2])
            print("New face detected")
            print("Hold still for 2 seconds...")
            turnLight(lights.get("blue"))
            timer = threading.Timer(2, face_detected_actions, [f'captured_faces/face_{face_id[0]}.png', recent_faces])
            timer.start()
            face_id[0] += 1 
        cv2.rectangle(frame, (full_x, full_y), (full_x + w, full_y + h), (0, 255, 0), 2)
    cv2.rectangle(frame, (roi_x0, roi_y0), (roi_x1, roi_y1), (255, 0, 0), 2)
    cv2.imshow('Face Detection', frame)

def send_image_to_server(image_path):
    url = 'http://192.168.0.102:8443/PostAnImage'  
    # Open the file in binary mode
    files = {'image': open(image_path, 'rb')}
    
    response = requests.post(url, files=files, data = {'token': getToken()})
    files['image'].close()
    return response.json()
    

def face_detected_actions(image_path, recent_faces):
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            roi_x0, roi_y0, roi_x1, roi_y1 = int(width * 0.3), int(height * 0.15), int(width * 0.7), int(height * 0.85)
            roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(roi_gray, 1.3, 5, minSize=(100, 100))
            if (len(faces) < 1):
                resetLights()
                print("Failed after hold still")
                recent_faces.pop()
                return False
            for (x, y, w, h) in faces:
                img = roi[y:y+h, x:x+w]
                img = cv2.resize(img, (160,160)) # 1x160x160x3
                cv2.imwrite(image_path, img)
                print("Image saved as " + image_path)
                result = send_image_to_server(image_path)
                resetLights()
                if (accuracy > float(result.get("Probability"))):
                    turnLight(lights.get("red"))
                    timer = threading.Timer(timeToFail, face_detected_actions, [f'captured_faces/face_{face_id[0]}.png', recent_faces])
                    timer.start()
                    print("failed to identify")
                    print(result)
                    return False
                else:
                    turnLight(lights.get("green"))
                    timer = threading.Timer(expiration_time, resetLights)
                    timer.start()
                    print("success")
                    print(result)
                    return True
    return False

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    recent_faces = []
    face_id = [0]
    start_time = time.time()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        ret, frame = cap.read()
        process_frame(frame, face_cascade, recent_faces, face_id)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

