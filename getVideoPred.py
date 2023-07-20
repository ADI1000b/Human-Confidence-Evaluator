import cv2
import numpy as np
from collections import deque
import keras
import tkinter as tk
from tkinter.filedialog import askopenfilename

width = 1280
height = 720
scale = 2
model = keras.models.load_model("VideoMachine.h5")
print("1.Recorded")
print("2.Live")
user_inp = int(input("Enter Choice: "))

if user_inp == 1:
    vid = askopenfilename()
    cap = cv2.VideoCapture(vid)
elif user_inp == 2:
    cap = cv2.VideoCapture(0)

dict = {
    1: "UnderConfident",
    2: "Less Confident",
    3: "Confident",
    4: "Very Confident",
    5: "Over Confident"
}

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
preds = []
while True:
    _,frame = cap.read()
    frame = cv2.resize(frame,(width,height))
    frame = cv2.flip(frame,1)
    down_scale = np.array(cv2.resize(frame,(150,150)))
    gray = cv2.cvtColor(down_scale, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    down_scale = np.expand_dims(down_scale, axis = 0)
    pred = model.predict(down_scale)
    pred = np.argmax(pred,axis = 1).tolist()[0]
    preds.append(pred)
    for (x,y,w,h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.putText(frame, dict[pred+1], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detects eyes of different sizes in the input image
        eyes = eye_cascade.detectMultiScale(roi_gray)

        #To draw a rectangle in eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

    cv2.imshow("Webcam", frame)

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
labels = [dict[i+1] for i in preds]
with open("preds.txt", "w+") as file1:
    for i in preds:
        file1.write(str(i))
        file1.write("\n")