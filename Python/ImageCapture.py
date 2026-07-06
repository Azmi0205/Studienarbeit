import os
import cv2

os.chdir(r"C:\Users\azmib\Desktop\Python")
from BosonSDK import CamAPI

myCam = CamAPI.pyClient(manualport='COM3')  # replace with your COM port
myCam.bosonRunFFC()  # optional: run flat field correction before capture
myCam.Close()

cap = cv2.VideoCapture(1)  # adjust index if Boson isn't device 1
if not cap.isOpened():
    print("Could not open Boson video stream. Try a different index (0, 1, 2...).")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("boson_capture.png", frame)
        print("Image saved as boson_capture.png")
    else:
        print("Failed to read frame from camera.")
    cap.release()