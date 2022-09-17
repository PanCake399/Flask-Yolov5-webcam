from flask import Flask, render_template, url_for, request, redirect, Response
import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import torch

app = Flask(__name__)
 
#model = torch.hub.load('.', 'custom', path='yolov5n.pt', force_reload=True, source='local') 
model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5n.pt')
# Set Model Settings
model.eval()
model.conf = 0.6  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1) 

def gen():
    cap=cv2.VideoCapture(0)
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-fram ## read the camera frame
        success, frame = cap.read()
        if success == True:

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)
            results.print()  
          
            #convert remove single-dimensional entries from the shape of an array
            img = np.squeeze(results.render()) #RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR

        else:
            break

        # Encode BGR image to bytes so that cv2 will convert to RGB
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        #print(frame)
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/")
@app.route("/home")
@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/login")
def login():
    return render_template('login.html')


@app.route("/register")
def register():
    return render_template('register.html')

@app.route("/rooms")
def rooms():
    return render_template('rooms.html')

@app.route("/yolo")
def yolo():
    return render_template('yolo.html')

@app.route("/video")
def video():
     return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=False)