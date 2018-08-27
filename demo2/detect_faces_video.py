# USAGE
# python3 detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from utilities import detect_face
from tracker import tracking
import sys


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())



(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


### 多種 tracker 都可以試試看
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRF', 'MOSSE']
tracker_type = tracker_types[2]

if int(minor_ver) < 3:
	tracker = cv2.Tracker_create(tracker_type)
else:
	if tracker_type == 'BOOSTING':
		tracker = cv2.TrackerBoosting_create()
	if tracker_type == 'MIL':
		tracker = cv2.TrackerMIL_create()
	if tracker_type == 'KCF':
		tracker = cv2.TrackerKCF_create()
	if tracker_type == 'TLD':
		tracker = cv2.TrackerTLD_create()
	if tracker_type == 'MEDIANFLOW':
		tracker = cv2.TrackerMedianFlow_create()
	if tracker_type == 'GOTURN':
		tracker = cv2.TrackerGOTURN_create()
	if tracker_type == 'CSRF':
		tracker = cv2.TrackerCSRT_create()
	if tracker_type == 'MOSSE':
		tracker = cv2.TrackerMOSSE_create()

		

#create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# 讀取我們訓練的人臉辨識模型
face_recognizer.read('training-data/trainner.yml')

# 讀取我們對應的的人臉標籤
#there is no label 0 in our training data so subject name for index/label 0 is empty
#read label subjects = ["", "Ramiz Raja", "Elvis Presley", "Alan", "MARS"]
subjects=['']
with open('labels.txt') as fp:
    for data in fp:
        subjects+=[data.strip('\n')]
#print(subjects)

# 定義人臉預測函式
def predict(test_img):
	#make a copy of the image as we don't want to chang original image
	img = test_img.copy()
	#detect face from the image
	face, rect = detect_face(img)

	if face is None or rect is None:
		return ''

	#predict the image using our face recognizer
	label, confidence = face_recognizer.predict(face)
	print(confidence)
	
	#get name of respective label returned by face recognizer
	label_text = subjects[label]
	print(label_text)

	return label_text

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
# in general this would be your laptop’s built in camera or your desktop’s first camera detected
#vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(0)
# Raspberry Pi + picamera users
#vs = VideoStream(usePiCamera=True).start()

time.sleep(1.0)

# 打開鏡頭，進到 video stream frame 模式
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 640 pixels
	okornot, frame = vs.read()
	frame = imutils.resize(frame, width=640)
	
	# Draw door(exit)
	cv2.rectangle(frame, (640,200), (540,400), (0,255,0), 2, 1)
	cv2.putText(frame, "door", (540,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
		if confidence < args["confidence"]:
			continue
		
		label_text = predict(frame)
		#print(label_text)
		
		# compute the (x, y)-coordinates of the bounding box for the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		#print((startX, startY, endX, endY))
		
		
		# 進入 tracking，此處鎖定特定人物，例如'Alan'
		if label_text == 'Alan':
		
			cv2.destroyWindow('Frame')
			bbox = (startX, startY, (endX-startX), (endY-startY))
			
			while True:
				# Read a new frame
				ok, frame = vs.read()
			
				# 載入座標
				# Initialize tracker with first frame and bounding box        
				ok = tracker.init(frame, bbox)
			
				# Start timer
				timer = cv2.getTickCount()
			
				# Update tracker
				ok, bbox = tracker.update(frame)
			
				# Calculate Frames per second (FPS)
				fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
			
				# Draw bounding box
				if ok:
					# Tracking success
					p1 = (int(bbox[0]), int(bbox[1]))
					p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
					cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)
					cv2.putText(frame, label_text, p1,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
					# Draw door(exit)
					cv2.rectangle(frame, (640,200), (540,400), (0,255,0), 2, 1)
					cv2.putText(frame, "door", (540,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
				else :
					# Tracking failure
					cv2.putText(frame, label_text+" leaves the house!!", (50,150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
					# Draw door(exit)
					cv2.rectangle(frame, (640,200), (540,400), (0,255,0), 2, 1)
					cv2.putText(frame, "door", (540,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

				# Display tracker type on frame
				cv2.putText(frame, tracker_type + " Tracker", (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

				# Display FPS on frame
				cv2.putText(frame, "FPS : " + str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

				# Display result
				cv2.imshow("Tracking", frame)

				# Exit if ESC pressed
				k = cv2.waitKey(1) & 0xff
				if k == ord("q") : 
					cv2.destroyWindow('Tracking')
					break


		# draw the bounding box of the face along with the associated
		# probability
		#text = "{:.2f}%".format(confidence * 100)
		text = label_text
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(255, 0, 0), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
		
	# show the output frame
	cv2.imshow("Frame", frame)
	
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
#vs.stop()
vs.release()