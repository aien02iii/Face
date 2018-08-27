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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


#create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# 讀取我們訓練的人臉辨識模型
#read trained data
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
#this function recognizes the person in image passed and draws a rectangle around detected face with name of the subject
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
vs = VideoStream(src=0).start()
# Raspberry Pi + picamera users
#vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)


# 打開鏡頭，進到 video stream frame 模式
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
 
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

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < args["confidence"]:
			continue
		
		label_text = predict(frame)
		
		# 紀錄報到名單，須設定 ubuntu 與 windows 共享資料夾，參考 https://www.youtube.com/watch?v=uQrcNzUWV_I&t=91s	
		# if label_text is not '':
		# 	file = open('/media/sf_faces/patient.csv','a')
		# 	file.write(label_text+'\n')
		# 	file.close()
		
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		#text = "{:.2f}%".format(confidence * 100)
		text = label_text
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		
	# show the output frame
	cv2.imshow("Frame", frame)
	
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()