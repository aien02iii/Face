from time import sleep
#from picamera import PiCamera
import sys
import os
import cv2
from imutils.video import VideoStream

# 自訂資料夾名稱 s6，自拍照片編號幾號到幾號
if len(sys.argv) != 4:
    print("Usage:",sys.argv[0],"<foldername> <start#> <# of images>")
    exit(2)

if not os.path.exists(sys.argv[1]):
    os.makedirs(sys.argv[1])

#camera = PiCamera()
#camera.resolution = (640, 480)
#camera.start_preview()

# Camera warm-up time
#sleep(1)

nstart=int(sys.argv[2])
nstop=nstart+int(sys.argv[3])

# src=0,1,2...取決有幾隻 webcame，從0開始試試看
#cap = cv2.VideoCapture(0)
cap = VideoStream(src=0).start()

# 按enter，開始自動拍照
while True:
    while True:
        k = input()
        break
    for i in range(nstart,nstop):
        #ret, frame = cap.read()
        sleep(2)
        frame = cap.read()
        pname=sys.argv[1]+'/'+str(i)+'.jpg'
        cv2.imwrite(pname, frame)
        cv2.imshow('frame', frame)
        cv2.destroyAllWindows()
        print(pname,'saved...')
    break
    
#cap.release()
cap.stop()
cv2.destroyAllWindows()
