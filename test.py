import cv2
import HandTrackingModule as htm
import time


pTime = 0 							# previous time = 0
cTime = 0							# current time = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()
while True:
	success, img = cap.read()
	img = detector.findHands(img)
	lmList = detector.findPosition(img, draw=False)
	if len(lmList)!= 0:
		print(lmList[4]) 	# print landmark of any point

	cTime = time.time()
	fps = 1/(cTime-pTime)			# Frames per second
	pTime = cTime

	# Printing fps 
	cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (255, 0, 255), 3)
	cv2.imshow("Image", img)
	cv2.waitKey(1)
