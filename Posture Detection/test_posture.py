import cv2
import time
import PostureModule as PM

cap = cv2.VideoCapture(0)
pTime = 0

detector = PM.poseDetector()
while True:
	success, img = cap.read()
	img = detector.findPose(img)
	lmList = detector.findPosition(img)
	print(lmList)
	# It is printing all the landmarks. 
	# We can use lmList[num] for specific data.
	# cv2.circle(img, (lmList[num][1], lmList[num][2]), 10, (0, 0, 255), cv2.FILLED)
	
	cTime = time.time()
	fps = 1/(cTime-pTime)			# Frames per second
	pTime = cTime

	# Printing fps 
	cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (255, 0, 255), 3)
	cv2.imshow("Image", img)
	cv2.waitKey(1)