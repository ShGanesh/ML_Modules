import cv2
import time
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import draw_landmarks


class poseDetector():
	def __init__(self, mode=False, upperBody= False, smooth=True, detectionCon=0.5, trackingCon=0.5):
		self.mode = mode
		self.upperBody= False
		self.smooth=True
		self.detectionCon=0.5
		self.trackingCon=0.5

		self.mpPose = mp.solutions.pose
		self.pose = self.mpPose.Pose(self.mode, self.upperBody, self.smooth, self.detectionCon, self.trackingCon)
		self.mpDraw = mp.solutions.drawing_utils

	def findPose(self, img, draw=True):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.pose.process(imgRGB)
		if self.results.pose_landmarks:
			if draw:
				self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
		return img
	
	def findPosition(self, img, draw=True):
		lmList = []
		if self.results.pose_landmarks:
			for id, lm in enumerate(self.results.pose_landmarks.landmark):
				h, w, c = img.shape
				cx, cy = int(lm.x*w), int(lm.y*h)		# Converting to pixels
				lmList.append([id, cx, cy])
				if draw:
					cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED) # Circles each landmark
		return lmList




def main():
	#cap = cv2.VideoCapture('path/to/video')
	cap = cv2.VideoCapture(0)
	pTime = 0

	detector = poseDetector()
	while True:
		success, img = cap.read()
		img = detector.findPose(img)
		lmList = detector.findPosition(img)
		print(lmList)
		# cv2.circle(img, (lmList[num][1], lmList[num][2]), 10, (0, 0, 255), cv2.FILLED)
		
		cTime = time.time()
		fps = 1/(cTime-pTime)			# Frames per second
		pTime = cTime

		# Printing fps 
		cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (255, 0, 255), 3)
		cv2.imshow("Image", img)
		cv2.waitKey(1)



if __name__ == "__main_":
	main()