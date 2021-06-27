import cv2
import mediapipe as mp
import time


class handDetector():
	def __init__(self, mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
		self.mode = mode
		self.maxhands = max_num_hands
		self.detectionCon = min_detection_confidence
		self.trackCon = min_tracking_confidence

		self.mpHands = mp.solutions.hands		# "Formality" for using hands model from mediapipe
		self.hands = self.mpHands.Hands(self.mode, self.maxhands, self.detectionCon, self.trackCon)
		self.mpDraw = mp.solutions.drawing_utils	# for drawing lines and whatnot


	def findHands(self, img, draw=True):

		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.hands.process(imgRGB)

		# Now opening results and extract data
		# Check if hands (one? two? zer0?)
		if self.results.multi_hand_landmarks:
			for handLms in self.results.multi_hand_landmarks:
				if draw:
					self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)	# will draw points on hand landmarks and connect them
		return img


	def findPosition(self, img, handNo=0, draw=True):
		lmList = []
		if self.results.multi_hand_landmarks:
			myHand = self.results.multi_hand_landmarks[handNo]


			for id, lm in enumerate(myHand.landmark):	# id  and landmark
				#print(id, lm)
				h, w, c = img.shape						# height, width, channel				
				cx, cy = int(lm.x*w), int(lm.y*h)		# Converting to pixels
				lmList.append((id, cx, cy))		# why append list inside list? y not append array?
				#cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED) # Circles each landmark

		return lmList



def main():
	pTime = 0 							# previous time = 0
	cTime = 0							# current time = 0
	cap = cv2.VideoCapture(0)
	detector = handDetector()
	while True:
		success, img = cap.read()
		img = detector.findHands(img)
		lmList = detector.findPosition(img)
		if len(lmList)!= 0:
			print(lmList[4]) 	# print landmark of any point

		cTime = time.time()
		fps = 1/(cTime-pTime)			# Frames per second
		pTime = cTime

		# Printing fps 
		cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (255, 0, 255), 3)
		cv2.imshow("Image", img)
		cv2.waitKey(1)


# main() code is the dummy code. I can copy it and use it wherever I want,
# Hence a module has been formed.



if __name__ == "__main__":
	main()
