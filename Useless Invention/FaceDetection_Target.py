import cv2
import time
import mediapipe as mp


class FaceDetector():
	def __init__(self, minDetectionCon = 0.5):

		self.minDetectionCon = minDetectionCon
		self.mpFaceDetection = mp.solutions.face_detection
		self.mpDraw = mp.solutions.drawing_utils
		self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionCon) # MinimumDetectionConfidence, by default = 0.5


	def findFaces(self, img, draw=True):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.faceDetection.process(imgRGB)

		bboxs = []
		# Now opening results and extract data
		if self.results.detections:
			for id, detection in enumerate(self.results.detections):			# aka for each face
				# Create rectangle over detected faces automatically using mediapipe
				#mpDraw.draw_detection(img, detection)

				# Create rectangle over detected faces manually using cv2 module
				bboxC = detection.location_data.relative_bounding_box
				h, w, c = img.shape
				bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
				bboxs.append([bbox, detection.score])
				
				if draw:
					# img = self.fancyDraw(img, bbox)
					img = self.uselessTarget(img, bbox)
				else:
					cv2.rectangle(img, bbox, (255, 0, 255), 2)		 # Creates rectagle over each detected face
				cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

		return img, bboxs
	
	
	def uselessTarget(self, img, bbox, l=30, lt=2, rt=1):
		x, y, w, h = bbox
		w += 15
		h += 15
		# Circles
		cir_coors = (x+int(w), y+int(h))
		cv2.circle(img, cir_coors, 20, (0, 0, 255), lt)		# Inner
		cv2.circle(img, cir_coors, int((w+h)/4), (0, 0, 255), lt+3) # Outer
		
		# Lines
		# Inner Vert
		cv2.line(img, (x+int(w), y+int(h)-12), (x+int(w), y+int(h)+12), (0, 0, 255), rt)
		# Inner Horz
		cv2.line(img, (x+int(w-12), y+int(h)), (x+int(w)+12, y+int(h)), (0, 0, 255), rt)

		return img

	# This one actually paints the target accurately. Hence, not to be used here.
	def fancyDraw(self, img, bbox, l=30, lt=2, rt=1):
		x, y, w, h = bbox
		x1, y1 = x+w, y+h
		# Circles
		cir_coors = (x+int(w/2), y+int(h/2))
		cv2.circle(img, cir_coors, 20, (0, 0, 255), lt)		# Inner
		cv2.circle(img, cir_coors, int((w+h)/4), (0, 0, 255), lt+3) # Outer
		
		# Lines
		# Vertical Line 1
		cv2.line(img, (x+int(w/2), y-20), (x+int(w/2), y+int(h/2)-20), (0, 0, 255), lt)
		# Vertical Line 2
		cv2.line(img, (x+int(w/2), y+int(h/2)+20), (x+int(w/2), y+int(h)+20), (0, 0, 255), lt)
		# Horizontal Line 1
		cv2.line(img, (x-20, y+int(h/2)), (x+int(w/2)-20, y+int(h/2)), (0, 0, 255), lt)
		# Horizontal Line 1
		cv2.line(img, (x+int(w/2)+20, y+int(h/2)), (x+int(w)+20, y+int(h/2)), (0, 0, 255), lt)
		# Inner Vert
		cv2.line(img, (x+int(w/2), y+int(h/2)-12), (x+int(w/2), y+int(h/2)+12), (0, 0, 255), 1)
		# Inner Horz
		cv2.line(img, (x+int(w/2-12), y+int(h/2)), (x+int(w/2)+12, y+int(h/2)), (0, 0, 255), 1)

		# Corners
		# Top Left x, y
		cv2.line(img, (x, y), (x+l, y), (0, 0, 0), lt)
		cv2.line(img, (x, y), (x, y+l), (0, 0, 0), lt)
		# Top Right x1, y
		cv2.line(img, (x1, y), (x1-l, y), (0, 0, 0), lt)
		cv2.line(img, (x1, y), (x1, y+l), (0, 0, 0), lt)
		# Bottom Left x, y1
		cv2.line(img, (x, y1), (x+l, y1), (0, 0, 0), lt)
		cv2.line(img, (x, y1), (x, y1-l), (0, 0, 0), lt)
		# Bottom Right x1, y1
		cv2.line(img, (x1, y1), (x1-l, y1), (0, 0, 0), lt)
		cv2.line(img, (x1, y1), (x1, y1-l), (0, 0, 0), lt)
		return img


def main():
	#cap = cv2.VideoCapture('path/to/video')
	cap = cv2.VideoCapture(0)
	pTime = 0

	detector = FaceDetector()
	while True:
		success, img = cap.read()

		img, _ = detector.findFaces(img)

		cTime = time.time()
		fps = 1/(cTime-pTime)			# Frames per second
		pTime = cTime

		# Printing fps 
		cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (255, 0, 255), 3)
		cv2.imshow("Image", img)
		cv2.waitKey(1)


if __name__ == "__main__":
	main()
