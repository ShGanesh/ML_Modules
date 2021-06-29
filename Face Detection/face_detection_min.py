import cv2
import time
import mediapipe as mp

#cap = cv2.VideoCapture('path/to/video')
cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75) # MinimumDetectionConfidence, by default = 0.5

while True:
	success, img = cap.read()
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = faceDetection.process(imgRGB)

	# Now opening results and extract data
	if results.detections:
		for id, detection in enumerate(results.detections):
			# Create rectangle over detected faces automatically using mediapipe
			#mpDraw.draw_detection(img, detection)
			
			# Create rectangle over detected faces manually using cv2 module
			bboxC = detection.location_data.relative_bounding_box
			h, w, c = img.shape
			bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
			cv2.rectangle(img, bbox, (255, 0, 255), 2)		 # Creates rectagle over each detected face
			cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

	
	#cv2.imshow("Image", img)

	cTime = time.time()
	fps = 1/(cTime-pTime)			# Frames per second
	pTime = cTime

	# Printing fps 
	cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (255, 0, 255), 3)
	cv2.imshow("Image", img)
	cv2.waitKey(1)