import cv2
import numpy as np
import dlib
from math import hypot
import time
from numpy import savetxt

from pygame import mixer

mixer.init() 
sound=mixer.Sound("beep.wav")


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio


def getActualsize(ratio, data):
	
	ratio.append(data)

	x = np.size(ratio)

	x = int(x)

	perc = 0

	if x > 1:
		perc = (ratio[-1]*100)/ratio[-2]

	return perc




cont = 1
tempo = []
n_blink = []
ratio = []
tempo_fora = []

check = []
cont2 = 1

while True:
	_, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = detector(gray)
	start = time.time()
	tempo.append(start)
	n_blink.append(cont)
	
	check.append(cont2)
	
	for face in faces:
		
		cont2+=1
		

		landmarks = predictor(gray, face)
		
		left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
		right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
		blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
		check_blink = False;
		actual_size = getActualsize(ratio, blinking_ratio)
		
		


		if actual_size < 90:
			if check_blink == False:
				cont +=1
				
			check_blink = True;

	
	
	
	if np.size(check) > 2:
		print(check[-1] - check[-2])
		diff = check[-1] - check[-2]
		if diff == 0:
			start_in = time.time()
			tempo_fora.append(start_in)
			sound.play()

	

	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()


data = [tempo, n_blink]



savetxt('data.csv', data, delimiter=',')
savetxt('tempo_fora.csv', tempo_fora, delimiter=',')