import mediapipe as mp
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pymongo
from pymongo import MongoClient

cluster = MongoClient("")
db = cluster["Sleep_Data"]
collection = db["UserData"]

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

def detect_mouth_open(face_landmarks, img_h, img_w):
    # Upper lip indices (including the top lip and the sides of the mouth)
    upper_lip_indices = [13, 312, 61, 291]  # Extended to include side landmarks
    # Lower lip indices (including the bottom lip and the sides of the mouth)
    lower_lip_indices = [14, 402, 91, 181]  # Extended to include side landmarks

    # Calculate the average y-coordinates for upper and lower lips
    upper_lip_avg = np.mean([face_landmarks.landmark[idx].y for idx in upper_lip_indices]) * img_h
    lower_lip_avg = np.mean([face_landmarks.landmark[idx].y for idx in lower_lip_indices]) * img_h

    # Determine the gap between the average y-coordinates of the upper and lower lips
    mouth_gap = lower_lip_avg - upper_lip_avg
    # Determine if the gap is greater than the threshold for considering the mouth to be open
    return mouth_gap > 10  # Threshold might need to be adjusted based on testing

    #return False


piCam=Picamera2()
piCam.preview_configuration.main.size=(640, 480)
piCam.preview_configuration.main.format="RGB888"
piCam.preview_configuration.controls.FrameRate= 15
piCam.preview_configuration.align()
piCam.configure("preview")
piCam.start()
fps = 0
pos=(10,460)
font = cv2.FONT_HERSHEY_SIMPLEX
height = 1.5
camColour = (0,0,255)
weight=3
rightCount = 0
# collection.insert_one({"_id":0, "sleep_direction":"Right", "time_spent":0})
leftCount = 0
# collection.insert_one({"_id":1, "sleep_direction":"Left", "time_spent":0})
straightCount = 0
# collection.insert_one({"_id":2, "sleep_direction":"Straight", "time_spent":0})


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
	drawing_spec = mp_drawing.DrawingSpec(color=(128,0,128),thickness=2,circle_radius=1)
	
	while True:
		timeStart=time.time()		#Start timer to be used for FPS
		frame = piCam.capture_array()
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)		#Change colour for the landmarks
		image.flags.writeable = False
		results = holistic.process(image)
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		img_h, img_w, img_c = image.shape  #image height, width and channels 
		face_2d = []
		face_3d = []
		
		
		if results.face_landmarks is not None:		#If there are any results found 
			face_landmarks = results.face_landmarks		# Get the landmarks of the the face
			for idx, lm in enumerate(face_landmarks.landmark):	#go through indexes and LandMarks found in the detection 
				if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:		#Index 33, 263, 1, 61, 291, 199 are indexes on the face for the nose, mouth ears etc (found on https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png)
					if idx == 1:
						nose_2d = (lm.x * img_w, lm.y * img_h)
						nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
					x,y = int(lm.x * img_w), int(lm.y * img_h)
					
					face_2d.append([x,y])
					face_3d.append(([x,y,lm.z]))

									
			
			face_2d = np.array(face_2d,dtype=np.float64)
			face_3d = np.array(face_3d,dtype=np.float64)
			focal_length = 1 * img_w
			
			cam_matrix = np.array([[focal_length,0,img_h/2], [0, focal_length,img_w/2], [0,0,1]])
			distortion_matrix = np.zeros((4,1),dtype=np.float64)
			success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)
			
			rmat,jac = cv2.Rodrigues(rotation_vec)
			
			angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)
			
			x = angles[0] * 360
			y = angles[1] * 360
			z = angles[2] * 360

			mouth_open = detect_mouth_open(results.face_landmarks, img_h, img_w)
			mouth_status = "Open" if mouth_open else "Closed"
			if mouth_status == "Open":
				collection.update_one({"_id": 3}, {"$set": {"mouth_status": "Open"}, "$inc": {"time_spent": 1}})
			else:
				collection.update_one({"_id": 4}, {"$set": {"mouth_status": "Closed"}, "$inc": {"time_spent": 1}})
			

			
			if y < - 10:
				text="Looking Right"
				rightCount+=1
				collection.update_one({"_id": 0}, {"$set": {"sleep_direction": "Right"}, "$inc": {"time_spent": 1}})
				time.sleep(1)
				
				print("Time Spent on Right: " + str(rightCount))
			elif y > 10:
				text="Looking Left"
				collection.update_one({"_id": 1}, {"$set": {"sleep_direction": "Left"}, "$inc": {"time_spent": 1}})
			#elif x < -10:
				#text="Looking Down"
			#elif x > 10:
				#text="Looking up"
			else:
				text="Forward"
				collection.update_one({"_id": 2}, {"$set": {"sleep_direction": "Straight"}, "$inc": {"time_spent": 1}})
			
			nose_3d_projection,jacobian = cv2.projectPoints(nose_3d,rotation_vec,translation_vec,cam_matrix,distortion_matrix)
			p1 = (int(nose_2d[0]),int(nose_2d[1]))
			p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] -x *10))
			
			cv2.line(image,p1,p2,(255,0,0),3)
			
			cv2.putText(frame,text,(20,50), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
			cv2.putText(frame, "x: " + str(np.round(x,2)),(500,50),font,1,(0,0,255),2)
			cv2.putText(frame, "y: " + str(np.round(y,2)),(500,100),font,1,(0,0,255),2)
			cv2.putText(frame, "z: " + str(np.round(z,2)),(500,150),font,1,(0,0,255),2)
			cv2.putText(frame, "Mouth: {}".format(mouth_status), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			
			timeEnd = time.time()
			loopTime = timeEnd-timeStart
			fps =.9*fps + .1*(1/loopTime)
			
			cv2.putText(frame, str(int(fps))+ ' FPS', pos, font, height, camColour, weight)
			
			
			mp_drawing.draw_landmarks(image=image,landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=drawing_spec,connection_drawing_spec=drawing_spec) 
						
			#print(results.face_landmarks) #Uncomment to see Landmarks x,y,z 
			#cv2.putText(frame, str(int(fps))+ ' FPS', pos, font, height, camColour, weight)
				
		# mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, drawing_spec)		
		cv2.imshow("piCam", frame) # Uncomment to see clear camera             
		#cv2.imshow("piCam", image) #Uncomment to see Landmarks on camera
				
		if cv2.waitKey(1)==ord('q'):
			break
		

cv2.destroyAllWindows()

