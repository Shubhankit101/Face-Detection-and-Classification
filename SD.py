import cv2
import numpy as np
#init Camera
cap=cv2.VideoCapture(0)
 
face_cascade=cv2.CascadeClassifier("C:\\Users\\91843\\Notebooks\\Sublime\\haarcascade_frontalface_alt.xml")  # we can add the location where haarcascade is saved.
# in my case it is saved at the above location

skip=0
face_data=[]
dataset_path='./data/'

file_name=input("enter the name of the person ")
while True:
	ret,frame=cap.read()

	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #to save memory

	if ret==False:
		continue

	#cv2.imshow("Video",frame)
	
	faces=face_cascade.detectMultiScale(frame,1.3,5)
	faces=sorted(faces, key=lambda f:f[2]*f[3]) # we are going to use reverse sorting so that the largest face comes at the front



	#print(faces)
	# store the 10th face

	face_section = np.zeros((100,100,3))
	for face in faces[-1:]:

		x,y,w,h=face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)


		#extract crop out the required face
		offset = 10 
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]										#slicing is preformed
		face_section=cv2.resize(face_section,(100,100))
		skip+=1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))
			   

	cv2.imshow("Video",frame)  
	cv2.imshow("face section",face_section)



	#button Part to exit the code
	key_pressed=cv2.waitKey(1)&0xFF
	if key_pressed==ord('q'):
		break
#convert our face list array into numpy array
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# Save the data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data succesfully save at "+dataset_path+file_name+'.npy')
cap.release()
cv2.destroyAllWindows()






