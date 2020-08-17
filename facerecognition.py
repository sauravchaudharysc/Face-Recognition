# Recognise Faces using some classification algorithm - KNN
# 1. load the training data (numpy arrays of all the persons)
		# x- values are stored in the numpy arrays
		# y-values we need to assign for each person
# 2. Read a video stream using opencv
# 3. Extract faces out of it
# 4. Use knn to find the prediction of face (int)
# 5. Map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name

import cv2
import numpy as np 
import os 


##Check out my article on KNN for further details
########## KNN CODE ############
def distance(v1, v2):
	# Eucledian 
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]
################################

#Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './data/'

#X values of the data
face_data = [] 
#Y values of the data
labels = []

class_id = 0 # Labels for the given file
#names dictionary
names = {} #Mapping btw id - name

# Now load the training data (numpy arrays of all the persons)
		# x- values are stored in the numpy arrays
		# y-values we need to assign for each person

# Data Preparation
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		#Create a mapping btw class_id and name
		#Last 4 characters .npy is slashed here to save the name of person in name array 
		#this is done on the basis of labels
		names[class_id] = fx[:-4]
		print("Loaded "+fx)
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)


		#Suppose we have a file a.npy and let say it has 10 faces of person A
		#So it has 10 rows. So for all the X-values. We created a array of size 10
		#And Multiplied it with class_id . So for zero it becomes array of zeroes and so on.
		#For each file we are computing one label.

		#Lateron i will concatenate the label and X values

		#Create Labels for the class
		target = class_id*np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

#So we store the X data finally in list
face_dataset = np.concatenate(face_data,axis=0)
#So we store the Y data i.e Labels in list
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

#We concatenate the Face_Data and Label . The Label are reprsented as columns
trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)		

#Testing

while True:
	ret,frame = cap.read()
	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if(len(faces)==0):
		continue

	for face in faces:
		x,y,w,h = face

		#Get the face ROI
		offset = 10
		try:	
			face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
			face_section = cv2.resize(face_section,(100,100))
			if np.all(np.array(face_section.shape)):
				#Call the KNN method to get the prediction
				#We give the training set and pass the linear face section by reshaping it
				#Predicted Label (out)
				out = knn(trainset,face_section.flatten())

				#Display on the screen the name and rectangle around it
				#The KNN will predict the most matching label
				pred_name = names[int(out)]
				#Python: cv2.PutText(img, text, org, font, color) → None
				cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		except:
			pass
		      		
	cv2.imshow("Faces",frame)

	key = cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()	