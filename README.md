# Face  Recognition

## RUN

- Download OPEN CV
- Download the Project
- Open your terminal and type datacollect.py
- Supply the user name and provide 20-30 frames.
- Press q to exit.
- Open your terminal and type facerecognition.py

### Explanation

- **OpenCV**

![Need](4.jpg)

- **Face Detection With HaarCascades**

   This is used to capture images from the webcam. It detect faces and draws a bounding boxes on each faces. We are going to crop the largest face , if there are multiple faces and are going to save it in numpy array. We will ask the user for the name of the face . We are going to create a 2-d matrix which is image itself then we are going to flatten it and save it as .npy file.

  Multiple experiment are done with different person to get the training data.

  ```
  #Read a video stream from Camera(Frame By Frame)
  import cv2
  
  #Open the default webcam
  cap = cv2.VideoCapture(0)
  
  #to read the file or classifier which work on facial data
  face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
  ```

  Now a Loop which exits when user presses q.

  ```
  while True:
  	#Code for face detection and save it.
  	
  	#Run Loops Till We press q
  	key_pressed = cv2.waitKey(1) & 0xFF
  	if key_pressed == ord('q'):
  		break		
  ```

  To Save the face list and close all the windows and release the camera.

  ```
  
  # Convert our face list array into a numpy array
  face_data = np.asarray(face_data)
  face_data = face_data.reshape((face_data.shape[0],-1))
  print(face_data.shape)
  
  # Save this data into file system
  np.save(dataset_path+file_name+'.npy',face_data)
  print("Data Successfully save at "+dataset_path+file_name+'.npy')
  	
  cap.release()
  cv2.destroyAllWindows()
  
  ```

  For further information see this [HaarCascades](https://github.com/sauravchaudharysc/Face-Detection)

- **KNN Algorithm For Face Prediction**

This will help you to match the faces with the faces saved on our data. It will basically gives the prediction . We are going to use KNN for the prediction of id. We will match the predicted id to the name of user. We are going to create a dictionary in which class id is mapped with file name. So that i can show on the screen the bounding box and name.

```
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
```

For More Details on KNN. You can check my  [article on it](https://sauravchaudharysc.github.io/KNN-Algorithm/)

#### A BUG

Traceback (most recent call last):
File “face_recognition.py”, line 93, in
face_section = cv2.resize(face_section,(100,100))
cv2.error: OpenCV(4.1.0) C:\projects\opencv-python\opencv\modules\imgproc\src\resize.cpp:3718:
error: (-215:Assertion failed) !ssize.empty() in function ‘cv::resize’

getting shape (67,0,3) for face_section, that was causing error.

`if np.all(np.array(face_section.shape)):` this statement solved the problem but if occurs try taking the face near the camera and let it predict.
