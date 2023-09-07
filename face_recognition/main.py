# face recognition part II
# IMPORT
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

# INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes1.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160X1601.pkl", 'rb'))

cap = cv.VideoCapture(0)

# Create a dictionary to map class labels to person names
label_to_name = {0: "Aryan", 1: "Chirag sir ", 2: "Dhaval", 3: "Drashti" , 4: "Hiten sir" , 5 : "Jacky" , 6:"Krupali" , 7:"Sonali" , 8:"Sourabh" , 9:"Suraj" , 10: "Vanshika" , 11:"Darshana Ma'am" , 12: "Genevieve Ma'am" , 13:"Sir"}

# WHILE LOOP
while cap.isOpened():
    _, frame = cap.read()  # Capture a frame from the camera
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)  # Detect faces using Haarcascades
    for x, y, w, h in faces:
        img = rgb_img[y:y+h, x:x+w]  # Extract the detected face
        img = cv.resize(img, (160,160))  # Resize the face image to match the FaceNet input size
        img = np.expand_dims(img, axis=0)  # Expand dimensions to match the input shape (batch, width, height, channels)
        ypred = facenet.embeddings(img)  # Get embeddings using the FaceNet model
        prob = model.predict_proba(ypred)  # Get probability distribution over classes
        prob_percentage = np.max(prob) * 100
        if prob_percentage >= 50:  # Check if probability is greater than or equal to 50
            face_label = model.predict(ypred)[0]  # Predict the class label using the trained SVM classifier
            final_name = label_to_name[face_label]  # Retrieve person name using the label-to-name dictionary
            display_name = f"{final_name} ({prob_percentage:.2f}%)"
        else:
            display_name = "Unknown"
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 10)  # Draw a rectangle around the detected face
        cv.putText(frame, display_name, (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,255), 3, cv.LINE_AA)  # Display the predicted name and probability above the face

    cv.imshow("Face Recognition:", frame)  # Display the frame with detected faces
    if cv.waitKey(1) & ord('q') == 27:  # Exit the loop if 'q' is pressed
        break


cap.release()
cv.destroyAllWindows()  # Closing parenthesis was missing here
