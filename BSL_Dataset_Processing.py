import mediapipe as mp  
import cv2 as cv  
import os  
import pickle  

# Initializing MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the hands object with static_image_mode=True for processing images
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize lists to store data and labels
data = []
labels = []

# Directory containing the dataset
datadir = '/Users/maliknayyar/Downloads/archive/2_HAND_DATASET/2_HAND_DATASET'

# Loop through each directory in the dataset directory
for dir_ in os.listdir(datadir):
    # Loop through each image file in the current directory
    for img_path in os.listdir(os.path.join(datadir, dir_)):
        aux = []  # Initialize an auxiliary list to store hand landmarks
        img = cv.imread(os.path.join(datadir, dir_, img_path))  
        # Convert image to RGB so that we can send it to MediaPipe
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        # Check if any hand landmarks are detected
        if results.multi_hand_landmarks:  
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    # Extract x and y coordinates of each hand landmark
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    aux.append(x)  
                    aux.append(y)  
            data.append(aux)  
            labels.append(dir_)  

#saving  the data and labels
f = open('data_BSL.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
