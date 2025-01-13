import cv2
import numpy as np
from scipy.spatial.distance import euclidean

def preprocess_image(image_path, target_size=(250, 250)):  
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load the image.")
        return None
    image = cv2.resize(image, target_size)
    
    
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
   
    thresholded = cv2.adaptiveThreshold(blurred, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(thresholded)
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(processed, 50, 150)
    return edges


def register(user_id, image_path):
    features = preprocess_image(image_path)
    if features is not None:
        np.save(f'user_{user_id}_features.npy', features)
        print(f"User {user_id} is Registered Successfully.")
    else:
        print(f"Error processing the image for user {user_id}.")


def check_user(image_path):
    features = preprocess_image(image_path)
    if features is None:
        return
    
    registered_user_features = [
        np.load('user_1_features.npy'),
        np.load('user_2_features.npy'),
    ]
    
    distances = [euclidean(features.flatten(), registered_features.flatten()) for registered_features in registered_user_features]
    threshold = 5000  

    min_distance = min(distances)
    if min_distance < threshold:
        matched_user = distances.index(min_distance) + 1  
        print(f"Match found! Access granted to User {matched_user}")
    else:
        print("Unregistered user detected.")


register(1, 'user1.jpeg')
register(2, 'user2.png')

check_user('unknown_user1.jpeg')
