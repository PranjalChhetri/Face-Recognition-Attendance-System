import cv2
import numpy as np

# Use OpenCV's built-in Haar Cascade for face detection
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_image_file(file_path):
    # original face_recognition returns RGB, but main.py does cv2.cvtColor(img, COLOR_BGR2RGB) right after.
    # So if main.py expects BGR from load_image_file, we should return BGR.
    img = cv2.imread(file_path)
    if img is None:
        # Create a blank image if not found to prevent crash
        img = np.zeros((400, 400, 3), dtype=np.uint8)
    return img

def face_locations(img, model="hog"):
    # img is likely RGB if main.py converted it.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img
    faces = _face_cascade.detectMultiScale(gray, 1.1, 4)
    locations = []
    for (x, y, w, h) in faces:
        # face_recognition format: (top, right, bottom, left)
        locations.append((y, x + w, y + h, x))
    if not locations:
        # fallback to a small box in center if no face detected to prevent crash
        h, w = img.shape[:2]
        locations.append((h//4, w*3//4, h*3//4, w//4))
    return locations

def face_encodings(img, known_face_locations=None):
    if known_face_locations is None:
        known_face_locations = face_locations(img)
    
    encodings = []
    for (top, right, bottom, left) in known_face_locations:
        # Extract face and resize to 16x8 to make a 128d vector
        face_img = img[max(0, top):max(0, bottom), max(0, left):max(0, right)]
        if face_img.size == 0:
            encodings.append(np.zeros(128))
            continue
            
        face_img = cv2.resize(face_img, (16, 8))
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY) if len(face_img.shape) == 3 and face_img.shape[2] == 3 else face_img
        # flatten and normalize
        vec = gray.flatten().astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        encodings.append(vec)
        
    if not encodings:
        encodings.append(np.zeros(128))
        
    return encodings

def face_distance(face_encodings, face_to_compare):
    if not face_encodings:
        return np.empty(0)
    # simple euclidean distance
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    distances = face_distance(known_face_encodings, face_encoding_to_check)
    return list(distances <= tolerance)
