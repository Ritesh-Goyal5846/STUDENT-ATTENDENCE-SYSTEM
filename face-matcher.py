import cv2
import os
import numpy as np

DATABASE_DIR = "student_database"

def load_student_database():
    known_histograms = []
    known_rolls = []

    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)
        return known_histograms, known_rolls

    for filename in os.listdir(DATABASE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            roll_number = os.path.splitext(filename)[0]
            image_path = os.path.join(DATABASE_DIR, filename)
            
            img = cv2.imread(image_path)
            if img is not None:
                img_resized = cv2.resize(img, (100, 100))
                hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                
                known_histograms.append(hist)
                known_rolls.append(roll_number)

    return known_histograms, known_rolls

def process_class_photo(image_path):
    known_histograms, known_rolls = load_student_database()
    
    class_image = cv2.imread(image_path)
    if class_image is None:
        return {"error": "Could not load class photo image."}

    gray_class = cv2.cvtColor(class_image, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        return {"error": "Could not load haarcascade_frontalface_default.xml from project folder."}

    # Stricter detection parameters
    raw_faces = face_cascade.detectMultiScale(
        gray_class, 
        scaleFactor=1.15, 
        minNeighbors=9, 
        minSize=(70, 70)
    )

    faces = []
    # Filter out weird shapes (like partial ears at the edge)
    for (x, y, w, h) in raw_faces:
        aspect_ratio = w / float(h)
        # Real faces are generally close to square (aspect ratio between 0.7 and 1.3)
        if 0.7 <= aspect_ratio <= 1.3:
            faces.append((x, y, w, h))

    present_students = []
    assigned_rolls = set()
    unknown_count = 0

    for (x, y, w, h) in faces:
        face_roi = class_image[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
            
        face_roi_resized = cv2.resize(face_roi, (100, 100))
        hsv_roi = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2HSV)
        hist_roi = cv2.calcHist([hsv_roi], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist_roi, hist_roi, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        best_match = "Unknown"
        max_score = -1.0

        for i, known_hist in enumerate(known_histograms):
            roll = known_rolls[i]
            if roll in assigned_rolls:
                continue

            score = cv2.compareHist(known_hist, hist_roi, cv2.HISTCMP_CORREL)
            if score > max_score:
                max_score = score
                if score > 0.40:  
                    best_match = roll

        if best_match != "Unknown":
            present_students.append(best_match)
            assigned_rolls.add(best_match)
            color = (0, 255, 0)
        else:
            unknown_count += 1
            color = (0, 0, 255)

        cv2.rectangle(class_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(class_image, f"{best_match} ({max_score:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    output_path = "output_marked_class.jpg"
    cv2.imwrite(output_path, class_image)

    return {
        "present_students": list(set(present_students)),
        "total_faces_detected": len(faces),
        "unknown_faces_count": unknown_count,
        "marked_image_path": output_path
    }

if __name__ == "__main__":
    print("Member 2 OpenCV module loaded successfully.")
    result = process_class_photo("test_class.jpg")
    print("Attendance Result:", result)