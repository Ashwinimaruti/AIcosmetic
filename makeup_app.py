import cv2
import numpy as np

# Load Haarcascade models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def apply_lipstick(img, face):
    x, y, w, h = face
    result = img.copy()

    # Lip area
    lip_y1 = y + int(h * 0.65)
    lip_y2 = y + int(h * 0.80)
    lip_x1 = x + int(w * 0.30)
    lip_x2 = x + int(w * 0.70)

    # Create mask
    mask = np.zeros(img.shape[:2], np.uint8)
    center = ((lip_x1 + lip_x2)//2, (lip_y1 + lip_y2)//2)
    axes = ((lip_x2 - lip_x1)//2, (lip_y2 - lip_y1)//2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # Apply color
    lipstick_color = (60, 20, 180)  # More natural red-pink
    overlay = result.copy()
    overlay[mask > 0] = lipstick_color

    alpha = 0.3 #more subtle
    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
    return result

def apply_blush(img, face):
    x, y, w, h = face
    result = img.copy()

    # Left & Right cheek
    cheek_y = y + int(h * 0.55)
    left_x = x + int(w * 0.25)
    right_x = x + int(w * 0.75)

    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, (left_x, cheek_y), int(w * 0.12), 255, -1)
    cv2.circle(mask, (right_x, cheek_y), int(w * 0.12), 255, -1)

    blush_color = (100, 120, 200) #More subtle coral
    overlay = result.copy()
    overlay[mask > 0] = blush_color

    result = cv2.addWeighted(overlay, 0.2, result, 0.8, 0) #Evenlighter
    return result

def apply_eyeshadow(img, eyes):
    result = img.copy()

    for (ex, ey, ew, eh) in eyes[:2]:
        mask = np.zeros(img.shape[:2], np.uint8)
        center = (ex + ew//2, ey + eh//3)
        axes = (ew//2, eh//3)
        cv2.ellipse(mask, center, axes, 0, 180, 360, 255, -1)

        shadow_color = (150, 100, 60)
        overlay = result.copy()
        overlay[mask > 0] = shadow_color

        result = cv2.addWeighted(overlay, 0.4, result, 0.6, 0)

    return result

# -----------------------------
# MAIN PROGRAM
# -----------------------------
img = cv2.imread("test.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("No face detected!")
    exit()

face = faces[0]  # first face
x, y, w, h = face

# Detect eyes
roi_gray = gray[y:y+h, x:x+w]
eyes = eye_cascade.detectMultiScale(roi_gray)
eyes = [(x+ex, y+ey, ew, eh) for (ex, ey, ew, eh) in eyes]

result = img.copy()
result = apply_lipstick(result, face)
result = apply_blush(result, face)
result = apply_eyeshadow(result, eyes)

cv2.imshow("Original", img)
cv2.imshow("Makeup Applied", result)
cv2.waitKey(0)
cv2.destroyAllWindows()