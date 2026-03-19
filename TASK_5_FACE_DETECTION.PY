import cv2

# 1. Load the pre-trained Haar Cascade model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. To use your webcam, use 0. For an image, use cv2.imread('path_to_image.jpg')
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, img = cap.read()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    # Display the output
    cv2.imshow('Face Detection Demo', img)
    
    # Stop if escape key is pressed
    if cv2.waitKey(30) & 0xff == 27:
        break
        
cap.release()
