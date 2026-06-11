import cv2
import numpy as np
from keras.models import load_model

# Load trained model
model = load_model('EmoSyncAI emotion_recognition_model_100epochs.h5')

# Open webcam
video = cv2.VideoCapture(0)

# Load face detector
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion labels
labels_dict = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# Store previous emotion to avoid terminal spam
last_emotion = ""

while True:
    ret, frame = video.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

    for (x, y, w, h) in faces:

        # Extract face region
        sub_face_img = gray[y:y+h, x:x+w]

        # Resize to match model input
        resized = cv2.resize(sub_face_img, (48, 48))

        # Normalize
        normalized = resized / 255.0

        # Reshape for model
        reshaped = np.reshape(normalized, (1, 48, 48, 1))

        # Predict emotion
        result = model.predict(reshaped, verbose=0)

        # Get predicted label
        label = np.argmax(result, axis=1)[0]

        # Get confidence score
        confidence = float(np.max(result) * 100)

        emotion = labels_dict[label]

        # Print only when emotion changes
        if emotion != last_emotion:
            print(f"\nDetected Emotion: {emotion}")
            print(f"Confidence: {confidence:.2f}%")
            print("-" * 30)
            last_emotion = emotion

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

        # Draw label background
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)

        # Display emotion + confidence on screen
        display_text = f"{emotion} {confidence:.1f}%"

        cv2.putText(
            frame,
            display_text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # Show webcam feed
    cv2.imshow("EmoSyncAI Emotion Detector", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
cv2.destroyAllWindows()