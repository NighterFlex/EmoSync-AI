import cv2
import numpy as np
# from keras.models import load_model
from keras.models import load_model

class EmotionDetector:
    def __init__(self):
        self.model = load_model(
            "EmoSyncAI emotion_recognition_model_100epochs.h5"
        )

        self.face_detector = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml"
        )

        self.labels = {
            0: "Angry",
            1: "Disgusted",
            2: "Fearful",
            3: "Happy",
            4: "Neutral",
            5: "Sad",
            6: "Surprised"
        }

    def detect(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3
        )

        emotion = "Neutral"
        confidence = 0.0

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (48, 48))

            face = face.astype("float32") / 255.0

            face = np.reshape(face, (1, 48, 48, 1))

            result = self.model.predict(
                face,
                verbose=0
            )

            label = np.argmax(result)

            confidence = float(np.max(result))

            emotion = self.labels[label]

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (50, 50, 255),
                2
            )

            cv2.putText(
                frame,
                f"{emotion} {confidence*100:.1f}%",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            break

        return frame, emotion, confidence