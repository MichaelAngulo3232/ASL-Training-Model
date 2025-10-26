import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3
import platform

# init TTS engine (handle platform voice backend)
if platform.system() == "Darwin":
    engine = pyttsx3.init(driverName='nsss')   # macOS
elif platform.system() == "Windows":
    engine = pyttsx3.init(driverName='sapi5')  # Windows
else:
    engine = pyttsx3.init()                    # Linux / other

engine.setProperty('rate', 170)      # words per minute
engine.setProperty('volume', 1.0)    # max volume

# Load your trained model
clf = joblib.load("asl_model.joblib")

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

def extract_features_from_landmarks(landmarks):
    feats = []
    for lm in landmarks:
        feats.extend([lm.x, lm.y, lm.z])
    return np.array(feats, dtype=float)

cap = cv2.VideoCapture(0)
last_spoken = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        feats = extract_features_from_landmarks(hand_landmarks)

        # model prediction + confidence
        probs = clf.predict_proba([feats])[0]
        classes = clf.classes_
        best_idx = int(np.argmax(probs))
        pred = classes[best_idx]
        prob = float(probs[best_idx])

        # draw the hand skeleton
        mp_drawing.draw_landmarks(
            frame,
            results.multi_hand_landmarks[0],
            mp.solutions.hands.HAND_CONNECTIONS
        )

        # display predicted letter + confidence
        cv2.putText(
            frame,
            f"{pred} ({prob:.2f})",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            3
        )

        # speak ONLY if (1) confident and (2) letter changed
        if prob > 0.4 and pred != last_spoken:
            engine.stop()         # clears any queued text
            engine.say(pred.lower())
            engine.runAndWait()
            last_spoken = pred

    # show video
    cv2.imshow("ASL Live Prediction", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
mp_hands.close()
cv2.destroyAllWindows()