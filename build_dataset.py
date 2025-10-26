import os
import cv2
import mediapipe as mp
import csv

IMAGE_ROOT = "archive/asl_alphabet_train/asl_alphabet_train"
OUT_CSV = "asl_data.csv"

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.3  # lowered from 0.5 to be more forgiving
)

num_total_images = 0
num_detected = 0
num_written = 0

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    # header row
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")
    writer.writerow(header)

    for label in sorted(os.listdir(IMAGE_ROOT)):
        folder = os.path.join(IMAGE_ROOT, label)
        if not os.path.isdir(folder):
            continue

        print(f"\n=== Processing label '{label}' ===")

        for filename in os.listdir(folder):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(folder, filename)
            num_total_images += 1

            img_bgr = cv2.imread(path)
            if img_bgr is None:
                print(f"  [skip: couldn't read] {path}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                print(f"  [no hand] {path}")
                continue

            # we saw a hand!
            num_detected += 1
            hand_landmarks = results.multi_hand_landmarks[0].landmark

            row = []
            for lm in hand_landmarks:
                row += [lm.x, lm.y, lm.z]
            row.append(label)

            writer.writerow(row)
            num_written += 1
            print(f"  [wrote] {path}")

mp_hands.close()

print("\nSummary:")
print("  Total images seen:     ", num_total_images)
print("  Images with a hand:    ", num_detected)
print("  Rows actually written: ", num_written)
print("✅ Done! Saved dataset as", OUT_CSV)
