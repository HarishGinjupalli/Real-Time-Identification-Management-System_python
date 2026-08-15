"""
recognize_live.py
------------------
This is the REAL-TIME part of the system.

It opens your webcam, continuously looks for faces, tries to match
each one against the trained model, and:
    - draws a box + name on the live video so you can see it working
    - logs every recognition event into SQL Server (recognition_logs)

Run this after you've registered at least one person and trained
the model:
    python recognition/recognize_live.py

Press 'q' to stop.

WHILE THIS IS RUNNING, open the dashboard in ANOTHER terminal to
watch recognition events appear live:
    streamlit run dashboard/app.py

HOW CONFIDENCE WORKS WITH LBPH:
Unlike most scores where "higher is better", LBPH's confidence
number works the OPPOSITE way - a LOWER number means a CLOSER
match. A confidence around 0-50 is usually a strong match, and
anything above our threshold is treated as "I don't recognize
this person" (UNKNOWN) rather than guessing.
"""

import sys
import os
import time

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database import get_db_connection

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
MODEL_PATH = os.path.join(MODELS_DIR, "lbph_model.yml")

# LBPH confidence: lower = more confident match. Anything above this
# threshold is logged as UNKNOWN instead of guessing a wrong name.
# If real people are showing up as "Unknown" too often, try raising
# this number. If strangers are getting matched to real names, lower it.
CONFIDENCE_THRESHOLD = 70

# Don't log the SAME person again within this many seconds, so a
# person standing in front of the camera doesn't flood the database
# with one row per video frame (~30 rows a second otherwise!).
LOG_COOLDOWN_SECONDS = 5

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def load_person_names():
    """
    Loads a {person_id: full_name} dictionary from the database, so
    we can display real names instead of just numeric IDs.

    Wrapped in a try/except: if the database is unreachable right
    when the script starts, we still let recognition run (everyone
    will just show as "Unknown" until the database comes back and
    the script is restarted) instead of refusing to start at all.

    Input: none
    Returns: a dictionary of {person_id: full_name} (empty if the
             database read failed)
    """
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT person_id, full_name FROM persons WHERE status = 'ACTIVE'")
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        return {row[0]: row[1] for row in rows}
    except Exception as error:
        print(f"Could not load registered people from the database: {error}")
        print("Continuing anyway - recognized faces will show as 'Unknown' "
              "until you restart this script with the database reachable.")
        return {}


def log_recognition_event(person_id, person_name, status, confidence):
    """
    Writes one row into the recognition_logs table.

    Wrapped in a try/except: if the database write fails for any
    reason (SQL Server briefly unavailable, network blip, etc.),
    we print a warning and carry on instead of crashing the whole
    live recognition loop. Losing one logged event is far better
    than losing the webcam feed entirely because of a momentary
    database problem.

    Input: person_id (or None if unknown), person_name, status
           ("RECOGNIZED" or "UNKNOWN"), confidence score
    Returns: True if the event was logged, False if it failed
    """
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(
            """INSERT INTO recognition_logs
               (person_id, person_name, status, confidence_score, camera_source)
               VALUES (?, ?, ?, ?, 'Webcam-0')""",
            (person_id, person_name, status, confidence)
        )
        connection.commit()
        cursor.close()
        connection.close()
        return True
    except Exception as error:
        print(f"Could not log this event to the database (continuing anyway): {error}")
        return False


def main():
    """
    Runs the live webcam recognition loop until 'q' is pressed.
    """
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Run these first:")
        print("    python registration/register_face.py")
        print("    python recognition/train_model.py")
        return

    print("Loading trained model...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    person_names = load_person_names()
    print(f"Loaded {len(person_names)} known person(s): {list(person_names.values())}")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open the webcam. Check that it's connected and not "
              "being used by another application.")
        return

    # Tracks the last time we logged each person, so we can apply the cooldown
    last_logged_at = {}

    print("\nReal-time recognition started. Press 'q' in the video window to stop.\n")

    # Tracks how many consecutive frames failed to read, so a single
    # dropped frame doesn't stop the whole script - but if the
    # camera is genuinely gone (unplugged, closed by another app),
    # we still want to stop instead of looping forever on nothing.
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 30

    while True:
        success, frame = camera.read()
        if not success:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print("\nLost the webcam feed (too many failed frames in a row). Stopping.")
                break
            continue  # skip this frame, try again instead of crashing
        consecutive_failures = 0

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_crop = gray_frame[y:y + h, x:x + w]
            face_crop = cv2.resize(face_crop, (200, 200))

            label, confidence = recognizer.predict(face_crop)

            if confidence < CONFIDENCE_THRESHOLD and label in person_names:
                display_name = person_names[label]
                status = "RECOGNIZED"
                box_color = (0, 255, 0)  # green
                person_id = label
            else:
                display_name = "Unknown"
                status = "UNKNOWN"
                box_color = (0, 0, 255)  # red
                person_id = None

            # Draw the result on the live video
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            label_text = f"{display_name} ({confidence:.0f})"
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            # Only log if we haven't logged this same person/unknown
            # recently, so we don't flood the database every frame
            cooldown_key = person_id if person_id is not None else "unknown"
            now = time.time()
            last_time = last_logged_at.get(cooldown_key, 0)

            if now - last_time >= LOG_COOLDOWN_SECONDS:
                log_recognition_event(person_id, display_name, status, float(confidence))
                last_logged_at[cooldown_key] = now
                print(f"Logged: {display_name} | {status} | confidence={confidence:.1f}")

        cv2.imshow("rIMS - Real-Time Recognition (press 'q' to stop)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("\nRecognition stopped.")


if __name__ == "__main__":
    main()
