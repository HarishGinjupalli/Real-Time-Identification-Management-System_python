"""
register_face.py
-----------------
This script registers a NEW PERSON into the system.

What it does, step by step:
    1. Asks you to type the person's name
    2. Creates a record for them in the SQL Server "persons" table
    3. Opens your webcam and captures a batch of face photos of them
    4. Saves those photos into face_data/<person_id>_<name>/
    5. Reminds you to run train_model.py next

Run this once per person you want the system to recognize:
    python registration/register_face.py

Press 'q' at any time during capture to stop early.

WHY WE NEED MULTIPLE PHOTOS PER PERSON:
The recognizer (LBPH) learns a person's face better when it sees
it from slightly different angles, expressions, and lighting.
We capture ~30 photos per person for a reasonably reliable model
- more than that gives diminishing returns for a project this size.
"""

import sys
import os

import cv2

# Add the project root folder to the path so we can import our
# own database helper from the "database" folder.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database import get_db_connection, insert_and_get_id

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
PHOTOS_PER_PERSON = 30
FACE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "face_data"
)

# OpenCV ships with a ready-made face detector (Haar Cascade) -
# we don't need to train our own "is this a face" detector, only
# our own "whose face is this" recognizer later.
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def sanitize_name_for_folder(name):
    """
    Turns a person's name into a safe folder name
    (no spaces or special characters).
    Input: name (string)
    Returns: a filesystem-safe version of the name
    """
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def register_person_in_database(full_name):
    """
    Inserts a new row into the persons table.
    Input: full_name (string)
    Returns: the new person_id
    """
    connection = get_db_connection()
    cursor = connection.cursor()

    person_id = insert_and_get_id(
        cursor,
        "INSERT INTO persons (full_name, photo_count, status) VALUES (?, 0, 'ACTIVE')",
        (full_name,)
    )

    connection.commit()
    cursor.close()
    connection.close()
    return person_id


def update_photo_count(person_id, photo_count):
    """
    Updates how many training photos were captured for this person.
    Input: person_id, photo_count
    Returns: nothing
    """
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute(
        "UPDATE persons SET photo_count = ? WHERE person_id = ?",
        (photo_count, person_id)
    )
    connection.commit()
    cursor.close()
    connection.close()


def capture_face_photos(person_id, full_name):
    """
    Opens the webcam and saves PHOTOS_PER_PERSON cropped, grayscale
    face images for this person.

    Input: person_id, full_name
    Returns: how many photos were actually saved
    """
    folder_name = f"{person_id}_{sanitize_name_for_folder(full_name)}"
    save_dir = os.path.join(FACE_DATA_DIR, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open the webcam. Check that it's connected and not "
              "being used by another application.")
        return 0

    print(f"\nLook at the camera, {full_name}. Capturing {PHOTOS_PER_PERSON} photos...")
    print("Move your head slightly between shots for better variety.")
    print("Press 'q' to stop early.\n")

    photo_count = 0

    while photo_count < PHOTOS_PER_PERSON:
        success, frame = camera.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Save just the cropped face, resized to a consistent size,
            # so every training image has the same dimensions.
            face_crop = gray_frame[y:y + h, x:x + w]
            face_crop = cv2.resize(face_crop, (200, 200))

            photo_path = os.path.join(save_dir, f"img_{photo_count}.jpg")
            cv2.imwrite(photo_path, face_crop)
            photo_count += 1

            # Draw a box on the live preview so you can see what's being captured
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {photo_count}/{PHOTOS_PER_PERSON}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            break  # only save one face per frame, even if several are detected

        cv2.imshow("Registering face - press 'q' to stop early", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if photo_count >= PHOTOS_PER_PERSON:
            break

    camera.release()
    cv2.destroyAllWindows()
    return photo_count


def main():
    """
    Runs the full registration flow.
    """
    print("=== Register a New Person ===")
    full_name = input("Enter the person's full name: ").strip()

    if not full_name:
        print("Name cannot be empty. Please run the script again.")
        return

    person_id = register_person_in_database(full_name)
    print(f"Created database record: person_id={person_id}, name={full_name}")

    photo_count = capture_face_photos(person_id, full_name)

    if photo_count == 0:
        print("No photos were captured. Registration incomplete - "
              "you can re-run this script to try again.")
        return

    update_photo_count(person_id, photo_count)
    print(f"\nSaved {photo_count} face photos for {full_name}.")
    print("\nNEXT STEP: run this to (re)train the recognizer with the new data:")
    print("    python recognition/train_model.py")


if __name__ == "__main__":
    main()
