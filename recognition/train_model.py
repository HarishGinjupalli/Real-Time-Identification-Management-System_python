"""
train_model.py
---------------
This script trains (or re-trains) the face recognizer using every
photo saved by register_face.py.

WHAT IS LBPH?
LBPH stands for "Local Binary Patterns Histograms". It's a simple,
classic face-recognition algorithm built into OpenCV. In plain
terms: it looks at small patterns of light/dark pixels around each
point in a face image, turns those patterns into a "fingerprint"
for that face, and later compares a new face's fingerprint against
the ones it learned. It's not as powerful as modern deep-learning
face recognition, but it's fast, lightweight, needs no GPU, and
installs with a single pip command - a great fit for a learning
project.

Run this every time you register a new person, or add more photos
for an existing one:
    python recognition/train_model.py

It reads every photo from face_data/, trains the recognizer, and
saves the trained model to models/lbph_model.yml.
"""

import os
import sys

import cv2
import numpy as np

FACE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "face_data"
)
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
MODEL_PATH = os.path.join(MODELS_DIR, "lbph_model.yml")


def load_training_data():
    """
    Walks through face_data/ and loads every photo along with the
    person_id it belongs to (taken from the folder name, which
    register_face.py names like "3_John_Smith").

    Input: none
    Returns: (faces, labels) - a list of grayscale face images (as
             numpy arrays) and a matching list of person_id integers
    """
    faces = []
    labels = []

    if not os.path.exists(FACE_DATA_DIR):
        return faces, labels

    for folder_name in os.listdir(FACE_DATA_DIR):
        folder_path = os.path.join(FACE_DATA_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Folder names look like "3_John_Smith" - the part before the
        # first underscore is the person_id we stored in the database.
        try:
            person_id = int(folder_name.split("_")[0])
        except ValueError:
            print(f"Skipping folder with an unexpected name: {folder_name}")
            continue

        for photo_name in os.listdir(folder_path):
            if not photo_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            photo_path = os.path.join(folder_path, photo_name)
            image = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            faces.append(image)
            labels.append(person_id)

    return faces, labels


def main():
    """
    Loads all training photos, trains the LBPH recognizer, and
    saves the trained model to disk.
    """
    print("Loading training photos from face_data/ ...")
    faces, labels = load_training_data()

    if len(faces) == 0:
        print("No training photos found. Run register_face.py first "
              "to register at least one person.")
        return

    unique_people = len(set(labels))
    print(f"Loaded {len(faces)} photos across {unique_people} person(s).")

    print("Training the LBPH recognizer... (this is quick, usually a few seconds)")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    os.makedirs(MODELS_DIR, exist_ok=True)
    recognizer.write(MODEL_PATH)

    print(f"\nModel trained and saved to: {MODEL_PATH}")
    print("\nNEXT STEP: start real-time recognition with:")
    print("    python recognition/recognize_live.py")


if __name__ == "__main__":
    main()
