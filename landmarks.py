import argparse
import glob
import cv2
import pandas as pd
import dlib

# Place for the pretrained model for dlib
CNN_FACE_MODEL_PATH = "weight/mmod_human_face_detector.dat"
PREDICTOR_PATH = "weight/shape_predictor_68_face_landmarks.dat"

# Not include the final points
EYEBROW_INDEX = (17, 27)
MOUTH_INDEX = (48, 68)

# Initialize the predictor and face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def str_to_tuple(tuple_str: str):
    r"""Turn str tuple to real tuple
    For example: "(25, 32)" -> (25, 32)

    Parameters
    ----------
    tuple_str: str
        String to be changed

    Returns
    -------
    first_num: int
    second_num: int
    """
    first_num, second_num = tuple_str[1:-1].split(", ")
    return int(first_num), int(second_num)


def row_to_list(row):
    tuple_list = []

    for pair in row:
        tuple_list.append(str_to_tuple(pair))

    return tuple_list


def convert(landmarks):
    result = []
    for points in landmarks.parts():
        result.append((points.x, points.y))

    return result


def detect_landmarks(img):
    # Predict the 68 landmarks and convert to list
    detect_faces = detector(img, 1)

    for detect_face in detect_faces:
        landmarks = predictor(img, detect_face)

    landmarks = convert(landmarks)

    # Get only the eyebrow and mouth landmarks
    eyebrow = landmarks[EYEBROW_INDEX[0]:EYEBROW_INDEX[1]]
    mouth = landmarks[MOUTH_INDEX[0]:MOUTH_INDEX[1]]

    results = [*eyebrow, *mouth]

    return results


def save_landmarks_csv(output_path):
    final_landmarks = []

    img_generator = glob.glob("*.jpg")
    for img_path in img_generator:
        landmarks = detect_landmarks(img_path)
        final_landmarks.append(landmarks)

    landmarks_csv = pd.DataFrame(final_landmarks)
    landmarks_csv.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",
                        type=str,
                        required=True)
    parser.add_argument("--path",
                        type=str,
                        required=True)
    parser.add_argument("--output",
                        type=str,
                        required=True)
    args = parser.parse_args()

    save_landmarks_csv(args.output)
