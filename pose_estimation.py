
import numpy as np
import cv2
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ArUCo-Markers-Pose-Estimation-Generation-Python"))
from utils import ARUCO_DICT
import time

CALIB_FX = 1042.00
CALIB_FY = 1042.00
CALIB_CX = 950.6477088
CALIB_CY = 557.5285168

CALIB_DIST = np.array([
    -0.0516463965,    # k1
    -0.04747710885,   # k2
    -0.0001566917679, # p1
     0.0002697267978, # p2
     0.01013947186,   # k3
     0.3133340326,    # k4
    -0.1464375728,    # k5
     0.02119491113,   # k6
], dtype=np.float64)

CALIB_W = 1920
CALIB_H = 1080

K = np.array([
    [CALIB_FX, 0,        CALIB_CX],
    [0,        CALIB_FY, CALIB_CY],
    [0,        0,        1       ],
], dtype=np.float64)


'''
Sample Usage:-
python pose_estimation.py --type DICT_5X5_100
'''


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected_img_points = detector.detectMarkers(gray)

    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return rvec and tvec
            # marker_length in meters — adjust to your actual printed tag size
            marker_length = 0.05
            obj_points = np.array([
                [-marker_length / 2,  marker_length / 2, 0],
                [ marker_length / 2,  marker_length / 2, 0],
                [ marker_length / 2, -marker_length / 2, 0],
                [-marker_length / 2, -marker_length / 2, 0],
            ], dtype=np.float32)

            _, rvec, tvec = cv2.solvePnP(obj_points, corners[i].reshape(-1, 2),
                                          matrix_coefficients, distortion_coefficients)

            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Draw Axis
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, marker_length * 0.5)

            # Print pose info
            print(f"ID {ids[i][0]}: tvec={tvec.flatten()}  rvec={rvec.flatten()}")

    return frame

if __name__ == '__main__':

    aruco_dict_type = ARUCO_DICT["DICT_5X5_100"]
    if aruco_dict_type is None:
        print("ArUCo tag type 'DICT_5X5_100' is not supported")
        sys.exit(0)

    k = K
    d = CALIB_DIST

    # Load images from folder
    image_dir = "/home/anurag/codes_ole/pose_estimation/Images/stream_captures_churaco_v1"
    IMAGE_EXTENSIONS = {".png", }

    image_paths = sorted(
        p for p in os.listdir(image_dir)
        if os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {image_dir}")

    for img_name in image_paths:
        frame = cv2.imread(os.path.join(image_dir, img_name))
        if frame is None:
            print(f"Could not read {img_name}, skipping")
            continue

        print(f"\n--- {img_name} ---")
        output = pose_estimation(frame, aruco_dict_type, k, d)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(0) & 0xFF  # Wait for key press to advance
        if key == ord('q'):
            break

    cv2.destroyAllWindows()