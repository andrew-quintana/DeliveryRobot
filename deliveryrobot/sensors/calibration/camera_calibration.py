"""

Author: Andrew Quintana
Email: aquintana7@gatech.edu
Version: 0.1.0
License: [License Name]

Usage:
[Usage Description]
https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0

Classes:
[Class descriptions]

Functions:
[Provide a list of functions in the module/package with a brief description of each]

Attributes:
[Provide a list of attributes in the module/package with a brief description of each]

Dependencies:
[Provide a list of external dependencies required by the module/package]

License:
[Include the full text of the license you have chosen for your code]

Examples:
[Provide some example code snippets demonstrating how to use the module/package]

"""

import cv2
import numpy as np
import time
import os
import glob

logging = True
debug = True
verbose = False

def calibrate_fisheye_checkerboard(DIR):
    if logging: print("Camera calibration START.")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # local camera intrinsics and calibration variables
    cameraMatrix = np.zeros((3, 3), dtype=np.float64)
    distCoeffs = np.zeros((4, 1), dtype=np.float64)
    rvecs = np.array([],np.float64)
    tvecs = np.array([],np.float64)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    frame_size = None

    # local image variables
    _image_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # board information
    CHECKERBOARD = (6, 9)  # (rows, cols)
    square_size_mm = 30

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, (CHECKERBOARD[0]*CHECKERBOARD[1]), 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    print(objp.dtype, objp.shape)

    # Arrays to store object points and image points
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane


    # Iterate through the calibration images
    for i, fname in enumerate(os.listdir(DIR)):

        # attept to import
        frame = cv2.imread(os.path.join(DIR, fname))

        # error checks
        if frame is None:
            print(f"Warning 100: Invalid Filetype - Image file {i} wasn't able to be imported: {fname}")
            continue

        if _image_shape == None:
            _image_shape = frame.shape[:2]
        else:
            assert _image_shape == frame.shape[:2], "All images must share the same size."

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[:2]

        # finding checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH)

        if ret:
            objpoints.append(objp.copy())
            
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            print(corners2.dtype, corners2.shape)
            imgpoints.append(corners2.astype(np.float32))

            if verbose:
                frame = cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
                cv2.imshow(f'frame {i}', frame)
                cv2.waitKey(0)

        else:
            print(f"Warning 200: Chessboard Processing - Not enough corners could be detected in image file {i}: {fname}")

    cv2.destroyAllWindows()

    # calibrate per opencv
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    
    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        np.array(objpoints, dtype=np.float32),
        np.array(imgpoints, dtype=np.float32),
        image_size,
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    # create camera matrices
    cameraMatrix = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, frame_size, 0)

    # undistort
    cv2.fisheye.initUndistortRectifyMap(
        cameraMatrix[0],
        distCoeffs,
        np.eye(3),
        cameraMatrix[0],
        image_size,
        cv2.CV_16SC2
    )

    # setup write out
    outputSettingsFile = DIR + "/default.xml"
    fs = cv2.FileStorage(outputSettingsFile, cv2.FILE_STORAGE_WRITE)

    # find time
    tm = time.time()
    t2 = time.localtime(tm)
    buf = time.strftime("%c", t2)
    fs.write("calibration_time", buf)

    # print all before writing
    if logging: print(f"WRITING TO {outputSettingsFile}")
    if logging: print("camera_matrix:", cameraMatrix[0])
    if logging: print("distortion_coefficients:", distCoeffs)
    if logging: print("rotation_vectors:", rvecs[0])
    if logging: print("translation_vectors:", tvecs)
    if logging: print("frame_size:", frame_size)

    # intrinsic parameters
    fs.write("camera_matrix", cameraMatrix[0])
    fs.write("distortion_coefficients", distCoeffs)
    fs.write("rotation_vectors", rvecs[0])
    fs.write("translation_vectors", tvecs[0])
    fs.write("frame_size", frame_size)

    if logging: print("Camera calibration COMPLETE.")