"""

Author: Andrew Quintana
Email: aquintana7@gatech.edu
Version: 0.1.0
License: [License Name]

Usage:
Calibrate fisheye camera and write camera calibration xml file.

Functions:
    calibrate_fisheye_checkerboard(): calibrate based on directory with checkerboard images

License:
[Include the full text of the license you have chosen for your code]

Examples:
[Provide some example code snippets demonstrating how to use the module/package]

Sources:
    https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0

"""

import cv2
import numpy as np
import time
import os
import glob
import xml.etree.ElementTree as ET


logging = True
debug = True
verbose = True

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

    # Arrays to store object points and image points
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane

    print("Prepared, about to iterate")
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
            imgpoints.append(corners2.astype(np.float32))
            """if verbose:
                frame = cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
                cv2.imshow(f'frame {i}', frame)
                cv2.waitKey(0)"""

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
    K = cv2.getOptimalNewCameraMatrix(K, D, image_size, 0)
    
    # Create empty UMat objects for the undistortion maps
    map1 = cv2.UMat()
    map2 = cv2.UMat()
    
    # update camera and distortion arrays to matrices
    K_mat = cv2.UMat(K[0].astype(np.float32))
    D_mat = cv2.UMat(D.astype(np.float32))

    # undistort
    cv2.fisheye.initUndistortRectifyMap(
        K_mat,
        D_mat,
        np.eye(3),
        K_mat,
        image_size,
        cv2.CV_32FC1,
        map1,
        map2
    )
    
    K = np.asarray(K_mat.get())
    D = np.asarray(D_mat.get())

    
    # Helper function to convert numpy arrays to strings
    def numpy_to_string(array):
        return ' '.join(map(str, array.flatten()))

    # Create the root element
    root = ET.Element("CalibrationData")

    # Create sub-elements
    camera_matrix_element = ET.SubElement(root, "CameraMatrix")
    camera_matrix_element.text = numpy_to_string(K)

    dist_coeffs_element = ET.SubElement(root, "DistCoeffs")
    dist_coeffs_element.text = numpy_to_string(D)

    rvecs_element = ET.SubElement(root, "Rvecs")
    for rvec in rvecs:
        rvec_element = ET.SubElement(rvecs_element, "Rvec")
        rvec_element.text = numpy_to_string(rvecs[0])

    tvecs_element = ET.SubElement(root, "Tvecs")
    for tvec in tvecs:
        tvec_element = ET.SubElement(tvecs_element, "Tvec")
        tvec_element.text = numpy_to_string(tvecs[0])

    frame_size_element = ET.SubElement(root, "FrameSize")
    frame_size_element.text = ' '.join(map(str, image_size))

    # Write the tree to an XML file
    tree = ET.ElementTree(root)
    tree.write(DIR + "/default.xml")

    if logging: print("Camera calibration COMPLETE.")