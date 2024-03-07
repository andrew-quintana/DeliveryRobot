//
// Created by Andrew Quintana on 11/29/23.
//

#include "CameraCalibration.h"

void calibrate_fisheye_checkerboard(const std::string& DIR)
{
    std::printf("Camera calibration START.\n");

    // local camera intrinsics and calibration variables
    cv::Mat cameraMatrix;                                   // camera matrix
    cv::Mat distCoeffs;                                     // distortion coefficients
    cv::Mat rvecs;                                          // rotation vectors
    cv::Mat tvecs;                                          // translation vectors
    cv::Size frame_size;                                    // size of frame for processing

    // local image variables
    std::vector<cv::String> images;                         // vector that will hold filepath strings
    std::vector<std::vector<cv::Point3f> > objpoints;        // world checkerboard points
    std::vector<std::vector<cv::Point2f> > imgpoints;        // image checkerboard points
    cv::Mat frame;                                          // matrix for calibration frame
    cv::Mat gray;                                           // matrix to hold bw version of frame

    // board information
    int CHECKERBOARD[2] = {6,9};    // [0] rows, [1] cols

    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;

    for(int i = 0; i < CHECKERBOARD[1]; i++)
    {
        for(int j = 0; j<CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j,i,0));
    }

    cv::glob(DIR, images);

    // vector to store the pixel coordinates of detected checkerboard corners
    std::vector<cv::Point2f> corner_pts;
    bool success;

    // Looping over all the images in the directory
    for(int i = 0; i<images.size(); i++)
    {
        frame = cv::imread(images[i]);
        std::cout << images[i].c_str() << std::endl;

        // failed import or calibration file check
        if (frame.empty()) {
            std::cout << "\033[33m" << "Warning 100: Invalid Filetype - Image file " << i
                      << " wasn't able to be imported." << std::endl << "\033[0m";
            continue;
        }
        if (i < 2) {
            frame_size = cv::Size(frame.rows,frame.cols);

        }
        cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);

        // Finding checkerboard corners
        // If desired number of corners are found in the image then success = true
        success = cv::findChessboardCorners(gray,cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH);

        /*
         * If desired number of corner are detected,
         * we refine the pixel coordinates and display
         * them on the images of checkerboard
        */
        if(success)
        {
            cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

            // refining pixel coordinates for given 2d points.
            cv::cornerSubPix(gray,corner_pts,cv::Size(1,1), cv::Size(-1,-1),criteria);

            // Displaying the detected corner points on the checkerboard
            cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_pts,success);

            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }
        else {
            std::cout << "\033[33m" << "Warning 200: Chessboard Processing - "
                                       "Not enough corners could be detected in image file " << i
                      << std::endl << "\033[0m";
        }
    }

    cv::fisheye::calibrate(objpoints,
                           imgpoints,
                           cv::Size(gray.rows,gray.cols),
                           cameraMatrix,
                           distCoeffs,
                           rvecs,
                           tvecs);
    cameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs,
                                                 frame_size, 0);

    std::cout
            << "K=" << cameraMatrix << std::endl
            << "D=" << distCoeffs << std::endl;

    cv::Mat map1, map2;
    cv::fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat::eye(3, 3, CV_32F), cameraMatrix, cv::Size(gray.rows,gray.cols), CV_16SC2, map1, map2);


    /*
    // show each remapped image
    for (auto fileName : images) {
        cv::Mat source = cv::imread(fileName);
        // failed import or calibration file check
        if (source.empty()) {
            std::cout << "\033[33m" << "Warning 100: Invalid Filetype - Image file wasn't able to be imported." << std::endl << "\033[0m";
            continue;
        }
        cv::Mat undistort;
        cv::remap(source, undistort, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

        cv::Mat result;
        cv::hconcat(source, undistort, result);
        cv::resize(result, result, cv::Size(), 0.25, 0.25);
        cv::imshow("result", result);
        cv::waitKey();
    }
     */

    std::cout
            << "K=" << cameraMatrix << std::endl
            << "D=" << distCoeffs << std::endl;

    // setup write out
    const std::string outputSettingsFile = DIR + "/default.xml";
    cv::FileStorage fs( outputSettingsFile, cv::FileStorage::WRITE );

    // find time
    std::time_t tm;
    time( &tm );
    struct tm *t2 = localtime( &tm );
    char buf[1024];
    strftime( buf, sizeof(buf), "%c", t2 );
    fs << "calibration_time" << buf;

    // intrinsic parameters

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "rotation_vectors" << rvecs;
    fs << "translation_vectors" << tvecs;
    fs << "frame_size" << frame_size;

    std::printf("Camera calibration COMPLETE.\n");
}