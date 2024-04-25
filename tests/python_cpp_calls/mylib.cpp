#include "mylib.h"
#include <iostream>
#include <fstream>

void testApriltag() {
    // Initialize the AprilTag detector
    apriltag_family_t* tf = tag36h11_create();
    apriltag_detector_t* td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);

    // Create a test image
    image_u8_t* im = image_u8_create(640, 480);

    // Detect AprilTags in the test image
    zarray_t* detections = apriltag_detector_detect(td, im);

    // Print the number of detected AprilTags
    std::cout << "Detected " << zarray_size(detections) << " AprilTags" << std::endl;

    // Clean up
    apriltag_detector_destroy(td);
    tag36h11_destroy(tf);
    image_u8_destroy(im);
    apriltag_detections_destroy(detections);

}

void testOpencv( const char* DIR ) {

	std::cout << "Printing DIR: ";
	std::cout << DIR << std::endl;
	std::ifstream file(DIR);
	
	if (file.is_open())
    {
        cv::FileStorage fs(DIR, cv::FileStorage::READ); // Read the settings

        // intrinsic parameters
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> distance_coefficients;
        fs["rotation_vectors"] >> rotation_vectors;
        fs["translation_vectors"] >> translation_vectors;
        fs["frame_size"] >> frame_size;
        fs.release();
    }
    
    std::cout << "Camera_Matrix: " << camera_matrix << std::endl;
    std::cout << "Distance_Coefficients: " << distance_coefficients << std::endl;
    std::cout << "rotation_vectors: " << rotation_vectors << std::endl;
    std::cout << "translation_vectors: " << translation_vectors << std::endl;
    std::cout << "frame_size: " << frame_size << std::endl;

}


bool testEigen() {
    // Create a 3x3 matrix
    Eigen::Matrix3d m;
    m << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

    // Print the matrix
    std::cout << "The matrix m is:\n" << m << std::endl;

    // Compute the determinant of the matrix
    double det = m.determinant();
    std::cout << "The determinant of m is: " << det << std::endl;
    
    return true;
}

bool testBoost() {
    // Create a simple undirected graph
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
    Graph g;

    // Add some vertices and edges
    boost::graph_traits<Graph>::vertex_descriptor v1 = boost::add_vertex(g);
    boost::graph_traits<Graph>::vertex_descriptor v2 = boost::add_vertex(g);
    boost::graph_traits<Graph>::vertex_descriptor v3 = boost::add_vertex(g);
    boost::add_edge(v1, v2, g);
    boost::add_edge(v2, v3, g);

    // Print the number of vertices and edges
    std::cout << "Number of vertices: " << boost::num_vertices(g) << std::endl;
    std::cout << "Number of edges: " << boost::num_edges(g) << std::endl;

    return 0;
}

TEST(MyTestCase, MyTest) {
    EXPECT_EQ(1, 1);
}

void testGTest() {
    ::testing::InitGoogleTest();
    RUN_ALL_TESTS();
}
