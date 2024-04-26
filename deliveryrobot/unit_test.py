"""

Author: Andrew Quintana
Email: aquintana7@gatech.edu
Version: 0.1.0
License: [License Name]

Usage:
[Usage Description]

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

import unittest

from utilities.computational_geometry import *
from utilities.mapper import *
from utilities.utilities import *
from sensors.calibration.camera_calibration import *
from apriltag import Detector
from sensors.camera.apriltagsensor import *

tol = 1e-6

class TestUtilities(unittest.TestCase):
    def test_approximately_equal(self):
        # Test case 1: vectors are approximately equal
        a = np.array([1, 2, 3])
        b = np.array([1.1, 2.1, 3.1])
        tol = 0.2
        result = approximately_equal(a, b, tol)
        self.assertTrue(result)

        # Test case 2: vectors are not approximately equal
        a = np.array([1, 2, 3])
        b = np.array([1.5, 2.5, 3.5])
        tol = 0.2
        result = approximately_equal(a, b, tol)
        self.assertFalse(result)

    def test_deg_rad(self):
        # Test case 1: Convert 0 degrees to radians
        deg = 0
        result = deg_rad(deg)
        expected_result = 0
        self.assertEqual(result, expected_result)

        # Test case 2: Convert 90 degrees to radians
        deg = 90
        result = deg_rad(deg)
        expected_result = np.pi / 2
        self.assertEqual(result, expected_result)

        # Test case 3: Convert 180 degrees to radians
        deg = 180
        result = deg_rad(deg)
        expected_result = np.pi
        self.assertEqual(result, expected_result)

    def test_insert_zeros(self):
        # Test case 1: Insert new rows and columns into a 2x2 matrix
        matrix = np.array([[1, 2], 
                           [3, 4]])
        row_index = 1
        col_index = 1
        num_rows = 1
        num_cols = 1
        result = insert_zeros(matrix, row_index, col_index, num_rows, num_cols)
        expected_result = np.array([[1, 0, 2], 
                                    [0, 0, 0], 
                                    [3, 0, 4]]).astype(float)
        self.assertTrue(np.array_equal(result, expected_result))

        # Test case 2: Insert new rows and columns into a 3x3 matrix
        matrix = np.array([[1, 2, 3], 
                           [4, 5, 6], 
                           [7, 8, 9]])
        row_index = 0
        col_index = 2
        num_rows = 2
        num_cols = 2
        result = insert_zeros(matrix, row_index, col_index, num_rows, num_cols)
        expected_result = np.array([[0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0], 
                                    [1, 2, 0, 0, 3], 
                                    [4, 5, 0, 0, 6], 
                                    [7, 8, 0, 0, 9]])
        self.assertTrue(np.array_equal(result, expected_result))

class TestMap(unittest.TestCase):

    def test_plot_radar(self):
        # Test case 1: Test plot_radar function
        map = Map("window_name", True)
        env = {"ROBOT": np.array([0, 0, 0]),
               "1": np.array([0, 200, np.pi]),
               "8": np.array([200, 100, 3*np.pi/4])}
        map.plot_radar(env)
        result = cv2.imread(f"{docs_dir}/images/map/live_map.png")
        expected_result = cv2.imread("test_samples/test_plot_radar.png")
        self.assertTrue(np.array_equal(result, expected_result))
        
    """def test_generate_obstacle(self):
        tag_state = np.array([0, 0, np.pi/4], dtype=np.float64)  # Rotated tag state
        side_length_m = 10  # Side length greater than 5
        result = generate_obstacle(tag_state, side_length_m)
        expected_result = np.array([[0, 5 * np.sqrt(2)], [-5 * np.sqrt(2), 5 * np.sqrt(2)], [-5 * np.sqrt(2), -5 * np.sqrt(2)], [0, -5 * np.sqrt(2)]])
        self.assertTrue(np.array_equal(result, expected_result))"""


    def test_translation_matrix_2d(self):
        tx = 1
        ty = 2
        result = translation_matrix_2d(tx, ty)
        expected_result = np.array([[1, 0, 1], [0, 1, 2], [0, 0, 1]])
        self.assertTrue(np.array_equal(result, expected_result))
    
    def test_rotation_matrix_2d(self):
        angle = np.pi / 2
        result = rotation_matrix_2d(angle)
        expected_result = np.array([[0, -1], [1, 0]])
        self.assertTrue(np.array_equal(result, expected_result))

class TestUtilities(unittest.TestCase):
    def test_angle(self):
        # Test case 1: datum is 0
        u = np.array([0, 0])
        v = np.array([1, 1])
        result = relative_angle(u, v)
        expected_result = np.pi / 4
        self.assertAlmostEqual(result, expected_result)

        # Test case 2: measured point is 0
        u = np.array([-1, 1])
        v = np.array([0, 0])
        result = relative_angle(u, v)
        expected_result = - np.pi / 4
        self.assertAlmostEqual(result, expected_result)

        # Test case 3: Q1 test
        u = np.array([2, 2])
        v = np.array([3, 3])
        result = relative_angle(u, v)
        expected_result = np.pi / 4
        self.assertAlmostEqual(result, expected_result)

        # Test case 3: Q2 test
        u = np.array([2, 2])
        v = np.array([1, 3])
        result = relative_angle(u, v)
        expected_result = 3 * np.pi / 4
        self.assertAlmostEqual(result, expected_result)

        # Test case 3: Q3 test
        u = np.array([2, 2])
        v = np.array([1, 1])
        result = relative_angle(u, v)
        expected_result = - 3 * np.pi / 4
        self.assertAlmostEqual(result, expected_result)

        # Test case 3: Q4 test
        u = np.array([2, 2])
        v = np.array([3, 1])
        result = relative_angle(u, v)
        expected_result = - np.pi / 4
        self.assertAlmostEqual(result, expected_result)

    def test_collinear(self):
        # Test case 1: Test collinear function
        a = np.array([0, 0])
        b = np.array([1, 1])
        c = np.array([2, 2])
        result = collinear(a, b, c)
        expected_result = True
        self.assertEqual(result, expected_result)

        # Test case 2: Test collinear function
        a = np.array([0, 0])
        b = np.array([1, 1])
        c = np.array([2, 3])
        result = collinear(a, b, c)
        expected_result = False
        self.assertEqual(result, expected_result)

    def test_distance_point_to_segment(self):
        # Test case 1: Test distance_point_to_segment function
        point = np.array([0, 0])
        a = np.array([1, 1])
        b = np.array([2, 2])
        result = distance_point_to_segment(point, a, b)
        expected_result = np.sqrt(2)
        self.assertAlmostEqual(result, expected_result)

        # Test case 2: Test distance_point_to_segment function
        point = np.array([1, 1])
        a = np.array([0, 0])
        b = np.array([2, 0])
        result = distance_point_to_segment(point, a, b)
        expected_result = 1
        self.assertAlmostEqual(result, expected_result)

    def test_intersects(self):
        # Test case 1: Test intersects function
        a = np.array([0, 0])
        b = np.array([1, 1])
        c = np.array([0, 1])
        d = np.array([1, 0])
        result = intersects(a, b, c, d)
        expected_result = True
        self.assertEqual(result, expected_result)

        # Test case 2: Test intersects function
        a = np.array([0, 0])
        b = np.array([1, 1])
        c = np.array([0, 2])
        d = np.array([1, 3])
        result = intersects(a, b, c, d)
        expected_result = False
        self.assertEqual(result, expected_result)

    def test_generate_visibility_points(self):
        # Test case 1: Test generate_visibility_points function
        robot_radius_m = 1
        fos = 1.5
        a = np.array([0, 0])
        b = np.array([1, 0])
        c = np.array([1, 1])
        d = np.array([0, 1])
        obstacle = np.array([a, b, c, d])
        result = generate_visibility_points(robot_radius_m, fos, obstacle)
        expected_result = np.array([[-1.06066017, -1.06066017, 0], 
                                    [ 2.06066017, -1.06066017, 0], 
                                    [ 2.06066017,  2.06066017, 0], 
                                    [-1.06066017,  2.06066017, 0]])
        self.assertTrue(np.all(np.isclose(result, expected_result, tol)))

    def test_left(self):
        # Test case 1: Test left function
        a = np.array([0, 0])
        b = np.array([1, 1])
        c = np.array([2, 2])
        result = left(a, b, c)
        expected_result = False
        self.assertEqual(result, expected_result)

        # Test case 2: Test left function
        a = np.array([0, 0])
        b = np.array([2, 0])
        c = np.array([1, 1])
        result = left(a, b, c)
        expected_result = True
        self.assertEqual(result, expected_result)

    def test_left_on(self):
        # Test case 1: Test left_on function
        a = np.array([0, 0])
        b = np.array([1, 1])
        c = np.array([2, 2])
        result = left_on(a, b, c)
        expected_result = True
        self.assertEqual(result, expected_result)

        # Test case 2: Test left_on function
        a = np.array([0, 0])
        b = np.array([2, 0])
        c = np.array([1, 1])
        result = left_on(a, b, c)
        expected_result = True
        self.assertEqual(result, expected_result)

    def test_test_edge(self):

        a = np.array([0, 0])
        b = np.array([1, 0])
        c = np.array([1, 1])
        d = np.array([0, 1])
        obstacles = [[a,b,c,d]]

        # Test case 1: too close
        line_start = np.array([-10, -10])
        line_end = np.array([-1, -1])
        agent_radius_m = 1
        fos = 1.5
        result = test_edge(line_start, line_end, obstacles, agent_radius_m, fos)
        expected_result = False
        self.assertEqual(result, expected_result)

        # Test case 2: intersects
        line_start = np.array([-10, -10])
        line_end = np.array([10, 10])
        
        agent_radius_m = 1
        fos = 1.5
        result = test_edge(line_start, line_end, obstacles, agent_radius_m, fos)
        expected_result = False
        self.assertEqual(result, expected_result)

        # Test case 3: edge of too close (inclusion test)
        line_start = np.array([2.5, -2.5])
        line_end = np.array([2.5, 2.5])
        agent_radius_m = 1
        fos = 1.5
        result = test_edge(line_start, line_end, obstacles, agent_radius_m, fos)
        expected_result = True
        self.assertEqual(result, expected_result)

    def test_test_node(self):

        a = np.array([0, 0])
        b = np.array([1, 0])
        c = np.array([1, 1])
        d = np.array([0, 1])
        obstacles = [[a,b,c,d]]
        agent_radius_m = 1
        fos = 1.5

        # Test case 1: Test too close to edge
        node = np.array([2.4, 0])
        
        result = test_node(node, obstacles, agent_radius_m, fos)
        expected_result = False
        self.assertEqual(result, expected_result)

        # Test case 2: Test inside
        node = np.array([.5, .5])
        result = test_node(node, obstacles, agent_radius_m, fos)
        expected_result = False
        self.assertEqual(result, expected_result)

        # Test case 3: Test passing
        node = np.array([20, 20])
        result = test_node(node, obstacles, agent_radius_m, fos)
        expected_result = True
        self.assertEqual(result, expected_result)

    def test_visibility_point(self):
        # Test case 1: Test visibility_point function
        robot_radius_m = 1
        fos = 1.5
        a = np.array([0, 1])
        b = np.array([0, 0])
        c = np.array([1, 0])
        result = visibility_point(robot_radius_m, fos, a, b, c)
        expected_result = np.array([-1.06066017, -1.06066017, 0],dtype=np.float64)
        self.assertTrue(np.all(np.isclose(result, expected_result, tol)))

    def test_xprod(self):
        # Test case 1: Test xprod function
        a = np.array([0, 0])
        b = np.array([1, 1])
        c = np.array([2, 2])
        result = xprod(a, b, c)
        expected_result = 0
        self.assertEqual(result, expected_result)

        # Test case 2: Test xprod function
        a = np.array([0, 0])
        b = np.array([1, 1])
        c = np.array([2, 0])
        result = xprod(a, b, c)
        expected_result = -2
        self.assertEqual(result, expected_result)

"""
class TestCalibration(unittest.TestCase):
    def test_calibrate_fisheye_checkerboard(self):
        
        calibrate_fisheye_checkerboard(cal_dir)

        # Assert that the calibration files were created
        self.assertTrue(os.path.exists(os.path.join(cal_dir, "default.xml")))

    def test_get_calibration_info(self):
        
        inputSettingsFile = os.path.join(cal_dir, "default.xml")

        camera_matrix = np.zeros((3, 3), dtype=np.float64)
        distance_coefficients = np.zeros((4, 1), dtype=np.float64)
        rotation_vectors = np.array([],np.float64)
        translation_vectors = np.array([],np.float64)
        frame_size = None

        try:
            with open(inputSettingsFile, "r"):
                # read the settings
                fs = cv2.FileStorage(inputSettingsFile, cv2.FILE_STORAGE_READ)

                # intrinsic parameters
                camera_matrix = fs.getNode("camera_matrix").mat()
                distance_coefficients = fs.getNode("distortion_coefficients").mat()
                rotation_vectors = fs.getNode("rotation_vectors").mat()
                translation_vectors = fs.getNode("translation_vectors").mat()
                frame_size = fs.getNode("frame_size")

                fs.release()
        except:
            self.assertTrue(False)
        
        self.assertTrue(True)
"""
        
class TestAprilTagSensor(unittest.TestCase):
    def setUp(self):
        self.sensor = AprilTagSensor(cal_dir)
        self.image_path = '/Users/aq_home/Library/CloudStorage/OneDrive-Personal/1ODocuments/Projects/jetbot_parking/DeliveryRobot_python/deliveryrobot/test_samples/0x_45y.jpg'

    def test_detect(self):
        # Load the sample image
        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform apriltag detection
        measurements = {}
        result = self.sensor.detect(self.image_path, measurements)

        # Assert that the detection was successful
        self.assertTrue(result)

        # Assert that at least one measurement was recorded
        self.assertTrue(len(measurements) > 0)

        # Assert that the measurements have the expected format
        for tag_id, measurement in measurements.items():
            self.assertIsInstance(tag_id, str)
            self.assertIsInstance(measurement, list)
            self.assertEqual(len(measurement), 3)
            self.assertIsInstance(measurement[0], float)
            self.assertIsInstance(measurement[1], float)
            self.assertIsInstance(measurement[2], float)

if __name__ == '__main__':
    unittest.main()