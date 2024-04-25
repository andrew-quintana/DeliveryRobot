import ctypes

# Load the shared library
lib = ctypes.CDLL('/home/jetbot/DeliveryRobot/build/libmylib.so')

# Set the return types of the functions
#lib.add.restype = ctypes.c_int
#lib.subtract.restype = ctypes.c_int

# Call the functions
lib.testApriltag()

directory = "/home/jetbot/iterative_setup/docs/default.xml"

# Define the function signature
lib.testOpencv.argtypes = [ctypes.c_char_p]
lib.testOpencv.restype = None

# Call the C++ function with a string
#c_directory = ctypes.c_char_p(directory.encode())
c_directory = ctypes.create_string_buffer(directory.encode())
lib.testOpencv(c_directory)

# Run the Eigen test
lib.testEigen()

# Run Boost test
lib.testBoost()

# Run GTest test
lib.testGTest()
