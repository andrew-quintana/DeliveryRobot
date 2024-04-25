from enum import Enum
import ctypes

# Load the shared library
lib = ctypes.CDLL('/home/jetbot/DeliveryRobot/build/libdelivery_interface.so')

# Set return type of the step_fsm() function
lib.step_fsm.argtypes = [ctypes.c_char_p]
lib.step_fsm.restype = ctypes.c_char_p

# Create interpreter function
def delivery_interpreter( filename ):

    # Encode the filename to pass to the step function
    c_filename = ctypes.create_string_buffer(filename.encode())

    # Call the step function, passing the most recent image's filename
    output = lib.step_fsm(c_filename)
    
    # Determine which 
    output = output.encode()
    args = output.split()

    return args[0], float(args[1]), float(args[2])