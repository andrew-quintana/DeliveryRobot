from lib.jetbot.jetbot import Robot
import time
import ctypes
from drivers import delivery_driver

robot = Robot()

robot.left(0.3)
time.sleep(0.5)
robot.stop()
