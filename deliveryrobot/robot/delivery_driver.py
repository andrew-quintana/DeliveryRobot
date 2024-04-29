
import platform
import sys
import os
sys.path.append("../../../jetbot/")
from jetbot import Robot, Camera, bgr8_to_jpeg

sys.path.append("../../deliveryrobot")
from utilities.utilities import *



from uuid import uuid1
import cv2
import os
import time

class DeliveryRobot:
    def __init__(self):
        self.robot = Robot()
        self.camera = Camera.instance(width=224, height=224)

    def take_picture(self, directory):
        filename = os.path.join(directory, str(uuid1()) + '.jpg')
        cv2.imwrite(filename, self.camera.value)
        return filename

    def stop(self):
        self.robot.stop()

    def step_forward(self, speed_m_s=0.4, time_s=0.5):
        self.robot.forward(speed_m_s)
        time.sleep(time_s)
        self.robot.stop()

    def step_backward(self, speed_m_s=0.4, time_s=0.5):
        self.robot.backward(speed_m_s)
        time.sleep(time_s)
        self.robot.stop()

    def step_left(self, speed_m_s=0.4, time_s=0.5):
        self.robot.left(speed_m_s)
        time.sleep(time_s)
        self.robot.stop()

    def step_right(self, speed_m_s=0.4, time_s=0.5):
        self.robot.right(speed_m_s)
        time.sleep(time_s)
        self.robot.stop()

