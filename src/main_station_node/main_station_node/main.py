import sensor_msgs
import numpy as np
import ros2_numpy as rnp
import rclpy
import cv2
import ctypes
import time
import os
import traceback
import socket
from rclpy.node import Node
from itala import itala
from datetime import datetime
from shared_libraries.client import AsyncClient
from shared_libraries.utils import Action, ErrorCode
from shared_libraries.robot_control import RobotController

    
DIR_PATH = "captured_images"
SAVE_TO_DIR = True 
TIMEOUT = 1000
ACTION = 1 # CAPTURE_ONLY: 0, PROCESSING: 1
CAM_SORTING_SERIAL_NUMBER = '600742'
CAM_METROLO_SERIAL_NUMBER = '600590'
ROBOT_HOST = "192.168.125.1"  # The server's hostname or IP address
ROBOT_PORT = 1025  # The port used by the server

class MainStationNode(Node):
    def __init__(self):
        super().__init__('main_station_node')
        self.get_logger().info("Main node initialized")
        
        ret = self.init_system()
        self.start_measuring_processes()
            
    def init_system(self):
        # init 2 camera nodes use async clients
        self.metrology_cam_cli = AsyncClient("metrology_client_async")
        self.sorting_cam_cli = AsyncClient("sorting_client_async")
        # connect to 2 cameras
        status = self.metrology_cam_cli.send_setup_camera_request(CAM_METROLO_SERIAL_NUMBER)
        if status is False:
            self.get_logger().error("Terminate client due to be unable to setup the Metrology camera!")
            return False 
        status = self.sorting_cam_cli.send_setup_camera_request(CAM_SORTING_SERIAL_NUMBER)
        if status is False:
            self.get_logger().error("Terminate client due to be unable to setup the Sorting camera!")
            return False 

        self.metrology_cam_cli.send_capture_request(DIR_PATH, SAVE_TO_DIR, TIMEOUT, ACTION)
        self.sorting_cam_cli.send_capture_request(DIR_PATH, SAVE_TO_DIR, TIMEOUT, ACTION)
        self.robot = RobotController(ROBOT_HOST, ROBOT_PORT)
        # move robot to home position
        
        self.get_logger().info("Connected to cameras.")
        return True  
    
    def start_measuring_processes(self):
        self.robot.moveHomePos()
        self.get_logger().info("Starting to process screws...")
        while True:
            detected_screws = self.sorting_cam_cli.process_sorting_station(DIR_PATH, SAVE_TO_DIR, TIMEOUT)
            if len(detected_screws) == 0:
                self.get_logger().info("Not found any screws now. Exit the program and return robot to Home positon!")
                self.robot.terminate()
                return 
            
            first_screw = detected_screws[0]
            xPos = int(first_screw.split(".")[0])
            yPos = int(first_screw.split(".")[1])
            self.robot.pickScrew(xPos, yPos)
        
            for i in range(1):
                self.robot.rotateScrew(45)
                # process the screw at Metrology station here
                time.sleep(1)
            
            self.robot.throwScrew(np.random.randint(0,2))
        

def main():
    rclpy.init()
    main_node = MainStationNode()
    rclpy.spin(main_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

        
    

