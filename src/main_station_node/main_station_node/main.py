import sensor_msgs
import numpy as np
import ros2_numpy as rnp
import rclpy
import cv2
import ctypes
import time
import os
import traceback
from rclpy.node import Node
from itala import itala
from datetime import datetime
from shared_libraries.client import AsyncClient
from shared_libraries.utils import Action, ErrorCode

    
DIR_PATH = "captured_images"
SAVE_TO_DIR = True 
TIMEOUT = 1000
ACTION = 0 # CAPTURE_ONLY: 0, PROCESSING: 1
CAM_SORTING_SERIAL_NUMBER = '600742'
CAM_METROLO_SERIAL_NUMBER = '600590'

class MainStationNode(Node):
    def __init__(self):
        super().__init__('main_station_node')
        self.get_logger().info("Main node initialized")
        
        ret = self.init_system()
        ret = self.control_workflow()
        if not ret:
            self.get_logger().error("The workflow ran into a problem!")
        else:
            self.get_logger().info("The workflow is not complete!")
            
    def init_system(self):
        # init 2 camera nodes use async clients
        self.metrology_cam_cli = AsyncClient("metrology_client_async")
        self.sorting_cam_cli = AsyncClient("sorting_client_async")
        # connect to 2 cameras
        status = self.metrology_cam_cli.send_setup_camera_request(CAM_METROLO_SERIAL_NUMBER)
        if status is False:
            self.get_logger().error("Terminate client due to be unable to setup the camera!")
            return False 
        status = self.sorting_cam_cli.send_setup_camera_request(CAM_SORTING_SERIAL_NUMBER)
        if status is False:
            self.get_logger().error("Terminate client due to be unable to setup the camera!")
            return False 

        self.metrology_cam_cli.send_capture_request(DIR_PATH, SAVE_TO_DIR, TIMEOUT, ACTION)
        self.sorting_cam_cli.send_capture_request(DIR_PATH, SAVE_TO_DIR, TIMEOUT, ACTION)
        # move robot to home position
        
        self.get_logger().info("Connected to cameras.")
        return True  
    
    def control_workflow(self):
        ret = True
        while True:
            # get num valid screws and their positions
            
            # if num valid = 0 --> break
            # else:
            
            # pick the first valid screw --> send pos to robot to pick
            # command robot to move to metrology station
            # pick and place 4 times
            for i in range(4):
                if i == 0:
                    # pick from right-side place
                    pass 
                else:
                    # pick from left-side place --> rotate 45 deg --> place
                    pass 
                # measure and inspect the screw
                
                # if the screw is invalid --> break loop --> move it to NG box
                ret = False 
                return ret 
                
            # no invalid screw is found --> move it to OK box
        
        return ret  
    
    def control_abb_robot(self):
        pass 
        

def main():
    rclpy.init()
    main_node = MainStationNode()
    rclpy.spin(main_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

        
    

