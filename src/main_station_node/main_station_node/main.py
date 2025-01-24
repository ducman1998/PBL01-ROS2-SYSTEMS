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
from shared_libraries.metrology_utils import statistic_on_measured_values
from shared_libraries.criteria import REF_VALUES, DOFF

    
DIR_PATH = "logged_client_images"
SAVE_TO_DIR = True 
TIMEOUT = 1000
ACTION = 1 # CAPTURE_ONLY: 0, PROCESSING: 1, SETUP_TEMPLATE: 2
CAM_SORTING_SERIAL_NUMBER = '600742'
CAM_METROLO_SERIAL_NUMBER = '600590'
ROBOT_HOST = "192.168.125.1"  # The server's hostname or IP address
ROBOT_PORT = 1025  # The port used by the server
OUTPUT_DIR = "outputs"
TEMPLATE_DIR = "templates"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

class MainStationNode(Node):
    def __init__(self):
        super().__init__('main_station_node')
        self.get_logger().info("Main node initialized")
        
        ret1 = self.init_system()
        if ret1:
            ret2 = input("Do you want to start measuring processes? (0: No & 1: Yes) \nYour answer is: ")
            if ret2 == "1" or ret2 == "yes":
                self.start_measuring_processes()
            self.get_logger().info("Exited the program.")
        else:
            self.get_logger().info("Cannot connect to the ABB Robot. Exited the program.")
        
            
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

        ret = input("Do you want to re-capture template images? (0: No & 1: Yes) \nYour answer is: ")
        if ret == "1" or ret == "yes":
            self.metrology_cam_cli.send_capture_request(DIR_PATH, SAVE_TO_DIR, TIMEOUT, 2)
            self.sorting_cam_cli.send_capture_request(DIR_PATH, SAVE_TO_DIR, TIMEOUT, 2)
        else:
            self.metrology_cam_cli.send_capture_request(DIR_PATH, SAVE_TO_DIR, TIMEOUT, 0)
            self.sorting_cam_cli.send_capture_request(DIR_PATH, SAVE_TO_DIR, TIMEOUT, 0)

        try:
            self.robot = RobotController(ROBOT_HOST, ROBOT_PORT)
        except:
            self.get_logger().error(f"Cannot connect to the robot. Exit the program now. Error: {traceback.format_exc()}")
            return False 
        
        self.get_logger().info("Connected to cameras.")
        return True  
    
    def start_measuring_processes(self):
        self.robot.moveHomePos()
        self.get_logger().info("Starting to process screws...")
        step_count = 0
        while True:
            detected_screws = self.sorting_cam_cli.process_sorting_station(DIR_PATH, SAVE_TO_DIR, TIMEOUT, step_count)
            if len(detected_screws) == 0:
                self.get_logger().info("Not found any screws now. Exit the program and return robot to Home positon!")
                self.robot.terminate()
                return 
            
            first_screw = detected_screws[0]
            xPos = int(first_screw.split(".")[0])
            yPos = int(first_screw.split(".")[1])
            self.robot.pickScrew(xPos, yPos)
            
            status = True 
            view_degree_count = 0
            measured_values_list = []
            # process the screw at Metrology station here
            try:
                inspection_status, measured_values = self.metrology_cam_cli.process_metrology_station(DIR_PATH, SAVE_TO_DIR, TIMEOUT, step_count, view_degree_count)
                if measured_values is not None:
                    measured_values_list.append(measured_values)
                if inspection_status is False:
                    status = False 
            except:
                self.get_logger().error(f"Got error when measuring: {traceback.format_exc()}")
                
            for i in range(5):
                if not status:
                    self.robot.throwScrew(0)
                    break
                if i == 4:
                    self.robot.throwScrew(1)
                
                self.robot.rotateScrew(45)
                view_degree_count += 45
                # process the screw at Metrology station here
                try:
                    inspection_status, measured_values = self.metrology_cam_cli.process_metrology_station(DIR_PATH, SAVE_TO_DIR, TIMEOUT, step_count, view_degree_count)
                    if measured_values is not None:
                        measured_values_list.append(measured_values)
                    if inspection_status is False:
                        status = False 
                except:
                    self.get_logger().error(f"Got error when measuring: {traceback.format_exc()}")
                    
            if len(measured_values_list):
                stat_df = statistic_on_measured_values(measured_values_list, REF_VALUES, doff=DOFF)
                saved_dir = os.path.join(OUTPUT_DIR, f"processed_{step_count}")
                stat_df.to_csv(os.path.join(saved_dir, f"measured_values_stats.csv"), index=False)
                
            step_count += 1

def main():
    rclpy.init()
    main_node = MainStationNode()
    rclpy.spin(main_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

        
    

