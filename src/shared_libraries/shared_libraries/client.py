import sys
import rclpy
import ros2_numpy as rnp
import numpy as np
import cv2
import json 
import os 
from rclpy.node import Node
from datetime import datetime 
from shared_interfaces.srv import CameraCapture, SetupCamera
from shared_libraries.utils import Action, ErrorCode


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AsyncClient(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        if "metrology" in node_name:
            self.capture_cli = self.create_client(CameraCapture, 'm_capturing_serivce')
            self.setup_cli = self.create_client(SetupCamera, 'm_setup_cam_service')
        else:
            self.capture_cli = self.create_client(CameraCapture, 's_capturing_serivce')
            self.setup_cli = self.create_client(SetupCamera, 's_setup_cam_service')
        
        while not self.capture_cli.wait_for_service(timeout_sec=1.0) or not self.setup_cli.wait_for_service(timeout_sec=1.0): 
            self.get_logger().info('Service not available, trying again...')
        
        self.node_name = node_name
        self.req_capture = CameraCapture.Request()
        self.req_setup = SetupCamera.Request()

    def send_capture_request(self, dir_path: str, save_to_dir: bool, timeout: int, action: Action):
        self.req_capture.dir_path = dir_path
        self.req_capture.save_to_dir = save_to_dir
        self.req_capture.timeout = timeout
        self.req_capture.action = action
        
        future = self.capture_cli.call_async(self.req_capture)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        
        print(f"Error code: {resp.e_code}")
        if resp and resp.image.data:
            image_np = rnp.numpify(resp.image)
            if isinstance(resp.image_filename, str) and len(resp.image_filename) > 0: 
                filename = f'client_{resp.image_filename}'
            else:
                if "metrology" in self.node_name:
                    current_time = datetime.now().strftime("client_metrology_srv_%Y-%m-%d_%H-%M-%S.png")   
                else:
                    current_time = datetime.now().strftime("client_sorting_srv_%Y-%m-%d_%H-%M-%S.png")
                filename = f"{current_time}.png"  # Change the extension as needed
                
            cv2.imwrite(f'{dir_path}/{filename}', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            self.get_logger().info(f'Result saved as {filename}')
        else:
            self.get_logger().info('No image received in response.')
            
    # below function aims to detect screw -> classify -> return valid screw positions
    # ONLY for Sorting station
    def process_sorting_station(self, dir_path: str, save_to_dir: bool, timeout: int, counter: int=0):
        self.req_capture.dir_path = dir_path
        self.req_capture.save_to_dir = save_to_dir
        self.req_capture.timeout = timeout
        self.req_capture.action = Action.PROCESSING
        
        future = self.capture_cli.call_async(self.req_capture)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        
        print(f"Recieved Ecode: {resp.e_code}")
        if resp and resp.image.data:
            self.get_logger().info(f"Detected screws in response msg: {resp.detected_screws}")
            detected_screws = json.loads(resp.detected_screws)
            image_np = rnp.numpify(resp.image)
            if isinstance(resp.image_filename, str) and len(resp.image_filename) > 0: 
                filename = f'client_{resp.image_filename}'
            else:
                current_time = datetime.now().strftime("client_sorting_srv_%Y-%m-%d_%H-%M-%S.png")
                filename = f"{current_time}.png"  # Change the extension as needed
                
            os.makedirs(os.path.join(OUTPUT_DIR, f"processed_{counter}"), exist_ok=True)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"processed_{counter}", filename), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            self.get_logger().info(f'Result saved as {filename}')
            return detected_screws
        else:
            self.get_logger().info('No image received in response.')
            return [] 
        
    # below function aims to measure screw dimensions
    # ONLY for Metrology station
    def process_metrology_station(self, dir_path: str, save_to_dir: bool, timeout: int, counter: int=0, step: int=0):
        self.req_capture.dir_path = dir_path
        self.req_capture.save_to_dir = save_to_dir
        self.req_capture.timeout = timeout
        self.req_capture.action = Action.PROCESSING
        
        future = self.capture_cli.call_async(self.req_capture)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        
        print(f"Recieved Ecode: {resp.e_code}")
        if resp and resp.e_code == ErrorCode.SUCCESS and resp.image.data:
            image_np = rnp.numpify(resp.image)
            filename = datetime.now().strftime(f"step_{step}_%Y-%m-%d_%H-%M-%S.png")
            saved_dir = os.path.join(OUTPUT_DIR, f"processed_{counter}")
            os.makedirs(saved_dir, exist_ok=True)
            cv2.imwrite(os.path.join(saved_dir, filename), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            self.get_logger().info(f'Result saved as {filename} in {saved_dir}') 
            measured_values = json.loads(resp.measured_values)
            if resp.inspection_status:
                return True, measured_values
            else:
                return False, measured_values 
        else:
            self.get_logger().info('No image received in response.')
            return False, None 
        
            
    def send_setup_camera_request(self, cam_serial_number: str):
        self.req_setup.cam_id = cam_serial_number
        future = self.setup_cli.call_async(self.req_setup)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp.status is True:
            print("Camera Initialization is now complete!")
            return True 
        else:
            print("Camera Initialization fail!")
            return False