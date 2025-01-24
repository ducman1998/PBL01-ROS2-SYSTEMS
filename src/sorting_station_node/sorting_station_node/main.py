import sensor_msgs
import numpy as np
import ros2_numpy as rnp
import rclpy
import cv2
import ctypes
import joblib
import json 
import time
import os
import traceback
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from rclpy.node import Node
from itala import itala
from datetime import datetime
from PIL import Image
from shared_interfaces.srv import CameraCapture, SetupCamera
from shared_libraries.utils import Action, ErrorCode
from shared_libraries.sorting_utils import detect_screw_region, detect_screws
from shared_libraries.classifier import ScrewClassifier 


OUTPUT_DIR = "outputs"
TEMPLATE_DIR = "templates"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

class SortingStationNode(Node):
    def __init__(self):
        self.save_to_dir = False 
        self.saved_dir = None 
        self.device = None 
        self.device_id = None 
        self.format = itala.PfncFormat_RGB8
        self.encoding = None
        
        self.template_fp = os.path.join(TEMPLATE_DIR, "sorting_station_template.png")
        self.template_rgb = None 
        self.background_rgb = None 
        self.corners = None 

        super().__init__('sorting_station_node')
        self.capture_service = self.create_service(CameraCapture, 's_capturing_serivce', self.capture_from_cam_callback)
        self.setup_cam_service = self.create_service(SetupCamera, 's_setup_cam_service', self.connect_camera_callback)
        
        self.classifier = ScrewClassifier("mlp_classifier_PblHogResnet18_861Features_model_v02.pkl", "scaler.pkl")
        
        if os.path.isfile(self.template_fp):
            self.template_rgb = cv2.cvtColor(cv2.imread(self.template_fp), cv2.COLOR_BGR2RGB)
            self.background_rgb, self.corners = detect_screw_region(self.template_rgb)
            
        self.get_logger().info("Service initialized")

    def capture_from_cam_callback(self, req, resp):
        self.get_logger().info(f"[request] Capture: dir_path ({req.dir_path}), save_to_dir ({req.save_to_dir}), timeout ({req.timeout})")
        if self.device is None:
            resp.e_code = ErrorCode.NOT_INIT_CAMERA
            return resp 
        image_np = self.capture(req.timeout)
        if image_np is None:
            resp.e_code = ErrorCode.CAPTURE_ERROR
            return resp 

        if req.save_to_dir:
            os.makedirs(req.dir_path, exist_ok=True)
            current_time = datetime.now().strftime("sorting_srv_%Y-%m-%d_%H-%M-%S")
            filename = f"{current_time}.png"  # Change the extension as needed
            resp.image_filename = filename
            file_path = os.path.join(req.dir_path, filename)
            if self.encoding == "rgb8":
                cv2.imwrite(file_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(file_path, image_np)

        resp.image = rnp.msgify(sensor_msgs.msg.Image, image_np, encoding=self.encoding)
        if int(req.action) == Action.CAPTURE_ONLY:
            resp.e_code = ErrorCode.SUCCESS
            return resp
        
        elif int(req.action) == Action.SETUP_TEMPLATE:
            background_cubid_rgb, corners = detect_screw_region(image_np, binary_thresh=200)
            if background_cubid_rgb is not None and len(corners) == 4:
                self.background_rgb = background_cubid_rgb
                self.corners = corners 
                self.template_rgb = image_np
                cv2.imwrite(self.template_fp, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(TEMPLATE_DIR, "background_cubid.png"), cv2.cvtColor(background_cubid_rgb, cv2.COLOR_RGB2BGR))
                resp.e_code = ErrorCode.SUCCESS
                self.get_logger().info("Processed template image for the Sorting station successfully!")
            else:
                resp.e_code = ErrorCode.PROCESS_ERROR
                self.get_logger().error("Cannot process template image of the Sorting station!")
            return resp 
                
        else:
            status, detected_screws, viz_im = self.process_image(image_np) # RGB image
            if status is True:
                resp.e_code = ErrorCode.SUCCESS
                resp.detected_screws = json.dumps(detected_screws)
                resp.image = rnp.msgify(sensor_msgs.msg.Image, viz_im, encoding=self.encoding)
            else:
                resp.e_code = ErrorCode.PROCESS_ERROR
                resp.detected_screws = "[]"
            return resp 
    
    def connect_camera_callback(self, req, resp):
        cam_serial_number = req.cam_id
        self.device = self.init_camera(cam_serial_number) 
        if self.device is None:
            resp.status = False 
            self.device_id = None 
        else:
            resp.status = True 
            self.device_id = cam_serial_number
        return resp 
    
    def init_camera(self, cam_serial_number):
        device = None 
        self.encoding = None 
        if self.device is not None:
            return self.device 
        
        try:                
            system = itala.create_system()
            device_infos = system.enumerate_devices(700)
            if len(device_infos) == 0:
                self.get_logger().info("No network interfaces found. Please check the network setup and connect again!")
            else:
                self.get_logger().info("Got a list camera below:")
                for i,dev in enumerate(device_infos):
                    dev:itala.DeviceInfo #type hint (does not type the variable!)
                    self.get_logger().info(f'\t{i}\n\tdisplay_name {dev.display_name}\n\tserial {dev.serial_number}\n\tIP: {dev.ip_address}')
                    if int(cam_serial_number) == int(dev.serial_number):
                        device = system.create_device(dev) 
                        return device
                    
                if device is None:
                    self.get_logger().error(f'serial number {cam_serial_number} not found!')
                    return 
                
        except Exception as e:
            self.get_logger().error(f"Error initializing device: {e}")
            self.get_logger().error(traceback.print_exc())
            return 
    
    def capture(self, timeout: int) -> np.ndarray:
            try:
                self.device.start_acquisition(2, 2)
                self.get_logger().info("acquisition started")
                time.sleep(0.5)
                if timeout is None or timeout <= 0 or isinstance(timeout, str):
                    timeout = 1000
                image = self.device.get_next_image(timeout)
            except:
                self.get_logger().info("No image available.")
                return 
            
            if image.is_incomplete:
                self.get_logger().info("Incomplete image received.")
                self.get_logger().info("Bytes filled: " + str(image.bytes_filled))
                self.get_logger().info("Timestamp: " + str(image.timestamp))
                return 

            height = image.height
            width = image.width
            fmt = image.pixel_format
            size = width * height
            buffer = image.get_data()
            
            if fmt == itala.PfncFormat_Mono8:
                channels = 1
                self.encoding = 'mono8'
            else:
                if fmt == itala.PfncFormat_RGB8:
                    self.encoding = 'rgb8'
                    channels = 3
                else:
                    self.get_logger().error(f'format unsuported! only mono8 or rgb8!')
                    return 
            
            print(f"Captured image's format: {itala.get_pixel_format_description(fmt)} ~ Valid format: {itala.get_pixel_format_description(self.format)}")
            print(f"Image size: w ({width}), h ({height}), c ({channels})")
            
            p = (ctypes.c_uint8 * size * channels).from_address(int(buffer))
            nparray = np.ctypeslib.as_array(p)
            image_np = nparray.reshape((height, width, channels)).squeeze().copy()
            self.device.stop_acquisition()  
            image.dispose()
            return image_np
    
    def classify_screws(self, input_screws: dict):
        # input images must be in RGB format
        predicted_classes = {}
        try:
            for k, screw in input_screws.items():
                    predicted_classes[k] = self.classifier.classify_screw(screw)
            return predicted_classes
        except:
            return 
                
    def process_image(self, screw_im_np):
        # image processing
        detected_screws_dict, viz_im = detect_screws(screw_im_np, self.corners, self.background_rgb, min_changed_area=5000)
        self.get_logger().info(f"Detected {len(detected_screws_dict)} screws: {list(detected_screws_dict.keys())}")
        if len(detected_screws_dict) == 0:
            self.get_logger().warn(f"Not found any screws on the sorting station now.")
            return False, None, None 
        else:
            predicted_classes_dict = self.classify_screws(detected_screws_dict)
            if len(predicted_classes_dict) == 0:
                self.get_logger().warn(f"Cannot infer classification model now.")
                return False, None, None 
            
            detected_screws = []
            for k, klass in predicted_classes_dict.items():
                if klass == 1:
                    detected_screws.append(k)
            self.get_logger().info(f"Valid screws: {detected_screws}")
            return True, detected_screws, viz_im
        

def main():
    rclpy.init()
    minimal_service = SortingStationNode()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

        
    

