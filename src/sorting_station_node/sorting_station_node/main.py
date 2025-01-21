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
from shared_libraries.utils import intersection, filter_nearby_intersections
from shared_libraries.utils import perspective_transform, calculate_angle, find_bounding_rectangle


class SortingStationNode(Node):
    def __init__(self):
        self.save_to_dir = False 
        self.saved_dir = None 
        self.device = None 
        self.device_id = None 
        self.format = itala.PfncFormat_RGB8
        self.encoding = None 

        super().__init__('sorting_station_node')
        self.capture_service = self.create_service(CameraCapture, 's_capturing_serivce', self.capture_from_cam_callback)
        self.setup_cam_service = self.create_service(SetupCamera, 's_setup_cam_service', self.connect_camera_callback)
        
        # init classifier (MLP)
        resnet18 = models.resnet18(pretrained=True)
        # Remove the fully connected layer to get feature vectors
        self.resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])
        self.resnet18.eval()  # Set the model to evaluation mode

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to 224x224
            transforms.ToTensor(),          # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        # load trained model
        self.trained_model = joblib.load('models/mlp_classifier_R18_512_features_model_v01.pkl')
        
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
        else:
            status, detected_screws = self.process_image(image_np) # RGB image
            if status is True:
                resp.e_code = ErrorCode.SUCCESS
                resp.detected_screws = json.dumps(detected_screws)
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
                image = Image.fromarray(screw)
                input_tensor = self.preprocess(image).unsqueeze(0)  # Preprocess and add batch dimension

                with torch.no_grad():
                    features = self.resnet18(input_tensor).squeeze().numpy()  # Extract and convert to NumPy array
                    pred_class = self.trained_model.predict(features.reshape(1,-1))[0]
                    predicted_classes[k] = pred_class
            return predicted_classes
        except:
            return 
        
    def detect_screws(self, captured_image: np.ndarray):
        return_dict = {}
        
        # captured_image should be in RGB format
        im_g = cv2.cvtColor(captured_image, cv2.COLOR_RGB2GRAY)
        im_bi = np.zeros_like(im_g)
        im_bi[im_g >= 125] = 255
        cv2.imwrite("binary_image.png", im_bi)
    
        # get biggest blob
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(im_bi)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip the background (label 0)
        largest_component = (labels == largest_label).astype(np.uint8) * 255
        
        # fill screw holes
        # des = cv2.bitwise_not(largest_component)
        cnts, hier = cv2.findContours(largest_component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            cv2.drawContours(largest_component,[cnt],0,255,-1)
        
        contours, _ = cv2.findContours(largest_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Loop through each contour to fit a polygon
        contour = contours[0] # get the first and only contour
        # Approximate the polygon with a precision proportional to the perimeter
        hull = cv2.convexHull(contour)
        cv2.fillPoly(largest_component, [hull], color=255)
        
        corners = find_bounding_rectangle(largest_component)
        
        if len(corners) == 4:
            screw_im = perspective_transform(captured_image, corners, im_size=416*3)
            cv2.imwrite("screw_image.png", screw_im[:,:,::-1])
            for r in range(3):
                for c in range(3):
                    crop_im = screw_im[r*416:(r+1)*416, c*416:(c+1)*416]
                    xPos = 3-r 
                    yPos = 3-c 
                    return_dict[f"{xPos}.{yPos}"] = crop_im.copy() 
            return return_dict
        else:
            print(f"[ERROR] Cannot detect 4 corners in this image.")
            return return_dict
                
    def process_image(self, screw_im_np):
        # image processing
        detected_screws_dict = self.detect_screws(screw_im_np)
        self.get_logger().info(f"Detected screws: {detected_screws_dict}")
        if len(detected_screws_dict) == 0:
            self.get_logger().warn(f"Not found any screws on the sorting station now.")
            return False, None  
        else:
            predicted_classes_dict = self.classify_screws(detected_screws_dict)
            if len(predicted_classes_dict) == 0:
                self.get_logger().warn(f"Cannot infer classification model now.")
                return False, None 
            
            detected_screws = []
            for k, klass in predicted_classes_dict.items():
                if klass == 1:
                    detected_screws.append(k)
            return True, detected_screws 
        

def main():
    rclpy.init()
    minimal_service = SortingStationNode()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

        
    

