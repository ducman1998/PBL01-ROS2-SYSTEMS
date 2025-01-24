import sensor_msgs
import numpy as np
import ros2_numpy as rnp
import rclpy
import cv2
import ctypes
import time
import os
import traceback
import json
from rclpy.node import Node
from itala import itala
from datetime import datetime
from shared_interfaces.srv import CameraCapture, SetupCamera
from shared_libraries.utils import Action, ErrorCode
from shared_libraries.metrology_utils import measure_screw_dimensions, inspect_screw_dimensions
from shared_libraries.criteria import REF_VALUES, DOFF


SCALING_FACTOR = 0.03460213    
OUTPUT_DIR = "outputs"
TEMPLATE_DIR = "templates"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

class MetrologyStationNode(Node):
    def __init__(self):
        self.save_to_dir = False 
        self.saved_dir = None 
        self.device = None 
        self.device_id = None 
        self.format = itala.PfncFormat_Mono8
        self.encoding = None 
        
        self.template_fp = os.path.join(TEMPLATE_DIR, "metrology_station_template.png")
        self.background_rgb = None 
        if os.path.isfile(self.template_fp):
            self.background_rgb = cv2.cvtColor(cv2.imread(self.template_fp), cv2.COLOR_BGR2RGB)
        
        super().__init__('metrology_station_node')
        self.capture_service = self.create_service(CameraCapture, 'm_capturing_serivce', self.capture_from_cam_callback)
        self.setup_cam_service = self.create_service(SetupCamera, 'm_setup_cam_service', self.connect_camera_callback)
        self.get_logger().info("Service initialized")

    def capture_from_cam_callback(self, req, resp):
        self.get_logger().info(f"[request] Capture: dir_path ({req.dir_path}), save_to_dir ({req.save_to_dir}), timeout ({req.timeout})")
        if self.device is None:
            resp.e_code = ErrorCode.NOT_INIT_CAMERA
            return resp 
        t1 = time.time()
        image_np = self.capture(req.timeout)
        t2 = time.time()
        if image_np is None:
            resp.e_code = ErrorCode.CAPTURE_ERROR
            return resp 

        if req.save_to_dir:
            os.makedirs(req.dir_path, exist_ok=True)
            current_time = datetime.now().strftime("metrology_srv_%Y-%m-%d_%H-%M-%S")
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
            self.background_rgb = image_np
            cv2.imwrite(self.template_fp, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            resp.e_code = ErrorCode.SUCCESS
            self.get_logger().info("Processed template image for the Metrology station successfully!")
            return resp 
                
        else:
            t3 = time.time()
            try:
                inspection_status, measured_values, viz_im = self.process_image(image_np)
                resp.e_code = ErrorCode.SUCCESS
                resp.image = rnp.msgify(sensor_msgs.msg.Image, viz_im, encoding=self.encoding)
                try:
                    del measured_values["lcrest_points"]
                    del measured_values["rcrest_points"]
                    del measured_values["lroot_points"]
                    del measured_values["rroot_points"]
                except:
                    pass
                resp.measured_values = json.dumps(measured_values)
                resp.inspection_status = inspection_status
            except:
                self.get_logger().error(f"Got error when measuring the screw: {traceback.format_exc()}")
                resp.e_code = ErrorCode.PROCESS_ERROR
            t4 = time.time()
            self.get_logger().warn(f"Metrology station time: captured ({round(t2-t1,4)}s), processing time ({round(t4-t3,4)}s)")
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
                
            # override value
            self.encoding = 'rgb8'
            print(f"Captured image's format: {itala.get_pixel_format_description(fmt)} ~ Valid format: {itala.get_pixel_format_description(self.format)}")
            print(f"Image size: w ({width}), h ({height}), c ({channels})")
            
            p = (ctypes.c_uint8 * size * channels).from_address(int(buffer))
            nparray = np.ctypeslib.as_array(p)
            image_np = nparray.reshape((height, width, channels)).squeeze().copy()
            if len(image_np.shape) == 2 or (len(image_np.shape) == 3 and image_np.shape[2] == 1):
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            self.device.stop_acquisition()  
            image.dispose()
            return image_np
    
    def process_image(self, im_rgb_np):
        # measuring screw dimensions
        measured_values, viz_im = measure_screw_dimensions(self.background_rgb, im_rgb_np, SCALING_FACTOR, verbose=True)
        # screw inspection
        inspection_status, viz_im = inspect_screw_dimensions(viz_im, measured_values, REF_VALUES, doff=DOFF, scaling_factor=SCALING_FACTOR)
        self.get_logger().info(f"\ninspection_status: {inspection_status}")
        viz_im_ret = np.concatenate([cv2.rotate(im_rgb_np, cv2.ROTATE_180), viz_im], axis=1)
        return inspection_status, measured_values, viz_im_ret
        
def main():
    rclpy.init()
    minimal_service = MetrologyStationNode()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

    
    

