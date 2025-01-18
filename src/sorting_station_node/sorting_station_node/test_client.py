import sys
import rclpy
import numpy as np
from shared_libraries.utils import Action, ErrorCode
from shared_libraries.client import AsyncClient


def main():
    DIR_PATH = "captured_images"
    SAVE_TO_DIR = True 
    TIMEOUT = 1000
    ACTION = 0 # CAPTURE_ONLY: 0, PROCESSING: 1
    CAM_SERIAL_NUMBER = '600742'
    
    rclpy.init()
    client = AsyncClient("sorting_client_async")
    ret = client.send_setup_camera_request(CAM_SERIAL_NUMBER)
    if ret is False:
        print("Terminate client due to be unable to setup the camera!")
    else:
        client.send_capture_request(DIR_PATH, SAVE_TO_DIR, TIMEOUT, ACTION)
    client.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()