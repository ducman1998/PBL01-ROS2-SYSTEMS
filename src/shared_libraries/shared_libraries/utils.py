import matplotlib.pyplot as plt
import numpy as np
import cv2 
import os 
from glob import glob
from tqdm import tqdm


class ErrorCode():
    NOT_INIT_CAMERA = -1
    SUCCESS = 0
    CAPTURE_ERROR = 1
    PROCESS_ERROR = 2
    UNKNOWN_ERROR = 3
    
class Action():
    CAPTURE_ONLY = 0
    PROCESSING = 1
    SETUP_TEMPLATE = 2 
    
