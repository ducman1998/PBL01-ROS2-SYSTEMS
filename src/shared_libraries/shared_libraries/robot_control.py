import socket 
import time 


class RobotController():
    def __init__(self, robot_ip: str, robot_port: int):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.s = self.connectRobot()
    
    def connectRobot(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.robot_ip, self.robot_port))
        return s 
    
    def send_msg(self, msg_str):
        t1 = time.time()
        raw_bytes = msg_str.encode()
        print(f"\tSending the message: {msg_str}")
        try:
            self.s.sendall(raw_bytes)
        except:
            self.s = self.connectRobot()
            self.s.sendall(raw_bytes)
            
        try:
            print("\tWaiting response from Robot...")
            data = self.s.recv(1024)
            if data:
                msg = data.decode("utf-8")
                print(f'the answer is: "{msg}"')
                if msg == "Done":
                    t2 = time.time()
                    print(f"[RobotABB] Processed msg: {msg_str}, total time: {round(t2-t1,4)}s")
                    return True 
        except:
            pass 
        
        t2 = time.time()
        print(f"[RobotABB] Processed msg: {msg_str}, total time: {round(t2-t1,4)}s")
        return False 
    
    def moveHomePos(self):
        return self.send_msg("MoveHomePos")
    
    def pickScrew(self, posX: int, posY: int):
        return self.send_msg(f"PickScrew_{posX}_{posY}")
    
    def rotateScrew(self, degree: int = 45):
        return self.send_msg(f"RotateScrew_{degree}") 
    
    def throwScrew(self, inspectionStatus: bool):
        self.send_msg(f"ThrowScrew_{inspectionStatus}") 
        
    def terminate(self):
        self.send_msg("Terminate")
    