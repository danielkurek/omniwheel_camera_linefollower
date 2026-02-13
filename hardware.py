import threading
import time
import numpy as np

from robot import import ImageStreamer, DriveInterface

from picamera2 import Picamera2
from libcamera import Transform
from PCA9685_smbus2 import PCA9685
from gpiozero import DigitalOutputDevice

class PicameraStream(ImageStreamer):
    def __init__(self, width=640, height=480):
        super().__init__()

        self.picam2 = Picamera2()
        self.img_width = width
        self.img_height = height
        
        # Configure the camera hardware (ISP) to resize images automatically.
        # This saves a massive amount of CPU power.
        config = self.picam2.create_video_configuration(
            main={"size": (self.img_width, self.img_height), "format": "YUV420"},
            transform=Transform(vflip=True)
        )
        self.picam2.configure(config)
        
        self.picam2.start()

        self.gray_frame = None
        
        with self.picam2.controls as ctrl:
            ctrl.AnalogueGain = 2.0 
            ctrl.ExposureTime = 50000    
            
        # Start the background thread,and kill the thread if done. 
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True 
    
    @override
    def start(self):
        self.t.start()
        self.running = True
        # Block until the first frame is ready (prevents startup errors)
        while self.gray_frame is None:
            time.sleep(0.01)
        return self
    
    @override
    def stop(self):
        self.running = False
        self.picam2.stop()

    def update(self):
        while self.running:
            try:
                self.gray_frame = self.picam2.capture_array("main")
                # crop image to original size because of the YUV420 format
                # this way we will get only grayscale image
                self.gray_frame = self.gray_frame[0:self.img_height, 0:self.img_width] 
            except Exception as e:
                print(f"Camera Thread Error: {e}")
                self.running = False
    
    @override
    def read_frame(self):
        """Returns the most recent frame available."""
        return self.gray_frame

class MotorsConfig:
    def __init__(self):
        self.motors_num = 0
        self.angles: list[float] = []
        self.distances: list[float] = []
        self.wheel_radii: list[float] = []
        self.pwm_channels: list[int] = []
        self.fw_pins: list[int] = []
        self.bw_pins: list[int] = []
    
    def add_motor(self, angle:float, pwm_channel:int, pin_fw:int, pin_bw:int, distance:float = 1.0, wheel_radius:float = 1.0) -> Self:
        self.angles.append(angle)
        self.distances.append(distance)
        self.wheel_radii.append(wheel_radius)
        self.pwm_channels.append(pwm_channel)
        self.fw_pins.append(pin_fw)
        self.bw_pins.append(pin_bw)
        self.motors_num += 1
        return self

class OmniwheelController(DriveInterface):
    def __init__(self, motors_config: MotorsConfig, pwm_interface: PCA9685, max_speed, verbose=False):
        super().__init__()
        assert motors_config.motors_num == 3
        self.verbose = verbose
        self.max_speed = max_speed
        self.running = False
        self._config = motors_config
        self._pwm = pwm_interface
        self._M = self._calc_kinematic_matrix(motors_config)
        self._wheel_radius = np.array(self._config.wheel_radii) # either a single number or numpy array of size motor_num
        self.fws = [DigitalOutputDevice(pin=x) for x in self._config.fw_pins]
        self.bws = [DigitalOutputDevice(pin=x) for x in self._config.bw_pins]
        self.pwm_channels = list(motors_config.pwm_channels)

    def _calc_kinematic_matrix(self, motors_config):
        # wheel angles around the robot
        x = np.array(motors_config.distances) * np.cos(motors_config.angles)
        y = np.array(motors_config.distances) * np.sin(motors_config.angles)

        x1, x2, x3 = x
        y1, y2, y3 = y

        # normal vectors of wheels (pointing in forward direction)
        nx1, ny1 = x1, y1
        nx2, ny2 = x2, y2
        nx3, ny3 = x3, y3

        if self.verbose:
            print(f"{nx1=} {ny1=}")
            print(f"{nx2=} {ny2=}")
            print(f"{nx3=} {ny3=}")

        # kinematic matrix (linear rim speed). For this symmetric layout,
        # the rotation coupling term (ny*x - nx*y) = 1 for all three.
        M = np.array([
            [nx1, ny1, motors_config.distances[0]],
            [nx2, ny2, motors_config.distances[1]],
            [nx3, ny3, motors_config.distances[2]],
        ], dtype=float)
        return M
    
    def _calc_motor_speeds(self, driving_vector):
        #wheel linear speeds (if r=1) or angular speeds (rad/s) if you divide by real r
        w = (self._M @ driving_vector) / self._wheel_radius
        if self.verbose:
            print(f"wheel speeds: {w}")
        return w
    
    @override
    def set_vector(self, driving_vector):
        """
        `driving_vector` - [x,y,rotation]
        """
        motor_speeds = self._calc_motor_speeds(driving_vector)
        self.set_motor_speeds(motor_speeds)

    def set_motor_speed(self, motor_num, speed):
        orig_speed = speed
        speed = int(speed * 300)
        speed = max(min(speed, self.max_speed), -self.max_speed) # Clamp

        if self.verbose:
            print(f"original speed= {orig_speed}  speed clipped = {speed}")
        
        if not self.running:
            return

        if speed >= 0:
            self.fws[motor_num].on()
            self.bws[motor_num].off()
            self._pwm.set_pwm(self.pwm_channels[motor_num], 0, abs(speed))
            
        else:
            self.fws[motor_num].off()
            self.bws[motor_num].on()
            self._pwm.set_pwm(self.pwm_channels[motor_num], 0, abs(speed))
    
    def set_motor_speeds(self, motor_speeds):
        for i,speed in enumerate(motor_speeds):
            self.set_motor_speed(i, speed)
    
    @override
    def start(self):
        self.running = True    
    @override
    def stop(self):
        self._pwm.set_all_pwm(0,0)
        for fw,bw in zip(self.fws, self.bws):
            fw.off()
            bw.off()
        self.running = False
