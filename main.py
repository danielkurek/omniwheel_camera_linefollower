import time
import threading
from typing import Any, Self, override
import numpy as np
import cv2
from picamera2 import Picamera2, Preview
from libcamera import Transform
import RPi.GPIO as GPIO
import os
import shutil
from PCA9685_smbus2 import PCA9685
from gpiozero import DigitalOutputDevice
import math

class ImageStreamer:
    def __init__(self):
        self.img_width: int|None = None
        self.img_height: int|None = None
        self.running = False
    
    def read_frame(self):
        raise NotImplementedError()
    
    def start(self):
        raise NotImplementedError()
    
    def stop(self):
        raise NotImplementedError()

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

class MotorsConfig():
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

class OmniwheelController():
    def __init__(self, motors_config: MotorsConfig, pwm_interface: PCA9685, max_speed, verbose=False):
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

    def start(self):
        self.running = True    
    
    def stop(self):
        self._pwm.set_all_pwm(0,0)
        for fw,bw in zip(self.fws, self.bws):
            fw.off()
            bw.off()
        self.running = False

class LineDetector:
    def __init__(self, stream: ImageStreamer, img_crop=(1.0, 1.0)):
        self.stream = stream
        self.running = False

        assert self.stream.img_height is not None and self.stream.img_width is not None
        self.img_crop = (int(img_crop[0] * self.stream.img_height), int(img_crop[1] * self.stream.img_height))

        self.last_img = None
        self.last_roi = None
        self.last_roi_binary = None
        self.last_vector = None, None, None, None

    def start(self):
        self.stream.start()
        self.running = True
    def stop(self):
        self.stream.stop()
        self.running = False
    
    def _preprocess_image(self, gray, crop_size):
        # We coud blur before threshold and clean afterwards
        # blur = cv2.GaussianBlur(roi, (5,5), 0)
        # binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
        # binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 1)

        height, width = gray.shape[:2]
        new_width, new_height = crop_size
        width_diff = width - new_width
        roi = gray[int(height - new_height):height, width_diff//2:width-width_diff//2]
        
        # Correct unpacking order
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        
        # 2. Adaptive Threshold (Keep existing logic)
        roi_bin_inv = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV, 201, 4)
        
        # 3. Morphological CLOSE to fill black "holes" inside the white line
        # This fixes the "hollow" line issue
        kernel = np.ones((5,5), np.uint8)
        roi_bin_inv = cv2.morphologyEx(roi_bin_inv, cv2.MORPH_CLOSE, kernel)

        # maybe add additional filtering by contour area
        return roi_bin_inv, roi
    
    def _middle_vector(self, binary):
        # 1) Moments (single pass, O(N))
        M = cv2.moments(binary, binaryImage=True)
        
        # TODO: better detection of empty frame (without line)
        if M["m00"] == 0:
            return None

        # m10 = suma de las coordinadas horizontales
        # m01 = suma de las coordinadas verticales
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # 2) Covariance of white pixel distribution (from central moments)
        mu20 = M["mu20"] / M["m00"]
        mu02 = M["mu02"] / M["m00"]
        mu11 = M["mu11"] / M["m00"]
        cov = np.array([[mu20, mu11],
                        [mu11, mu02]])

        # 3) Principal axis via eigen decomposition (largest eigenvalue)
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, np.argmax(vals)]      # shape (2,), unit vector in image coords (x right, y down)
        dx, dy = float(v[0]), float(v[1])

        return cx, cy, dx, dy
    
    def _draw_debug_img(self, image: np.ndarray, cx: float, cy: float, dx: float, dy: float) -> np.ndarray:
        # We need a color image to draw in color
        if len(image.shape) == 2:
            debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            debug_img = image.copy() # Make a copy

        cx_int, cy_int = int(cx), int(cy)
        cv2.circle(debug_img, (cx_int, cy_int), 5, (0, 0, 255), -1) # Red centroid

        p2_x = int(cx_int + dx * 250) # Scale the vector by 50px
        p2_y = int(cy_int + dy * 250)
        cv2.line(debug_img, (cx_int, cy_int), (p2_x, p2_y), (255, 0, 0), 2) # Blue line
        
        return debug_img
    
    def get_debug_images(self):
        return self.last_img, self.last_roi, self.last_roi_binary, self._draw_debug_img(self.last_roi, *self.last_vector)

    def save_debug_images(self, save_dir: str, frame_num: int):
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        labels = "gray", "roi", "bin", "vector"
        imgs = self.get_debug_images()
        
        for label, img in zip(labels, imgs):
            if img is not None:
                # Create a unique filename, e.g., "debug_frames/frame_001_Mask.jpg"
                filename = os.path.join(save_dir, f"frame_{frame_num:04d}_{label}.jpg")
                cv2.imwrite(filename, img)
    
    def calculate_line_vector(self):
        if not self.running:
            return None
        gray = self.stream.read_frame()
        roi_binary, roi = self._preprocess_image(gray, self.img_crop)
        
        vector = self._middle_vector(roi)

        self.last_img = gray
        self.last_roi = roi
        self.last_roi_binary = roi_binary

        if vector is not None:
            cx, cy, dx, dy = vector
            self.last_vector = cx, cy, dx, dy
            return cx, cy, dx, dy
        
        return None

class Robot():
    def __init__(self, wheel_controller: OmniwheelController, line_detector: LineDetector, verbose=False):
        self.verbose = verbose
        self.wheel_controller = wheel_controller
        self.line_detector = line_detector
        
        # okay
        self.max_physical_speed = 1.0

        # -- Debugging --
        self.frame_count = 0
        self.debug_save_dir = "debug_frames" # Folder to save images in
        self._clean_debug_dir()
        
        # -- Constants --

        self.linearSpeed = 1                      # m/s linear speed along (dx,dy)
        self.angularVelocity = 0.0                  # rad/s (spin). Set >0 to rotate CCW
        
        # PID constants
        self.kp = 0.05
        self.ki = 0.0
        self.kd = 0.1
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()  
        
    def pid_correction(self, error):
        current_time = time.time()
        delta_time = current_time - self.last_time
        
        # Avoid division by zero on first run
        if delta_time <= 0:
            delta_time = 0.001

        # 1. Proportional term
        P = self.kp * error

        # 2. Integral term (accumulation of error)
        self.integral += error * delta_time
        I = self.ki * self.integral

        # 3. Derivative term (rate of change)
        delta_error = error - self.prev_error
        D = self.kd * (delta_error / delta_time)

        # Update state for next loop
        self.prev_error = error
        self.last_time = current_time

        return P + I + D
    
    def _clean_debug_dir(self):
        """Deletes the debug directory and recreates it empty."""
        if os.path.exists(self.debug_save_dir):
            shutil.rmtree(self.debug_save_dir)  # Deletes dir and all contents
        os.makedirs(self.debug_save_dir, exist_ok=True) # Creates fresh dir
        print(f"[INFO] Cleared and renewed debug directory: {self.debug_save_dir}")
          
    def stop_all(self):
        print(" EMERGENCY STOP: Halting all systems.")
        self.wheel_controller.stop()
        self.line_detector.stop()

    def run(self):
        # --- CONFIGURATION ---
        MAX_LOST_FRAMES = 100000  # If line lost for 10 frames, STOP.
        lost_counter = 0
        
        print("Starting Camera Thread...")
        self.line_detector.start()
        time.sleep(1)

        try:
            print("Robot Loop Started. Press Ctrl+C to stop.")
            while True:
                # 1. GET IMAGE (Instant non-blocking read)
                start = time.time()
                vector = self.line_detector.calculate_line_vector()
                if vector is None:
                    print("Could not detect a line")
                    lost_counter += 1
                    if lost_counter > MAX_LOST_FRAMES:
                        print("Line lost for too long! Safety Stop.")
                        self.stop_all()
                        break
                    continue
                cx, cy, dx, dy = vector
                
                self.frame_count += 1
                    
                # --- 4. DEBUG SAVING (every 10th frame) ---
                if self.frame_count % 100 == 0:
                    print(f"Saving debug frames for frame {self.frame_count}...")
                    self.line_detector.save_debug_images(self.debug_save_dir, self.frame_count)
                
                # 3. CONTROL LOOP

                # Calculate Error
                # error = CENTER_X - self.cx
                
                # PID Calc
                # correction = self.pid_correction(error)
                # self.angularVelocity = -correction * 0.01

                # Drive Motors
                angle = math.atan2(dy,dx)
                self.angularVelocity = -1 * (0.5 * math.pi - angle)
                vx, vy = -self.linearSpeed * dx, self.linearSpeed * dy
                v = np.array([vx, vy, self.angularVelocity], dtype=float)
                self.wheel_controller.set_vector(v)
                
                end = time.time()
                print(f"time = {end-start}")
        except KeyboardInterrupt:
            print("\nUser Interrupted (Ctrl+C)")
        except Exception as e:
            print(f"\nCRITICAL ERROR: {e}")
        finally:
            # 4. CLEANUP (Guaranteed to run)
            self.stop_all()
            print("Cleanup complete.")

def main():
    pwm = PCA9685.PCA9685(interface=1)
    pwm.set_pwm_freq(1600)
    pwm.set_all_pwm(0,0)
    wheel_distance = 0.09 # maybe change to 1
    wheel_radius = 0.045 # maybe change to 1
    motors_config = MotorsConfig() \
        .add_motor(angle = 240, distance = wheel_distance, pwm_channel = 0, pin_fw=25, pin_bw=24, wheel_radius=wheel_radius) \
        .add_motor(angle = 120, distance = wheel_distance, pwm_channel = 3, pin_fw=22, pin_bw=23, wheel_radius=wheel_radius) \
        .add_motor(angle =   0, distance = wheel_distance, pwm_channel = 1, pin_fw=26, pin_bw=27, wheel_radius=wheel_radius)
    wheel_controller = OmniwheelController(motors_config, pwm, 2000)

    stream = PicameraStream(640,480)
    line_detector = LineDetector(stream, img_crop=(0.4, 0.5))

    robot = Robot(wheel_controller, line_detector)
    robot.run()	

if __name__ == "__main__":
    main()