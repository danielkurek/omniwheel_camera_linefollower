import time
import threading
import numpy as np
import cv2
from picamera2 import Picamera2, Preview
from libcamera import Transform
import RPi.GPIO as GPIO
import os
import shutil
from PCA9685_smbus2 import PCA9685
from gpiozero import DigitalOutputDevice

class PicameraStream:
    def __init__(self, width=640, height=480):
        self.picam2 = Picamera2()
        
        # Configure the camera hardware (ISP) to resize images automatically.
        # This saves a massive amount of CPU power.
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "YUV420"},
            transform=Transform(vflip=True)
        )
        self.picam2.configure(config)
        
        self.picam2.start()

        self.stopped = False
        self.gray_frame = None
        
        with self.picam2.controls as ctrl:
            ctrl.AnalogueGain = 2.0 
            ctrl.ExposureTime = 50000    
            
        # Start the background thread,and kill the thread if done. 
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True 

    def start(self):
        self.t.start()
        # Block until the first frame is ready (prevents startup errors)
        while self.gray_frame is None:
            time.sleep(0.01)
        return self

    def update(self):
        while not self.stopped:
            try:
                self.gray_frame = self.picam2.capture_array("main")
                self.gray_frame = self.gray_frame[0:480, 0:640] 
            except Exception as e:
                print(f"Camera Thread Error: {e}")
                self.stopped = True

    def read_gray(self):
        """Returns the most recent frame available."""
        return self.gray_frame

    def stop(self):
        self.stopped = True
        self.picam2.stop()

class Robot():

    def __init__(self):
        
        self.m1fw, self.m1bw = DigitalOutputDevice(pin=23), DigitalOutputDevice(pin=22)  # Motor 1
        self.m2fw, self.m2bw = DigitalOutputDevice(pin=24), DigitalOutputDevice(pin=25)  # Motor 2
        self.m3fw, self.m3bw = DigitalOutputDevice(pin=27), DigitalOutputDevice(pin=26)  # Motor 3

        self.pwm = PCA9685.PCA9685(interface=1)
        self.pwm.set_pwm_freq(1600)
        self.pwm.set_all_pwm(0,0)
        
        # okay
        self.max_physical_speed = 1.0 
        
        # dont judge, just enjoy :D
        self.motor2 = [3 ,self.m1fw, self.m1bw]
        self.motor1 = [0 ,self.m2fw, self.m2bw]
        self.motor3 = [1 ,self.m3fw, self.m3bw]
        self.motors = [self.motor1, self.motor2, self.motor3]

        # -- Debugging --
        self.frame_count = 0
        self.debug_save_dir = "debug_frames" # Folder to save images in
        self._clean_debug_dir()
        
        # -- Constants --
        self.angles = np.deg2rad([90, 210, 330])    # angles of the weels
        self.linearSpeed = 0.8                      # m/s linear speed along (dx,dy)
        self.angularVelocity = 0.0                  # rad/s (spin). Set >0 to rotate CCW
        self.wheelRadius = 0.045                      # wheel radius in meters (or set to your real radius)
        self.radius = 0.09                             # radius of robot

        # -- Runtime
        self.w = 0
        self.cx = 0
        self.cy = 0
        self.dx = 0
        self.dy = 0
        
        # PID constants
        self.kp = 0.05
        self.ki = 0.0
        self.kd = 0.1
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
        
        # camera configuration
        print("Starting Camera Thread...")
        # 320x240 is the sweet spot for speed/accuracy on Pi
        self.camera = PicameraStream(width=640, height=480).start()
        time.sleep(1)
    def preprocess_image(self, gray, th_w, th_h):
        """
        # We coud blur before threshold and clean afterwards
        # blur = cv2.GaussianBlur(roi, (5,5), 0)
        # binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
        # binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 1)
        """

        h, w = gray.shape[:2]
        roi = gray[int(h*th_h):h, int(w*th_w):int(w-w*th_w)]
        
        # Correct unpacking order
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        
        # 2. Adaptive Threshold (Keep existing logic)
        binv = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV, 101, 2)
        
        # 3. Morphological CLOSE to fill black "holes" inside the white line
        # This fixes the "hollow" line issue
        kernel = np.ones((5,5), np.uint8)
        binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, kernel)
        return binv, roi

    def middle_vector(self, binary):  # binary: 0/255, white = line
        # 1) Moments (single pass, O(N))
        M = cv2.moments(binary, binaryImage=True)
        if M["m00"] == 0:
            raise ValueError("No white pixels found")

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


    def draw_debug_info(self, image: np.ndarray, cx: float, cy: float, dx: float, dy: float) -> np.ndarray:
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

    def debug_save_images(self, images: dict, save_dir: str, frame_id: int):
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        for suffix, img in images.items():
            if img is not None:
                # Create a unique filename, e.g., "debug_frames/frame_001_Mask.jpg"
                filename = os.path.join(save_dir, f"frame_{frame_id:03d}_{suffix}.jpg")
                cv2.imwrite(filename, img)

    def get_motorW(self):
        # wheel angles around the robot
        x = self.radius * np.cos(self.angles)
        y = self.radius * np.sin(self.angles)

        x1, x2, x3 = x
        y1, y2, y3 = y

        # drive directions = perpendiculars (90 CCW): (nx, ny) = (-y, x)
        nx1, ny1 = -y1, x1
        nx2, ny2 = -y2, x2
        nx3, ny3 = -y3, x3

        # kinematic matrix (linear rim speed). For this symmetric layout,
        # the rotation coupling term (ny*x - nx*y) = 1 for all three.
        M = np.array([
            [nx1, ny1, ny1*x1 - nx1*y1],
            [nx2, ny2, ny2*x2 - nx2*y2],
            [nx3, ny3, ny3*x3 - nx3*y3],
        ], dtype=float)


        vx, vy = self.linearSpeed*self.dx, self.linearSpeed*self.dy
        v = np.array([vx, vy, self.angularVelocity], dtype=float)
        #v = np.array([0,1000,0], dtype=float)
        # wheel linear speeds (if r=1) or angular speeds (rad/s) if you divide by real r
        w = (M @ v) / self.wheelRadius
        print("wheel speeds:", w)
        return w
        
        
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
    
    def set_single_motor(self, pwm_channel, infw, inbw, speed):
        speed = max(min(speed, 100), -100) # Clamp
        speed /= 100
        speed = speed * 1000
        speed = int(speed)
        print("speed clipped = ", speed)
            
        if speed >= 0:
            infw.on()
            inbw.off()
            self.pwm.set_pwm(pwm_channel, 0, abs(speed))
            
        else:
            infw.off()
            inbw.on()
            self.pwm.set_pwm(pwm_channel, 0, abs(speed))
    
    # digitaloutout device inpins x2 , on or of freq 42k | set duty cycle, value 0.5 + pid
    def apply_wheel_speeds(self, w):
        """
        Takes the calculated wheel speeds (w), normalizes them, 
        and sends PWM signals to the motors.
        """

        for motor_info, speed in zip(self.motors, w):
            pwm_val = (speed / self.max_physical_speed) * 100
            
            # motor_info = [pwm_object, pin_a, pin_b]
            self.set_single_motor(motor_info[0], motor_info[1], motor_info[2], pwm_val)
          
    def stop_all(self):
        print(" EMERGENCY STOP: Halting all systems.")
        for motor in self.motors:
            if hasattr(motor, 'stop'):
                motor.stop()
        if hasattr(self, 'camera'):
            self.camera.stop()

    def run(self):
        # --- CONFIGURATION ---
        MAX_LOST_FRAMES = 100000  # If line lost for 10 frames, STOP.
        CENTER_X = 320        # Target center (half of image width 320)
        
        lost_counter = 0

        try:
            print("Robot Loop Started. Press Ctrl+C to stop.")
            while True:
                # 1. GET IMAGE (Instant non-blocking read)
                start = time.time()
                img_gray = self.camera.read_gray()
                print(f"{img_gray.shape=}")
                
                if img_gray is None: 
                    continue # Wait if camera glitches

                self.frame_count += 1

                # 2. IMAGE PROCESSING
                mask, roi = self.preprocess_image(img_gray, th_w=0.35, th_h=0.1)
                print(f"{mask.shape=} {roi.shape=}")

                try:
                    cx, cy, dx, dy = self.middle_vector(mask)
                    
                    # Update State (Found Line)
                    self.cx, self.cy = cx, cy
                    self.dx, self.dy = dx, dy
                    lost_counter = 0
                    debug_overlay = self.draw_debug_info(roi, cx, cy, dx, dy)
                    print(f"{debug_overlay.shape=}")
                    

                except ValueError:
                    # SAFETY WATCHDOG: Handle Lost Line
                    lost_counter += 1
                    if lost_counter > MAX_LOST_FRAMES:
                        print("Line lost for too long! Safety Stop.")
                        self.stop_all()
                        break 
                    
                # --- 4. DEBUG SAVING (every 10th frame) ---
                if self.frame_count % 100 == 0:
                     print(f"Saving debug frames for frame {self.frame_count}...")
                     self.debug_save_images(
                         images={
                             "1_Gray": img_gray,
                             "2_ROI": roi,
                             "3_Mask": mask,
                             "4_Overlay": debug_overlay
                         },
                         save_dir=self.debug_save_dir,
                         frame_id=self.frame_count
                     )
                
                # 3. CONTROL LOOP (Only drive if line is valid)
                if lost_counter <= MAX_LOST_FRAMES:
                    # Calculate Error
                    error = CENTER_X - self.cx
                    
                    # PID Calc
                    correction = self.pid_correction(error)
                    # self.angularVelocity = -correction * 0.01

                    # Drive Motors
                    w = self.get_motorW()
                    print(f"{w=} {self.cx=}")
                    self.apply_wheel_speeds(w)
                
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

if __name__ == "__main__":
    robot = Robot()
    robot.run()