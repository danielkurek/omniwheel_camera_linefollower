from picamera2 import Picamera2, Preview
import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
import os
import shutil


class Robot():

    def __init__(self):
        
        self.ena, self.in1, self.in2 = 18, 23, 24  # Motor 1
        self.enb, self.in3, self.in4 = 19, 5, 6    # Motor 2
        self.enc, self.in5, self.in6 = 13, 16, 26  # Motor 3
        
        ## set pins
        GPIO.setmode(GPIO.BCM)
        pins = [self.ena, self.in1, self.in2, 
                self.enb, self.in3, self.in4, 
                self.enc, self.in5, self.in6]
        GPIO.setup(pins, GPIO.OUT)
        
        self.pwm_a = GPIO.PWM(self.ena, 42000)
        self.pwm_b = GPIO.PWM(self.enb, 42000)
        self.pwm_c = GPIO.PWM(self.enc, 42000)
        
        self.pwm_a.start(0)
        self.pwm_b.start(0)
        self.pwm_c.start(0)
        
        # okay
        self.max_physical_speed = 1.0 
        
        # dont judge, just enjoy :D
        self.motor1 = [self.pwm_a, self.in1, self.in2]
        self.motor2 = [self.pwm_b, self.in3, self.in4]
        self.motor3 = [self.pwm_c, self.in5, self.in6]     
        self.motors = [self.motor1, self.motor2, self.motor3]

        # -- Debugging --
        self.frame_count = 0
        self.debug_save_dir = "debug_frames" # Folder to save images in
        self._clean_debug_dir()
        
        # -- Constants --
        self.angles = np.deg2rad([90, 210, 330])    # angles of the weels
        self.linearSpeed = 0.8                      # m/s linear speed along (dx,dy)
        self.angularVelocity = 0.0                  # rad/s (spin). Set >0 to rotate CCW
        self.wheelRadius = 1.0                      # wheel radius in meters (or set to your real radius)
        self.radius = 1                             # radius of robot

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
        self.camera = self.configure_camera()
    
    def configure_camera(self):
        picam2 = Picamera2()  
        picam2.start_preview(Preview.NULL)  
        capture_config = picam2.create_still_configuration(main={"size": (320, 240)})  
        picam2.configure(capture_config)
        picam2.start()

        time.sleep(1)
        with picam2.controls as ctrl:
            ctrl.AnalogueGain = 1.0
            ctrl.ExposureTime = 50000
        time.sleep(1)

        return picam2
    
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
        """
        Draws the centroid and direction vector onto an image for debugging.
        Returns a new BGR image with info drawn on it.
        """
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
        """
        Saves multiple images to a specified directory.
        
        Args:
            images: A dictionary of {"file_suffix": image_array}.
            save_dir: The directory to save images to (e.g., "debug_frames").
            frame_id: The current frame number (e.g., 1, 2, 3...).
        """
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
    
    def capture_frame_gray(self):
        im = self.camera.capture_array()
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        self.camera.capture_file("demo.jpg")
        return grey
    
    def _clean_debug_dir(self):
        """Deletes the debug directory and recreates it empty."""
        if os.path.exists(self.debug_save_dir):
            shutil.rmtree(self.debug_save_dir)  # Deletes dir and all contents
        os.makedirs(self.debug_save_dir, exist_ok=True) # Creates fresh dir
        print(f"[INFO] Cleared and renewed debug directory: {self.debug_save_dir}")
    
    def set_single_motor(self, pwm_obj, in_x, in_y, speed):
        """Helper to set one motor's speed (-100 to 100)"""
        speed = max(min(speed, 100), -100) # Clamp
        
        if speed >= 0:
            GPIO.output(in_x, GPIO.HIGH)
            GPIO.output(in_y, GPIO.LOW)
            pwm_obj.ChangeDutyCycle(speed)
        else:
            GPIO.output(in_x, GPIO.LOW)
            GPIO.output(in_y, GPIO.HIGH)
            pwm_obj.ChangeDutyCycle(abs(speed))
    
    # digitaloutout device inpins x2 , on or of freq 42k | set duty cycle, value 0.5 + pid
    def apply_wheel_speeds(self, w):
        """
        Takes the calculated wheel speeds (w), normalizes them, 
        and sends PWM signals to the motors.
        """
        print(f"Target Speeds: {w}")

        for motor_info, speed in zip(self.motors, w):
            pwm_val = (speed / self.max_physical_speed) * 100
            
            # motor_info = [pwm_object, pin_a, pin_b]
            self.set_single_motor(motor_info[0], motor_info[1], motor_info[2], pwm_val)
        
        
    def stop_all(self):
        for motor in self.motors:
            motor.stop()
        self.camera.stop()


    def run(self):
        while True:
            # Increment frame counter
            self.frame_count += 1
            
            # --- 1. Get Image ---
            img_gray = self.capture_frame_gray()
            mask, roi = self.preprocess_image(img_gray, th_w=0.35, th_h=0.1)
            
            # --- 3. Handle line detection ---
            debug_overlay = None 
            try:
                cx, cy, dx, dy = self.middle_vector(mask)
                self.cx = cx
                self.cy = cy
                self.dx = dx
                self.dy = dy
                
                # We only need to draw if we found a line
                debug_overlay = self.draw_debug_info(roi, cx, cy, dx, dy)

            except ValueError:
                print("Line lost, continuing with last known direction...")
                pass # debug_overlay remains None

            
            # --- 4. DEBUG SAVING (every 10th frame) ---
            if self.frame_count % 10 == 0:
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
                
            
            # how much have we moved from the center
            error = 160 - self.cx
            correction = self.pid_correction(error)
            self.angularVelocity = -correction * 0.01

            # with the corrected anculgar velocity, get wheel speeds
            w = self.get_motorW()
            
            
            self.apply_wheel_speeds(w)

if __name__ == "__main__":  
    robot = Robot()
    robot.run()
