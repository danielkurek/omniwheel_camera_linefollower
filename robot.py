import time
import numpy as np
import cv2
import os
import shutil
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

class DriveInterface:
    def __init__(self):
        self.running = False
    def start(self):
        raise NotImplementedError()
    def stop(self):
        raise NotImplementedError()
    def set_vector(self, driving_vector):
        raise NotImplementedError()

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
    def __init__(self, drive_controller: DriveInterface, line_detector: LineDetector, verbose=False):
        self.verbose = verbose
        self.drive_controller = drive_controller
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
        self.drive_controller.stop()
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
                self.drive_controller.set_vector(v)
                
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
