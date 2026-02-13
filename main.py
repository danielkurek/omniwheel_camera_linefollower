from robot import Robot, LineDetector
from hardware import MotorsConfig, OmniwheelController, PicameraStream
from PCA9685_smbus2 import PCA9685

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