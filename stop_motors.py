from PCA9685_smbus2 import PCA9685
from gpiozero import DigitalOutputDevice

in1a, in1b = DigitalOutputDevice(pin=22), DigitalOutputDevice(pin=23)  # Motor 1
in2a, in2b = DigitalOutputDevice(pin=24), DigitalOutputDevice(pin=25)  # Motor 2
in3a, in3b = DigitalOutputDevice(pin=26), DigitalOutputDevice(pin=27)  # Motor 3

pwm = PCA9685.PCA9685(interface=1)
pwm.set_pwm_freq(1600)
pwm.set_all_pwm(0,0)

for pin in [in1a, in1b, in2a, in2b, in3a, in3b]:
    pin.off()