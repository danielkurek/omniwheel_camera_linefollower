from PCA9685_smbus2 import PCA9685
from gpiozero import DigitalOutputDevice

in1a, in1b = DigitalOutputDevice(pin=22), DigitalOutputDevice(pin=23)  # Motor 1
in2a, in2b = DigitalOutputDevice(pin=24), DigitalOutputDevice(pin=25)  # Motor 2
in3a, in3b = DigitalOutputDevice(pin=26), DigitalOutputDevice(pin=27)  # Motor 3

pwm = PCA9685.PCA9685(interface=1)
pwm.set_pwm_freq(1600)
pwm.set_all_pwm(0,2048)

def forward_check(in_a, in_b, id):
    result = None
    while True:
        in_a.on()
        if input(f"Is the motor {id} running in forward direction?") in ["n","no"]:
            in_a.off()
            in_b.on()
            if input(f"Is the motor {id} running in forward direction?") in ["y","yes"]:
                result = "b"
                break
        else:
            result = "a"
            break
    in_a.off()
    in_b.off()
    return result

def pwm_channel_check(in_a, pwm, id):
    in_a.on()
    result = None
    while True:
        for i in range(10):
            pwm.set_pwm(i, 0, 4000)
            if input(f"Is the motor {id} spinning faster?") in ["y", "yes"]:
                result = i
                break
            pwm.set_pwm(i, 0, 2048)
        if result is not None:
            break
    in_a.off()
    pwm.set_all_pwm(0,2048)
    return result


forward_directions = [None for _ in range(3)]
forward_directions[0] = forward_check(in1a, in1b, 1)
forward_directions[1] = forward_check(in2a, in2b, 2)
forward_directions[2] = forward_check(in3a, in3b, 3)

pwm_channel = [None for _ in range(3)]
pwm_channel[0] = pwm_channel_check(in1a, pwm, 1)
pwm_channel[1] = pwm_channel_check(in2a, pwm, 2)
pwm_channel[2] = pwm_channel_check(in3a, pwm, 3)


print(f"{forward_directions=}")
print(f"{pwm_channel=}")