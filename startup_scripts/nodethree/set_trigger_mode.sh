import Jetson.GPIO as GPIO
import time

PIN = 13
FREQUENCY = 60  # Hz
DUTY_CYCLE = 2 # percent

GPIO.setmode(GPIO.BOARD)
GPIO.setup(PIN, GPIO.OUT)

pwm = GPIO.PWM(PIN, FREQUENCY)
pwm.start(DUTY_CYCLE)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

finally:
    pwm.stop()
    GPIO.cleanup()
    print("PWM stoppet, GPIO ryddet opp.")