import RPi.GPIO as GPIO
import time

dirA1 = 5
dirA2 = 6
spdA = 13

#GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(dirA1,GPIO.OUT)
GPIO.setup(dirA2,GPIO.OUT)
GPIO.setup(spdA,GPIO.OUT)
pwmA=GPIO.PWM(spdA,1000)
pwmA.start(10)

while(1):
    GPIO.output(dirA1,True)
    GPIO.output(dirA2,False)
    pwmA.ChangeDutyCycle(25)
    time.sleep(5)
    GPIO.output(dirA1,False)
    GPIO.output(dirA2,True)
    pwmA.ChangeDutyCycle(10)
    time.sleep(5)

#Always at the end of the code
GPIO.cleanup()
