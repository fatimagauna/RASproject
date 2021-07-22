import RPi.GPIO as GPIO
import time

dirA1 = 5
dirA2 = 6
spdA = 13
dirB1 = 26
dirB2 = 16
spdB = 12

#GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(dirA1,GPIO.OUT)
GPIO.setup(dirA2,GPIO.OUT)
GPIO.setup(spdA,GPIO.OUT)
pwmA=GPIO.PWM(spdA,1000)
pwmA.start(10)
GPIO.setup(dirB1,GPIO.OUT)
GPIO.setup(dirB2,GPIO.OUT)
GPIO.setup(spdB,GPIO.OUT)
pwmB=GPIO.PWM(spdB,1000)
pwmB.start(10)


try:
    while(1):
        GPIO.output(dirA1,True)
        GPIO.output(dirA2,False)
        pwmA.ChangeDutyCycle(75)
        GPIO.output(dirB1,True)
        GPIO.output(dirB2,False)
        pwmB.ChangeDutyCycle(75)
        time.sleep(5)
        GPIO.output(dirA1,False)
        GPIO.output(dirA2,True)
        pwmA.ChangeDutyCycle(75)
        GPIO.output(dirB1,False)
        GPIO.output(dirB2,True)
        pwmB.ChangeDutyCycle(75)
        time.sleep(5)
        
except KeyboardInterrupt:
    #pwmA.stop()
    #pwmB.stop()
    GPIO.cleanup() 

#Always at the end of the code
GPIO.cleanup()
