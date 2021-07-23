import cv2
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
GPIO.setup(dirB1,GPIO.OUT)
GPIO.setup(dirB2,GPIO.OUT)
GPIO.setup(spdB,GPIO.OUT)
pwmA=GPIO.PWM(spdA,1000)
pwmB=GPIO.PWM(spdB,1000)
pwmB.start(10)
pwmA.start(10)

cap = cv2.VideoCapture(0)

x = 1

try:
	while(1):
		ret, frame = cap.read()
		cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
		cv2.imshow('Video', frame)
		if ((cv2.waitKey(33) == -1) and (x == 1)):
			print ("Ninguna")
			x = 0
			GPIO.output(dirA1,False)
			GPIO.output(dirA2,False)
			GPIO.output(dirB1,False)
			GPIO.output(dirB2,False)
			pwmA.ChangeDutyCycle(100)
			pwmB.ChangeDutyCycle(100)

		if (cv2.waitKey(33) == ord('w')):
			print ("Upkey")
			x = 1
			GPIO.output(dirA1,True)
			GPIO.output(dirA2,False)
			GPIO.output(dirB1,True)
			GPIO.output(dirB2,False)
			pwmA.ChangeDutyCycle(100)
			pwmB.ChangeDutyCycle(100)

		if (cv2.waitKey(33) == ord('a')):
			print ("Leftkey")
			x = 1
			GPIO.output(dirA1,True)
			GPIO.output(dirA2,False)
			GPIO.output(dirB1,True)
			GPIO.output(dirB2,False)
			pwmA.ChangeDutyCycle(100)
			pwmB.ChangeDutyCycle(75)

		if (cv2.waitKey(33) == ord('s')):
			print ("Dowkey")
			x = 1
			GPIO.output(dirA1,False)
			GPIO.output(dirA2,True)
			GPIO.output(dirB1,False)
			GPIO.output(dirB2,True)
			pwmA.ChangeDutyCycle(100)
			pwmB.ChangeDutyCycle(100)

		if (cv2.waitKey(33) == ord('d')):
			print ("Rightkey")
			x = 1
			GPIO.output(dirA1,True)
			GPIO.output(dirA2,False)
			GPIO.output(dirB1,True)
			GPIO.output(dirB2,False)
			pwmA.ChangeDutyCycle(75)
			pwmB.ChangeDutyCycle(100)

		if (cv2.waitKey(33) == ord('q')):
			cap.release()
			cv2.destroyAllWindows()
			break
		
except KeyboardInterrupt:
    GPIO.cleanup() 


#Always at the end of the code
GPIO.cleanup()
