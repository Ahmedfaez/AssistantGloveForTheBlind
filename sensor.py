import RPi.GPIO as GPIO
import time
import signal
import sys

GPIO.setmode(GPIO.BCM)

trigger = 18
echo = 24
buzzer = 17

def close(signal, frame):
	GPIO.cleanup()
	sys.exit(0)

signal.signal(signal.SIGINT, close)

GPIO.setup(trigger, GPIO.OUT)
GPIO.setup(echo, GPIO.IN)
GPIO.setup(buzzer, GPIO.OUT)

while True:
	GPIO.output(trigger, True)
	time.sleep(0.1)
	GPIO.output(trigger, False)

	startTime = time.time()
	stopTime = time.time()

	while 0 == GPIO.input(echo):
		startTime = time.time()

	while 1 == GPIO.input(echo):
		stopTime = time.time()
		TimeElapsed = stopTime - startTime
		distance = (TimeElapsed * 34300) / 2
		print ("Distance: %.1f cm" % distance)

	while distance <= 30:
		GPIO.output(buzzer, True)
		time.sleep(0.1)
		GPIO.output(buzzer, False)
		print ("WALL")
		break
	time.sleep(1)
