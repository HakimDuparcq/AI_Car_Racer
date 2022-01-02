from random import *
import numpy as np
import math
import random
# Init your variables here

# Put your name Here
name = "hakim"

BRAKE = 0
ACCELERATE = 1
LEFT5 = 2
RIGHT5 = 3
NOTHING = 4

FPS = 600000000
interval= 10
recommencer = 0
epsilon =  0.00001


NB_CASES = 1419857 + 10



vitesse_min=0.5
vitesse_max=3

NB_CASES = 1419857 + 10

Qtable1 = "Qtable1.npy"


Qtable = np.load(Qtable1)


def setup():
	aaaaaaa=2

def reduc_cour_laser(number,interval):
	if number <=interval:
		return 0
	if number>interval and number<=interval*2:
		return 1
	if number>interval*2 and number<=interval*3:
		return 2
	if  number>interval*3 and number<=interval*4:
		return 3
	if  number>interval*4 and number<=interval*5:
		return 4
	if  number>interval*5 and number<=interval*6:
		return 5
	if  number>interval*6 and number<=interval*7:
		return 6
	if  number>interval*7 and number<=interval*8:
		return 7
	if  number>interval*8 and number<=interval*9:
		return 8
	if  number>interval*9 and number<=interval*10:
		return 9
	if  number>interval*10 and number<=interval*11:
		return 10
	if  number>interval*11 and number<=interval*12:
		return 11
	if  number>interval*12 and number<=interval*13:
		return 12
	if  number>interval*13 and number<=interval*14:
		return 13
	if  number>interval*14 and number<=interval*15:
		return 14
	if  number>interval*15 and number<=interval*16:
		return 15
	return 16


def base17todecimal(n1,n2,n3,n4,n5):
	return n1*17**4 + n2*17**3 + n3*17**2 + n4*17**1 +n5*17**0


def playGame(d1,d2,d3,d4,d5):
	global interval
	return ( base17todecimal(  reduc_cour_laser(d1,interval) ,reduc_cour_laser(d2,interval) ,reduc_cour_laser(d3,interval) ,reduc_cour_laser(d4,interval) ,reduc_cour_laser(d5,interval) ) )




def bestChoice(d1,d2,d3,d4,d5):
	global Qtable
	index=playGame(d1,d2,d3,d4,d5)
	return np.argmax(Qtable[index])




def drive(d1, d2, d3, d4, d5,car_velocity,acceraration):
		vitesse_min=0.5
		vitesse_max=5

		if car_velocity<vitesse_max and car_velocity>vitesse_min:
				if random.uniform(0, 1) < epsilon :
						choice = random.randint(0, 2)
				else:
						choice =  bestChoice(d1,d2,d3,d4,d5)
		elif car_velocity<vitesse_min:
				choice=0
		else:
				choice=3




		if choice==0:
				return ACCELERATE

		if choice==1:
			return LEFT5

		if choice==2:
			return RIGHT5

		if choice==3:
			return BRAKE




