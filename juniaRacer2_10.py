import pygame
import random
import os
import math
import numpy as np
import time
import sys
from datetime import datetime
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from PIL import Image
from operator import attrgetter
import importlib

#driver = importlib.import_module("c3po2_5", package = None)

pygame.init() #Initialize pygame
#Some variables initializations
img = 0 #This one is used when recording frames
size = width,height = 1600, 900 #Size to use when creating pygame window

#Colors
white = (255,255,255)
green = (0, 255, 0) 
blue = (0, 0, 128)  
black = (0,0,0)
gray = pygame.Color('gray12')
Color_line = (255,0,0)

generation = 1
mutationRate = 90
#FPS = 30
#selectedCars = []
selected = 0
lines = True #If true then lines of player are shown
player = True #If true then player is shown
display_info = True #If true then display info is shown
frames = 0
maxspeed = 10 
number_track = 1

white_small_car = pygame.image.load('Images\Sprites\white_small.png')
white_big_car = pygame.image.load('Images\Sprites\white_big.png')
green_small_car = pygame.image.load('Images\Sprites\green_small.png')
green_big_car = pygame.image.load('Images\Sprites\green_big.png')

bg = pygame.image.load('bg73.png')
bg4 = pygame.image.load('bg43.png')


def calculateDistance(x1,y1,x2,y2): #Used to calculate distance between points
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def rotation(origin, point, angle): #Used to rotate points #rotate(origin, point, math.radians(10))
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
    
def move(point, angle, unit): #Translate a point in a given direction
  x = point[0]
  y = point[1]
  rad = math.radians(-angle % 360)

  x += unit*math.sin(rad)
  y += unit*math.cos(rad)

  return x, y
  
def sigmoid(z): #Sigmoid function, used as the neurons activation function
    return 1.0/(1.0+np.exp(-z))

class Cell:
    #A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}
        self.color = 0, 0, 0
        self.track = ""

    def has_all_walls(self):
        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        #Knock down the wall between cells self and other
        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False

class Car:
  def __init__(self, sizes):
    self.score = 0
    self.num_layers = len(sizes) #Number of nn layers
    self.sizes = sizes #List with number of neurons per layer
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #Biases
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] #Weights 
    #c1, c2, c3, c4, c5 are five 2D points where the car could collided, updated in every frame
    self.c1 = 0,0
    self.c2 = 0,0
    self.c3 = 0,0
    self.c4 = 0,0
    self.c5 = 0,0
    #d1, d2, d3, d4, d5 are distances from the car to those points, updated every frame too and used as the input for the NN
    self.d1 = 0
    self.d2 = 0
    self.d3 = 0
    self.d4 = 0
    self.d5 = 0
    self.yaReste = False
    #The input and output of the NN must be in a numpy array format
    self.inp = np.array([[self.d1],[self.d2],[self.d3],[self.d4],[self.d5]])
    self.outp = np.array([[0],[0],[0],[0]])
    #Boolean used for toggling distance lines
    self.showlines = True
    #Initial location of the car
    self.x = 120
    self.y = 480
    self.center = self.x, self.y
    #Height and width of the car
    self.height = 35 #45
    self.width = 17 #25
    #These are the four corners of the car, using polygon instead of rectangle object, when rotating or moving the car, we rotate or move these
    self.d = self.x-(self.width/2),self.y-(self.height/2)
    self.c = self.x + self.width-(self.width/2), self.y-(self.height/2)
    self.b = self.x + self.width-(self.width/2), self.y + self.height-(self.height/2) #El rectangulo está centrado en (x,y)
    self.a = self.x-(self.width/2), self.y + self.height-(self.height/2)              #(a), (b), (c), (d) son los vertices
    #Velocity, acceleration and direction of the car
    self.velocity = 0
    self.acceleration = 0  
    self.angle = 180
    #Boolean which goes true when car collides
    self.collided = False
    #Car color and image
    self.color = white
    self.car_image = white_small_car
  def set_accel(self, accel): 
    self.acceleration = accel
  def rotate(self, rot): 
    self.angle += rot
    if self.angle > 360:
        self.angle = 0
    if self.angle < 0:
        self.angle = 360 + self.angle
  def update(self): #En cada frame actualizo los vertices (traslacion y rotacion) y los puntos de colision
    self.score += self.velocity
    if self.acceleration != 0:
        self.velocity += self.acceleration
        if self.velocity > maxspeed:
            self.velocity = maxspeed
        elif self.velocity < 0:
            self.velocity = 0
    else:
        self.velocity *= 0.92
        
    self.x, self.y = move((self.x, self.y), self.angle, self.velocity)
    self.center = self.x, self.y
    
    self.d = self.x-(self.width/2),self.y-(self.height/2)
    self.c = self.x + self.width-(self.width/2), self.y-(self.height/2)
    self.b = self.x + self.width-(self.width/2), self.y + self.height-(self.height/2) #El rectangulo está centrado en (x,y)
    self.a = self.x-(self.width/2), self.y + self.height-(self.height/2)              #(a), (b), (c), (d) son los vertices
        
    self.a = rotation((self.x,self.y), self.a, math.radians(self.angle)) 
    self.b = rotation((self.x,self.y), self.b, math.radians(self.angle))  
    self.c = rotation((self.x,self.y), self.c, math.radians(self.angle))  
    self.d = rotation((self.x,self.y), self.d, math.radians(self.angle))    
    
    self.c1 = move((self.x,self.y),self.angle,10)
    while bg4.get_at((int(self.c1[0]),int(self.c1[1]))).a!=0:
        self.c1 = move((self.c1[0],self.c1[1]),self.angle,10)
    while bg4.get_at((int(self.c1[0]),int(self.c1[1]))).a==0:
        self.c1 = move((self.c1[0],self.c1[1]),self.angle,-1)

    self.c2 = move((self.x,self.y),self.angle+45,10)
    while bg4.get_at((int(self.c2[0]),int(self.c2[1]))).a!=0:
        self.c2 = move((self.c2[0],self.c2[1]),self.angle+45,10)
    while bg4.get_at((int(self.c2[0]),int(self.c2[1]))).a==0:
        self.c2 = move((self.c2[0],self.c2[1]),self.angle+45,-1)

    self.c3 = move((self.x,self.y),self.angle-45,10)
    while bg4.get_at((int(self.c3[0]),int(self.c3[1]))).a!=0:
        self.c3 = move((self.c3[0],self.c3[1]),self.angle-45,10)
    while bg4.get_at((int(self.c3[0]),int(self.c3[1]))).a==0:
        self.c3 = move((self.c3[0],self.c3[1]),self.angle-45,-1)
        
    self.c4 = move((self.x,self.y),self.angle+90,10)
    while bg4.get_at((int(self.c4[0]),int(self.c4[1]))).a!=0:
        self.c4 = move((self.c4[0],self.c4[1]),self.angle+90,10)
    while bg4.get_at((int(self.c4[0]),int(self.c4[1]))).a==0:
        self.c4 = move((self.c4[0],self.c4[1]),self.angle+90,-1)
        
    self.c5 = move((self.x,self.y),self.angle-90,10)
    while bg4.get_at((int(self.c5[0]),int(self.c5[1]))).a!=0:
        self.c5 = move((self.c5[0],self.c5[1]),self.angle-90,10)
    while bg4.get_at((int(self.c5[0]),int(self.c5[1]))).a==0:
        self.c5 = move((self.c5[0],self.c5[1]),self.angle-90,-1)
        
    self.d1 = int(calculateDistance(self.center[0], self.center[1], self.c1[0], self.c1[1]))
    self.d2 = int(calculateDistance(self.center[0], self.center[1], self.c2[0], self.c2[1]))
    self.d3 = int(calculateDistance(self.center[0], self.center[1], self.c3[0], self.c3[1]))
    self.d4 = int(calculateDistance(self.center[0], self.center[1], self.c4[0], self.c4[1]))
    self.d5 = int(calculateDistance(self.center[0], self.center[1], self.c5[0], self.c5[1]))
    
    
  def draw(self,display):
    rotated_image = pygame.transform.rotate(self.car_image, -self.angle-180)
    rect_rotated_image = rotated_image.get_rect()
    rect_rotated_image.center = self.x, self.y
    gameDisplay.blit(rotated_image, rect_rotated_image)
  
    center = self.x, self.y
    if self.showlines: 
        pygame.draw.line(gameDisplay,Color_line,(self.x,self.y),self.c1,2)
        pygame.draw.line(gameDisplay,Color_line,(self.x,self.y),self.c2,2)
        pygame.draw.line(gameDisplay,Color_line,(self.x,self.y),self.c3,2)
        pygame.draw.line(gameDisplay,Color_line,(self.x,self.y),self.c4,2)
        pygame.draw.line(gameDisplay,Color_line,(self.x,self.y),self.c5,2) 
    
  def showLines(self):
    self.showlines = not self.showlines
    
  def collision(self):
      if (bg4.get_at((int(self.a[0]),int(self.a[1]))).a==0) or (bg4.get_at((int(self.b[0]),int(self.b[1]))).a==0) or (bg4.get_at((int(self.c[0]),int(self.c[1]))).a==0) or (bg4.get_at((int(self.d[0]),int(self.d[1]))).a==0):
        return True
      else:
        return False    

  def resetPosition(self):
      self.x = 120
      self.y = 480
      self.angle = 180
      return
      
  def takeAction(self): 
    if self.outp.item(0) > 0.5: #Accelerate
        self.set_accel(0.2)
    else:
        self.set_accel(0)      
    if self.outp.item(1) > 0.5: #Brake
        self.set_accel(-0.2)     
    if self.outp.item(2) > 0.5: #Turn right
        self.rotate(-5)    
    if self.outp.item(3) > 0.5: #Turn left
        self.rotate(5) 
    return
  

#These is just the text being displayed on pygame window
infoX = 1365
infoY = 600 
font = pygame.font.Font('freesansbold.ttf', 18) 
text1 = font.render('0..9 - Change Mutation', True, white) 
text2 = font.render('LMB - Select/Unselect', True, white)
text3 = font.render('RMB - Delete', True, white)
text4 = font.render('L - Show/Hide Lines', True, white)
text5 = font.render('R - Reset', True, white)
text6 = font.render('B - Breed', True, white)
text7 = font.render('C - Clean', True, white)
text8 = font.render('N - Next Track', True, white)
text9 = font.render('A - Toggle Player', True, white)
text10 = font.render('D - Toggle Info', True, white)
text11 = font.render('M - Breed and Next Track', True, white)
text1Rect = text1.get_rect().move(infoX,infoY)
text2Rect = text2.get_rect().move(infoX,infoY+text1Rect.height)
text3Rect = text3.get_rect().move(infoX,infoY+2*text1Rect.height)
text4Rect = text4.get_rect().move(infoX,infoY+3*text1Rect.height)
text5Rect = text5.get_rect().move(infoX,infoY+4*text1Rect.height)
text6Rect = text6.get_rect().move(infoX,infoY+5*text1Rect.height)
text7Rect = text7.get_rect().move(infoX,infoY+6*text1Rect.height)
text8Rect = text8.get_rect().move(infoX,infoY+7*text1Rect.height)
text9Rect = text9.get_rect().move(infoX,infoY+8*text1Rect.height)
text10Rect = text10.get_rect().move(infoX,infoY+9*text1Rect.height)
text11Rect = text11.get_rect().move(infoX,infoY+10*text1Rect.height)

def displayTexts():  
    infotextX = 20
    infotextY = 600
    infotext1 = font.render('Gen ' + str(generation), True, white) 
    #infotext2 = font.render('Cars: ' + str(num_of_nnCars), True, white)
    infotext3 = font.render('Alive: ' + str(alive), True, white)
    infotext4 = font.render('Selected: ' + str(selected), True, white)
    if lines == True:
        infotext5 = font.render('Lines ON', True, white)
    else:
        infotext5 = font.render('Lines OFF', True, white)
    if player == True:
        infotext6 = font.render('Player ON', True, white)
    else:
        infotext6 = font.render('Player OFF', True, white)
    #infotext7 = font.render('Mutation: '+ str(2*mutationRate), True, white)
    #infotext8 = font.render('Frames: ' + str(frames), True, white)
    infotext9 = font.render('FPS: 30', True, white)
    infotext1Rect = infotext1.get_rect().move(infotextX,infotextY)
    infotext2Rect = infotext2.get_rect().move(infotextX,infotextY+infotext1Rect.height)
    infotext3Rect = infotext3.get_rect().move(infotextX,infotextY+2*infotext1Rect.height)
    infotext4Rect = infotext4.get_rect().move(infotextX,infotextY+3*infotext1Rect.height)
    infotext5Rect = infotext5.get_rect().move(infotextX,infotextY+4*infotext1Rect.height)
    infotext6Rect = infotext6.get_rect().move(infotextX,infotextY+5*infotext1Rect.height)
    #infotext7Rect = infotext7.get_rect().move(infotextX,infotextY+6*infotext1Rect.height)
    #infotext8Rect = infotext8.get_rect().move(infotextX,infotextY+7*infotext1Rect.height)
    infotext9Rect = infotext9.get_rect().move(infotextX,infotextY+6*infotext1Rect.height)

    gameDisplay.blit(text1, text1Rect)  
    gameDisplay.blit(text2, text2Rect)  
    gameDisplay.blit(text3, text3Rect) 
    gameDisplay.blit(text4, text4Rect) 
    gameDisplay.blit(text5, text5Rect) 
    gameDisplay.blit(text6, text6Rect)
    gameDisplay.blit(text7, text7Rect)   
    gameDisplay.blit(text8, text8Rect)  
    gameDisplay.blit(text9, text9Rect)     
    gameDisplay.blit(text10, text10Rect) 
    gameDisplay.blit(text11, text11Rect)  
    
    gameDisplay.blit(infotext1, infotext1Rect)  
    gameDisplay.blit(infotext2, infotext2Rect)  
    gameDisplay.blit(infotext3, infotext3Rect) 
    gameDisplay.blit(infotext4, infotext4Rect) 
    gameDisplay.blit(infotext5, infotext5Rect) 
    gameDisplay.blit(infotext6, infotext6Rect)
    gameDisplay.blit(infotext9, infotext9Rect) 
    return
 

gameDisplay = pygame.display.set_mode(size) #creates screen
clock = pygame.time.Clock()

inputLayer = 6
hiddenLayer = 6
outputLayer = 4
car = Car([inputLayer, hiddenLayer, outputLayer])
auxcar = Car([inputLayer, hiddenLayer, outputLayer])

##♥########
##
##
# os.chdir("C:/Users/hakdu/OneDrive/Documents/ISEN - Copie/M1/IA/JUNIA_RACER/Junia_Racer2_6")


FPS = 100
interval= 10
recommencer = 0
epsilon = 0.08
commande_manuelle = 0

vitesse_min=0.5
vitesse_max=3

NB_CASES = 1419857 + 10

Qtable1 = "C:/Users/hakdu/OneDrive/Documents/ISEN - Copie/M1/IA/JUNIA_RACER/JuniaRacer2_10/Qtable1.npy"
CSV="C:/Users/hakdu/OneDrive/Documents/ISEN - Copie/M1/IA/JUNIA_RACER/JuniaRacer2_10/CSV.csv"


if recommencer==0:
    Qtable = np.load(Qtable1)
    
    
    print("Continue Train")
else:
    Qtable = np.zeros((NB_CASES,4), dtype=float)
    print("New Train")
    
    
for i in range (len(Qtable)):
    Qtable[i][3]=-10000000000000


states = np.array([],dtype=int)
choices = np.array([],dtype=int)
frame=0

D1 = np.array([],dtype=int)
D2 = np.array([],dtype=int)
D3 = np.array([],dtype=int)
D4 = np.array([],dtype=int)
D5 = np.array([],dtype=int)

#np.save(Qtable1, Qtable)
#Qtable = np.load(Qtable1)
##
##

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





def base4todecimal(n1,n2,n3,n4,n5):
    return n1*4**4 + n2*4**3 + n3*4**2 + n4*4**1 +n5*4**0
    
def base7todecimal(n1,n2,n3,n4,n5):
    return n1*7**4 + n2*7**3 + n3*7**2 + n4*7**1 +n5*7**0

def base12todecimal(n1,n2,n3,n4,n5):
    return n1*12**4 + n2*12**3 + n3*12**2 + n4*12**1 +n5*12**0

def base17todecimal(n1,n2,n3,n4,n5):
    return n1*17**4 + n2*17**3 + n3*17**2 + n4*17**1 +n5*17**0
    
    
def playGame(d1,d2,d3,d4,d5):
    global interval
    return ( base17todecimal(  reduc_cour_laser(d1,interval) ,reduc_cour_laser(d2,interval) ,reduc_cour_laser(d3,interval) ,reduc_cour_laser(d4,interval) ,reduc_cour_laser(d5,interval) ) )



def bestChoice(d1,d2,d3,d4,d5):
    index=playGame(d1,d2,d3,d4,d5)
    return np.argmax(Qtable[index])

def inputChoice():
    # You can override agent action by keyboard
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        return 2
        #driver_action = driver.LEFT5
    elif keys[pygame.K_RIGHT]:
        return 3
        #driver_action = driver.RIGHT5
    elif keys[pygame.K_UP]:
        return 0
        #driver_action = driver.ACCELERATE
    elif keys[pygame.K_DOWN]:
        return 1
        #driver_action = driver.BRAKE
    

def parallele(d1,d2,d3,d4,d5,choice):
    reward=0.5
    if d3!=0 and d4!=0:
        if choice==0:
            if (d5/ d3) > math.cos(48*math.pi/180)  and (d5 / d3) < math.cos(42*math.pi/180) :
                reward += 0.5
            if (d4/ d2) > math.cos(48*math.pi/180)  and (d4 / d2) < math.cos(42*math.pi/180) :
                reward += 0.5
        if choice==1:
            if (d5/ d3) > math.cos(48*math.pi/180)  and (d5 / d3) < math.cos(42*math.pi/180) :
                reward += 0.3
            if (d4/ d2) > math.cos(48*math.pi/180)  and (d4 / d2) < math.cos(42*math.pi/180) :
                reward += 0.3
        if choice==2 or choice==3:
            if (d5/ d3) > math.cos(48*math.pi/180)  and (d5 / d3) < math.cos(42*math.pi/180) :
                reward += 0.2
            if (d4/ d2) > math.cos(48*math.pi/180)  and (d4 / d2) < math.cos(42*math.pi/180) :
                reward += 0.2
    return reward

def proximty_to_wall(d1,d2,d3,d4,d5,choice):
    long=60
    court=40
    reward=1
    if d1<long:
        reward+= -0.3
    if d2<court:
        reward+= -0.1
    if d3<court:
        reward+= -0.1
    if d4<court:
        reward+= -0.1
    if d5<court:
        reward+= -0.1
    return reward
    
    
def milieu(d4,d5):
    if d4<=d5:
        return d4-30
    else:
        return d5-30
    

def redrawGameWindow2(epsilon,d1,d2,d3,d4,d5): #Called on very frame   

    
    global alive  
    global frames
    global img
    frames += 1

    global states
    global choices
    global reward
    
    global D1
    global D2
    global D3
    global D4
    global D5
    
    global frame
    
    dead=0
    
    index=playGame(d1,d2,d3,d4,d5)
    
    if commande_manuelle==0:
        
        if car.velocity<vitesse_max and car.velocity>vitesse_min:
            if random.uniform(0, 1) < epsilon :
                choice = random.randint(0, 2)
            else:
                choice =  bestChoice(d1,d2,d3,d4,d5)
        elif car.velocity<vitesse_min:
            choice=0
        else:
            choice=3
            
    else:
        choice = inputChoice()
    
    
    states = np.append (states, index)  # States start at 0
    choices = np.append (choices, choice)
    
    #print(choice,car.velocity)

    D1=np.append (D1, d1)
    D2=np.append (D2, d2)
    D3=np.append (D3, d3)
    D4=np.append (D4, d4)##
    D5=np.append (D5, d5)##
    
    gameD = gameDisplay.blit(bg, (0,0))  

    if player:
        car.update()
        if car.collision():
            dead=1
            #reward += 0
            car.resetPosition()
            car.update()
        car.draw(gameDisplay) 

    pygame.display.update() #updates the screen

    
    # if d4<=d5:
    #     print(d4-30)
    # else:
    #     print(d5-30)
    

    if (dead==1):
        dead=0
        numb_reward_double_neg=0
        numb_reward_une_neg=15
        numb_reward_zero_neg=15
        #print('=',reward)
        for i in (reversed(range(len(states)))):        
            state = states[i]
            choice = choices[i]

            
            if numb_reward_double_neg>=0:
                numb_reward_double_neg -=1
                Qtable[state, choice] += -30
                
                
            elif numb_reward_une_neg>=0:
                numb_reward_une_neg -=1
                Qtable[state, choice] += -20   
                
            elif numb_reward_zero_neg>=0:
                numb_reward_zero_neg -=1
                Qtable[state, choice] += 0 
                #print(-5)
            
            
           
            
            else:
                Qtable[state, choice] +=  20 #milieu(D4[i], D5[i])
                #print(milieu(D4[i], D5[i]))
            
 
        
        D1= np.array([],dtype=int)
        D2= np.array([],dtype=int)
        D3= np.array([],dtype=int)
        D4= np.array([],dtype=int)
        D5= np.array([],dtype=int)
        
        
        states = np.array([],dtype=int)
        choices = np.array([],dtype=int)
        #print("____________")
        np.set_printoptions(threshold=np.inf)
    
    if(choice==3):
        car.set_accel(-0.2)
        #♦print("BRAKE")
        #return BRAKE
    if(choice==0):
        car.set_accel(0.2)
        #print("ACCELERATE")
        #return ACCELERATE
    if(choice==1):
        car.rotate(-5)
        #print("LEFT5") 
        #return LEFT5
    if(choice==2):
        car.rotate(5)
        #print("RIGHT5")
        #return RIGHT5

    #print(choices)



while True:
    #now1 = time.time()  
  
    for event in pygame.event.get(): #Check for events
        
        if event.type == pygame.KEYDOWN: #If user uses the keyboard
            if event.key == pygame.K_SPACE:
                np.save(Qtable1, Qtable)
                np.savetxt(CSV, Qtable, delimiter=",")
                print("SAVED")
                pygame.quit() #quits
                #quit()
            



    d1=car.d1
    d2=car.d2
    d3=car.d3
    d4=car.d4
    d5=car.d5
    
    #print(d1,d2,d3,d4,d5)
    
# d1  front
# 	# d2  mid left
# 	# d3  mid right
# 	# d4  left
# 	# d5  right
        
    redrawGameWindow2(epsilon,d1,d2,d3,d4,d5)
    
    
    
    clock.tick(FPS)



    