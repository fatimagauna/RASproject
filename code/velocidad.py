#Código de control y gráficas
import numpy as np
import time
from gpiozero import Robot  
def Velocidad():
    robot=Robot((17,18),(23,22))                                    #Definir que un motor del robot se conecta a GPIO 17 y 18 y el otro motor a GPIO 22 y 23
    #Left Lane
    xminl=69
    xmaxl=256
    y_bottoml=259
    y_topl=197
    #Right Lane
    xminr=429
    xmaxr=715
    y_bottomr=193
    y_topr=309
    #Pendiente
    ml=(-y_topl+y_bottoml)/(xmaxl-xminl)
    mr=(-y_topr+y_bottomr)/(xmaxr-xminr)
    #B's
    bl=y_topl-ml*xminl
    br=y_bottomr-mr*xmaxr
    #Pendientes impresas
    print("\nLas fórmulas de las rectas izq y der son: ")
    print("y=",ml,"x +",bl)
    print("y=",mr,"x +",br,"\n")
    #Calculo de xr y xl en ya
    ya=250 
    xl=(ya-bl)/ml
    xr=(ya-br)/mr
    #Xroi
    xroi=(xr-xl)/2+xl
    print("xroi es: ",xroi)
    A = np.matrix([[-ml, 1],[-mr, 1]])
    b = np.matrix([[bl],[br]])
    [yint,xint] = (A**-1)*b
    print("Las coordenadas son: ",xint)


    constRap=0.5
    if (xroi < xint):
        correcion = (abs(xroi-xint)*constRap)/xroi
        rapIzq=constRap-correcion
        rapDer=constRap+correcion
        #robot.forward(speed=abs(constRap),curve_right=y)
        #robot.source = zip(rapIzq , rapDer)
        robot.value = (rapDer, rapIzq)
        time.sleep(5)
        print(rapIzq, rapDer)
    elif (xroi > xint):
        correcion = ((xroi-xint)*constRap)/xroi
        rapIzq=constRap+correcion
        rapDer=constRap-correcion
        #robot.forward(speed=abs(constRap),curve_left=abs(rapIzq))
        #time.sleep(5)
        #robot.source = zip(rapIzq, rapDer)
        robot.value = (rapDer, rapIzq)
        time.sleep(5)
        print(rapDer, rapIzq)
    else:
        rapIzq=constRap
        rapDer=constRap
        robot.forward(speed=abs(constRap)) 
        
    return [rapIzq, rapDer]
  
Velocidad()
