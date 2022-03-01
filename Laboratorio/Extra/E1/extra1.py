"""
EXTRA 1: TRIÁNGULO DE SIERPINSKI
Martín Fernández de Diego
"""

import matplotlib.pyplot as plt
import numpy as np
import random as rand
from PIL import Image

"""
Dados una x, un cierto delta y un intervalo [a,b]
ajusta la eps para que x+eps y x-eps quepan en dicho intervalo
"""
# Restando epsilon evitamos que epsilon sature en los extremos indeseablemente
def ajustar(x,delta,a,b,epsilon=0.001):
    if x + delta > b:
        return b - x - epsilon
    if x - delta < a:
        return x - a - epsilon
    return delta

def sierpinski(a,dim):
    if dim != 1:
        aux = dim//3
        #rellena el interior de 0s
        a[aux:2*aux,aux:2*aux] = 0
        for i in range(0,dim,aux):
            for j in range(0,dim,aux):
                #llamada recursiva menos a la del centro
                if i != aux or j != aux:
                    sierpinski(a[i:i+aux,j:j+aux],aux)

#Recibe una matriz de 0s y 1s y la pinta
def pintar(a,dim):
    dibujo=np.empty((dim,dim,3), dtype=np.uint8)
    for i in range(dim):
        for j in range(dim):
            if a[i,j]==1:
                #negro
                dibujo[i,j]=[0,0,0]
            else:
                #amarillo
                dibujo[i,j]=[255,255,255]
    Image.fromarray(dibujo).save("Sierpinski.png")

it = 5
dim = 3**it
alfombra=np.ones((dim,dim))
sierpinski(alfombra,dim)
pintar(alfombra,dim)
