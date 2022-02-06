# -*- coding: utf-8 -*-
"""
PRÁCTICA 1: ATRACTOR LOGÍSTICO
Martín Fernández de Diego
"""

import matplotlib.pyplot as plt
import numpy as np
import random as rand

"""
"""
def igual(x,y,epsilon=0.001):
    return np.max(abs(x - y)) < epsilon

"""
Dado un punto x
devuelve f(x)
"""
def logistica(x):
    return r*x*(1-x);

"""
Dado un punto x0, una función f y un entero n
devuelve f^n(x0)
"""
def fn(x0,f,n):
    x = x0
    for j in range(n):
        x = f(x)
    return(x)
    
"""
Dado un punto x0, una función f y un entero N
devuelve una lista de f^i(x0) con 0 <= i < N
"""
def orbita(x0,f,N):
    orb = np.empty([N])
    for i in range(N):
        orb[i] = fn(x0,f,i)
    return(orb)

"""
Dado una suborbita y un epsilon suficientemente pequeño
devuelve min(i) tal que |f^n(x0) - f^i(x0)| < epsilon
"""
# Encontramos el tamaño de los ciclos
def periodo(suborb,epsilon=0.001):
    N = len(suborb)
    per = -1
    for i in np.arange(2,N-1,1):
        # Buscamos la distancia mínima a partir de la cual se vuelven a repetir puntos
        if abs(suborb[N-1]) - abs(suborb[N-i]) < epsilon:
            per = i-1
            break
    return(per)

"""
Dada una función f, dos enteros N0 y N y un epsilon suficientemente pequeño
devuelve una lista ordenada con los per elementos de una suborbita final de N respecto de una orbita de N0 elementos
"""
def atractor(x0,f,N0,N,epsilon=0.001):
    orb = orbita(x0,f,N0)
    ult = orb[-1*np.arange(N,0,-1)]
    per = periodo(ult, epsilon)
    V0 = np.sort([ult[N-1-i] for i in range(per)])
    return V0

"""
"""
def tiempo_transitorio(x0,f,N0,MAX_M):
    recubierto = False
    while not recubierto and N0 < MAX_M:
        # Primera iteración fuera del bucle para obtener máximos de referencia
        xn = fn(x0,f,N0)
        suborb = orbita(xn,f,N0)
        max_ant, min_ant = abs(np.max(suborb)), abs(np.min(suborb))
        # Si no cumple la condición del algoritmo, no hay recubrimiento para ese tiempo trasnitorio
        for i in np.arange(1,3):
            xn = logistica(suborb[-1])
            suborb = orbita(xn,f,N0*2**i)
            max_act, min_act = abs(np.max(suborb)), abs(np.min(suborb))
            # Condición del algoritmo
            if (max_act <= max_ant and min_act >= min_ant):
                max_ant, min_ant = max_act, min_act
                recubierto = True
            else :
                # Aumentamos el tiempo transitorio
                N0 *= 2
                recubierto = False
                break
    return N0

"""
"""
def error(x0,f,N0,N,V0,MAX_M,MAX_ERR,epsilon=0.1):
    estable = False
    while not estable and MAX_ERR > 0:
        estable = True
        # Comprobación de si es un punto estable
        N1_mayor = tiempo_transitorio(abs(x0+epsilon),f,N0,MAX_M)
        N1_menor = tiempo_transitorio(abs(x0-epsilon),f,N0,MAX_M)
        if (N1_mayor >= MAX_M or N1_menor >= MAX_M):
            epsilon /= 2
            MAX_ERR -= 1
            estable = False
        else :
            # Comprobación de si está cerca de una bifurcación
            V1_mayor = atractor(abs(x0+epsilon),f,N1_mayor,N)
            V1_menor = atractor(abs(x0-epsilon),f,N1_menor,N)
            if (len(V1_mayor) != len(V0) or len(V1_menor) != len(V0)):
                epsilon /= 2
                MAX_ERR -= 1
                estable = False
            # Comprobación de si es suficientemente estable
            elif (not igual(V1_mayor,V0) or not igual(V1_menor,V0)):
                epsilon /= 2
                MAX_ERR -= 1
                estable = False
    return epsilon



# VARIABLES
BOLD = "\033[1m"
RESET = "\033[0m"

MAX_M, MAX_ERR = 500, 5
N0, N = 50, 50
x0 = rand.uniform(0,1)

# APARTADO i)
print("\n" + BOLD + "Apartado i)" + RESET)

i = 0
while i < 2:
    r = rand.uniform(3.000,3.544)
    N1 = tiempo_transitorio(x0,logistica,N0,MAX_M)
    if N1 < MAX_M:
        V0 = atractor(x0,logistica,N1,N)
        err = error(x0,logistica,N0,N,V0,MAX_M,MAX_ERR)
        print(i+1, "> Cuenca de atracción de x0 =", x0, "+-",err , "en r =", r)
        print("  >", V0)
        i += 1

'''
# APARTADO ii)
print("\n" + BOLD + "Apartado ii)" + RESET)

V0 = np.empty([])
while V0.size != 8:
    r = rand.uniform(3.544,4.000)
    V0 = atractor(x0,logistica,N0,N)

print("Cuenca de atracción de 8 elementos con r =", r)
print("Conjunto atractor:", V0)
'''
