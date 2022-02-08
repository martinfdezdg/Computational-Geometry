# -*- coding: utf-8 -*-
"""
PRÁCTICA 1: ATRACTOR LOGÍSTICO
Martín Fernández de Diego
"""

import matplotlib.pyplot as plt
import numpy as np
import random as rand

"""
Dados dos elementos o conjuntos de elementos y un épsilon suficientemente pequeño
devuelve si son lo suficientemente parecidos
"""
def igual(x,y,epsilon=0.001):
    return np.max(abs(x - y)) < epsilon

"""
Dado un punto x y una r
devuelve f(x,r)
"""
def logistica(x,r):
    return r*x*(1-x)

"""
Dado un punto x0, una r, una función f y un entero n
devuelve f^n(x0,r)
"""
def fn(x0,r,f,n):
    x = x0
    for j in range(n):
        x = f(x,r)
    return(x)
    
"""
Dado un punto x0, una r, una función f y un entero N
devuelve una lista de f^i(x0) con 0 <= i < N
"""
def orbita(x0,r,f,N):
    orb = np.empty([N])
    for i in range(N):
        orb[i] = fn(x0,r,f,i)
    return(orb)

"""
Dado una subórbita
devuelve min(i) tal que |f^n(x0) - f^i(x0)| < epsilon
"""
# Encontramos el tamaño de los ciclos
def periodo(suborb):
    N = len(suborb)
    per = -1
    for i in np.arange(2,N-1,1):
        # Buscamos la distancia mínima a partir de la cual se vuelven a repetir puntos
        if igual(suborb[N-1],suborb[N-i]):
            per = i-1
            break
    return(per)

"""
Dada una función f y dos enteros N0 y N
devuelve una lista ordenada con los per elementos de una subórbita final de N respecto de una órbita de N0 elementos
"""
def atractor(x0,r,f,N0,N):
    orb = orbita(x0,r,f,N0)
    ult = orb[-1*np.arange(N,0,-1)]
    per = periodo(ult)
    V0 = np.sort([ult[N-1-i] for i in range(per)])
    return V0

"""
Dada una función f y dos enteros N0 y MAX_N0
devuelve el mínimo N0 que cumpla la condición de recubrimiento del algoritmo
"""
# Encontramos un N1 a partir del que se puede intuir una cuenca de atracción
def tiempo_transitorio(x0,r,f,N0,MAX_N0):
    recubierto = False
    while not recubierto and N0 < MAX_N0:
        # Primera iteración fuera del bucle para obtener máximos de referencia
        xn = fn(x0,r,f,N0)
        suborb = orbita(xn,r,f,N0)
        max_ant, min_ant = abs(np.max(suborb)), abs(np.min(suborb))
        # Si no cumple la condición del algoritmo, no hay recubrimiento para ese tiempo transitorio
        for i in np.arange(1,3):
            xn = logistica(suborb[-1],r)
            suborb = orbita(xn,r,f,N0*2**i)
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
Dada una función f, dos enteros N0 y N que indican longitudes en la sucesión, un conjunto atractor, dos enteros que limitan la ejecución y un épsilon suficientemente pequeño
devuelve el mayor épsilon posible que define un entorno de atracción respecto al V0
"""
def error_x(x0,r,f,N0,N,V0,MAX_N0,MAX_ERR,epsilon=0.1):
    estable = False
    while not estable and MAX_ERR > 0:
        estable = True
        # Comprobación de si es un punto estable
        N1_mayor = tiempo_transitorio(min(1,x0+epsilon),r,f,N0,MAX_N0)
        N1_menor = tiempo_transitorio(max(0,x0-epsilon),r,f,N0,MAX_N0)
        if (N1_mayor >= MAX_N0 or N1_menor >= MAX_N0):
            epsilon /= 2
            MAX_ERR -= 1
            estable = False
        else :
            # Comprobación de si está cerca de una bifurcación
            V1_mayor = atractor(abs(min(1,x0+epsilon)),r,f,N1_mayor,N)
            V1_menor = atractor(abs(x0-epsilon),r,f,N1_menor,N)
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

"""
Dada una función f, dos enteros N0 y N que indican longitudes en la sucesión, un conjunto atractor, dos enteros que limitan la ejecución y un épsilon suficientemente pequeño
devuelve el mayor épsilon posible que define un entorno de atracción respecto al V0
"""
def error_r(x0,r,f,N0,N,V0,MAX_ERR,epsilon=0.1):
    estable = False
    while not estable and MAX_ERR > 0:
        estable = True
        # Comprobación de si está cerca de una bifurcación
        r_mayor = min(4.000,r + epsilon)
        V1_mayor = atractor(x0,r_mayor,f,N0,N)
        r_menor = max(3.544,r - epsilon)
        V1_menor = atractor(x0,r_menor,f,N0,N)
        if (len(V1_mayor) != len(V0) or len(V1_menor) != len(V0)):
            epsilon /= 2
            MAX_ERR -= 1
            estable = False
    return epsilon



# FORMATO
class Formato:
    BOLD = "\033[1m"
    RESET = "\033[0m"

# CONSTANTES
MAX_N0, MAX_ERR = 500, 10
N0, N = 50, 50
x0 = rand.uniform(0,1)

# APARTADO i)
print("\n" + Formato.BOLD + "Apartado i)" + Formato.RESET)

i = 0
V0_aux = np.empty([N])
while i < 2:
    r = rand.uniform(3.000,3.544)
    N1 = tiempo_transitorio(x0,r,logistica,N0,MAX_N0)
    if N1 < MAX_N0:
        V0 = atractor(x0,r,logistica,N1,N)
        err_x = error_x(x0,r,logistica,N0,N,V0,MAX_N0,MAX_ERR)
        if len(V0) != len(V0_aux) or not igual(V0,V0_aux):
            print(i+1, "> Cuenca de atracción de x0 =", x0, "+-", err_x, "en r =", r)
            print("  >", V0)
            V0_aux = V0
            i += 1

# APARTADO ii)
print("\n" + Formato.BOLD + "Apartado ii)" + Formato.RESET)

V0 = np.empty([])
while V0.size != 8:
    r = rand.uniform(3.544,4.000)
    V0 = atractor(x0,r,logistica,N0,N)
err_r = error_r(x0,r,logistica,N0,N,V0,MAX_ERR)
print("  > Hay una cuenca de atracción de 8 elementos con r =", r, "+-", err_r)
print("  >", V0)
