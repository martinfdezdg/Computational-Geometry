# -*- coding: utf-8 -*-
"""
PRÁCTICA 1: ATRACTOR LOGÍSTICO
Martín Fernández de Diego
"""

import matplotlib.pyplot as plt
import numpy as np
import random as rand

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
        if abs(suborb[N-1] - suborb[N-i]) < epsilon:
            per = i-1
            break
    return(per)

"""
Dada una función f, dos enteros N0 y N y un epsilon suficientemente pequeño
devuelve una lista ordenada con los per elementos de una suborbita final de N respecto de una orbita de N0 elementos
"""
def atractor(x0,f,N0,N,epsilon=0.001):
    orb = orbita(x0,f,N0)
    # plt.plot(orb)
    ult = orb[-1*np.arange(N,0,-1)]
    # print "ult:", str(ult)
    per = periodo(ult, epsilon)
    # print "per:", str(per)
    V0 = np.sort([ult[N-1-i] for i in range(per)])
    return V0

"""
"""
def error(V0,f,M):
    per = len(V0)
    for i in range(M):
        # Generamos M ciclos más
        x = fn(V0[per-1],logistica,i*per+1)
        V1 = orbita(x,f,per)
        for i in range(per):
            



# VARIABLES
BOLD = "\033[1m"
RESET = "\033[0m"

N0 = 200
N = 50
x0 = rand.uniform(0,1)

print "Punto inicial x =", x0

# APARTADO i)
print "\n" + BOLD + "Apartado i)" + RESET

i = 0
while i < 2:
    r = rand.uniform(3.000,3.544)
    V0 = atractor(x0,logistica,N0,N)
    # TODO: Estimación del error y evitar que sean la misma órbita
    # err = error(V0,)
    if len(V0) != 0:
        print "Órbita", i+1, "de", len(V0), "elementos con r =", r
        print "Conjunto atractor:", V0
        i += 1
        # plt.show()

# APARTADO ii)
print "\n" + BOLD + "Apartado ii)" + RESET

V0 = np.empty([])
while V0.size != 8:
    r = rand.uniform(3.544,4.000)
    V0 = atractor(x0,logistica,N0,N)

print "Órbita de 8 elementos con r =", r
print "Conjunto atractor:", V0
