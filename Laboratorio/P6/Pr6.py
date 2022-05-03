# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:53:36 2020
@author: Jorge Sainero y Lucas de Torre con la plantilla de Robert
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

os.getcwd()

# q = variable de posición, dq0 = \dot{q}(0) = valor inicial de la derivada
# d = granularidad del parámetro temporal


def deriv(q, dq0, d):
    #dq = np.empty([len(q)])
    dq = (q[1:len(q)]-q[0:(len(q)-1)])/d
    dq = np.insert(dq, 0, dq0)  # dq = np.concatenate(([dq0],dq))
    return dq

# Ecuación de un sistema dinámico continuo
# Ejemplo de oscilador simple


def F(q):
    #k = 1
    ddq = -2*q*(q**2-1)
    return ddq

# Resolución de la ecuación dinámica \ddot{q} = F(q), obteniendo la órbita q(t)
# Los valores iniciales son la posición q0 := q(0) y la derivada dq0 := \dot{q}(0)


def orb(n, q0, dq0, F, args=None, d=0.001, n0=0):
    #q = [0.0]*(n+1)
    q = np.empty([n+1])
    q[0] = q0
    q[1] = q0 + dq0*d
    for i in np.arange(2, n+1):
        args = q[i-2]
        q[i] = - q[i-2] + d**2*F(args) + 2*q[i-1]
    return q[n0:]  # np.array(q),


def periodos(q, d, max=True):
    # Si max = True, tomamos las ondas a partir de los máximos/picos
    # Si max == False, tomamos los las ondas a partir de los mínimos/valles
    epsilon = 5*d
    dq = deriv(q, dq0=None, d=d)  # La primera derivada es irrelevante
    if max == True:
        waves = np.where(
            (np.round(dq, int(-np.log10(epsilon))) == 0) & (q > 0))
    if max != True:
        waves = np.where(
            (np.round(dq, int(-np.log10(epsilon))) == 0) & (q < 0))
    diff_waves = np.diff(waves)
    waves = waves[0][1:][diff_waves[0] > 1]
    pers = diff_waves[diff_waves > 1]*d
    return pers, waves


d = 10**(-3.5)
# Pintamos el espacio de fases


def simplectica(q0, dq0, F, col=0, d=10**(-3.5), n=int(16/d), marker='-', n0=0):
    q = orb(n, q0=q0, dq0=dq0, F=F, d=d, n0=n0)
    dq = deriv(q, dq0=dq0, d=d)
    p = dq/2
    plt.plot(q, p, marker, c=plt.get_cmap("winter")(col))


def print_espacio_fasico(n=int(16/d), n0=0):
    fig = plt.figure(figsize=(8, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    seq_q0 = np.linspace(0., 1., num=12)
    seq_dq0 = np.linspace(0., 2., num=12)
    for i in range(len(seq_q0)):
        for j in range(len(seq_dq0)):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            ax = fig.add_subplot(1, 1, 1)
            col = (1+i+j*(len(seq_q0)))/(len(seq_q0)*len(seq_dq0))
            #ax = fig.add_subplot(len(seq_q0), len(seq_dq0), 1+i+j*(len(seq_q0)))
            simplectica(q0=q0, dq0=dq0, F=F, col=col,
                        marker='ro', d=d, n=n, n0=n0)
    ax.set_xlabel("q(t)", fontsize=12)
    ax.set_ylabel("p(t)", fontsize=12)
    #fig.savefig('Simplectic.png', dpi=250)
    plt.show()

#################################################################
#  CÁLCULO DE ÓRBITAS
#################################################################


def apartado1():
    print("Apartado 1\n")

    #################################################################
    #  ESPACIO FÁSICO
    #################################################################

    print_espacio_fasico()


#################################################################
#  CÁLCULO DEL ÁREA DEL ESPACIO FÁSICO
#################################################################
# Tomamos un par (q(0), p(0)) y nos quedamos sólo en un trozo/onda de la órbita, sin repeticiones
# Para eso, tomaremos los periodos de la órbita, que definen las ondas
# Paso1: Buscamos las condiciones iniciales que maximizan/minimizan en área
#q0 = 0.
#dq0 = 2.
#d = 10**(-3.5)
def area_espacio_fasico(d):
    areas = []
    for q0 in np.linspace(0., 1., num=11):
        for dq0 in np.linspace(0., 2., num=11):

            n = int(32/d)
            #t = np.arange(n+1)*d
            q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
            dq = deriv(q, dq0=dq0, d=d)
            p = dq/2
            # Tomaremos los periodos de la órbita, que definen las ondas
            T, W = periodos(q, d, max=False)
            if len(W) > 1:

                # Tomamos la mitad de la "curva cerrada" para integrar más fácilmente
                mitad = np.arange(W[0], W[0]+np.int((W[1]-W[0])/2), 1)

                # Regla de Simpson
                area = simps(p[mitad], q[mitad])
                areas.append([q0, dq0, area])
    sort_areas = sorted(areas, key=lambda x: x[2])
    min_a = sort_areas[0][2]
    max_a = 2*sort_areas[len(sort_areas)-1][2]
    return max_a-min_a


def apartado2():
    # Definimos distintos valores de d
    iseq = np.linspace(3.01, 3.99, num=11)
    # Hallamos el área de cada espacio fásico para cada d
    areas = [area_espacio_fasico(10**(-d)) for d in iseq]
    # Hallamos el valor absoluto de las diferencias de cada área comparada con "la mejor"
    resta_areas = [abs(areas[i]-areas[10]) for i in range(len(areas)-1)]
    sort_resta_areas = sorted(resta_areas)
    # Cogemos el cuantil de orden 0,9
    print("El área calculada es:", round(areas[10], 3),
          "con un error de", round(sort_resta_areas[8], 3))

    ################################################################
    ################## TEOREMA DE LIOUVILLE ########################
    ################################################################
    for i in range(4):
        print_espacio_fasico((i+1)*200, i*200)


apartado1()
apartado2()
