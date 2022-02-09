"""
PRÁCTICA 1: ATRACTOR LOGÍSTICO
Martín Fernández de Diego
Belén Sánchez Centeno
"""

import matplotlib.pyplot as plt
import numpy as np
import random as rand

"""
Dados una x, un cierto eps y un intervalo [a,b]
ajusta la eps para que x+eps y x-eps quepan en dicho intervalo
"""
# Restando epsilon evitamos que epsilon sature en los extremos indeseablemente
def ajustar(x,eps,a,b,epsilon=0.001):
    if x + eps > b:
        return b - x - epsilon
    if x - eps < a:
        return x - a - epsilon
    return eps

"""
Dados dos elementos
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
Dado un punto x0, una r, una función f y dos enteros M y N con N > M
devuelve una lista de f^i(xm) con 0 <= i < N y xm = f^m(x0)
"""
def suborbita(x0,r,f,M,N):
    xm = fn(x0,r,f,M)
    orb = np.empty([N-M])
    for i in range(N-M):
        orb[i] = fn(xm,r,f,i)
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
Dada una función f y dos enteros N y N_cola
devuelve una lista ordenada con los per elementos de una subórbita final de N_cola respecto de una órbita de N elementos
"""
def atractor(x0,r,f,N,N_cola):
    orb = orbita(x0,r,f,N)
    ult = orb[-1*np.arange(N_cola,0,-1)]
    per = periodo(ult)
    V = np.sort([ult[N_cola-1-i] for i in range(per)])
    return V

"""
Dada una función f y un entero N
devuelve el mínimo m que cumpla la condición de recubrimiento del algoritmo
"""
# Encontramos un m a partir del que se puede intuir una cuenca de atracción
def tiempo_transitorio(x0,r,f,N):
    m = N
    recubierto = False
    while not recubierto:
        # Si no cumple la condición del algoritmo, no hay recubrimiento para ese tiempo transitorio
        for i in range(3):
            suborb = suborbita(x0,r,f,2**i*m,2**(2**i)*m) # m - 2m / 2m - 4m / 4m - 16m
            Dt_act = np.max(suborb) - np.min(suborb)
            # Condición del algoritmo
            if (i == 0) or (Dt_act <= Dt_ant and igual(Dt_act,Dt_ant)):
                # Guardamos el recubrimiento anterior
                Dt_ant = Dt_act
                recubierto = i == 2;
            else :
                # Aumentamos el tiempo transitorio
                m *= 2
                break
    return m

"""
Dada una función f, dos enteros N y N_cola que indican longitudes en la sucesión y el conjunto atractor
devuelve el mayor épsilon posible que define un entorno de atracción respecto al V
"""
def error_x(x0,r,f,N,N_cola,V,epsilon=0.5):
    epsilon = ajustar(x0,epsilon,0,1)
    estable = False
    while not estable:
        N_der = tiempo_transitorio(x0+epsilon,r,f,N)
        N_izq = tiempo_transitorio(x0-epsilon,r,f,N)
        V_der = atractor(x0+epsilon,r,f,N_der,N_cola)
        V_izq = atractor(x0-epsilon,r,f,N_izq,N_cola)
        # Comprobamos si es un punto estable
        if not ((len(V_der) == len(V) and len(V_izq) == len(V)) and (igual(V_der,V) and igual(V_izq,V))):
            epsilon /= 2
        else :
            estable = True
    return epsilon



# FORMATO
class Formato:
    BOLD = "\033[1m"
    RESET = "\033[0m"

# CONSTANTES
N0, N_cola = 100, 50

x0 = rand.uniform(0,1)

# APARTADO i)
print("\n" + Formato.BOLD + "Apartado i)" + Formato.RESET)

i = 0
while i < 2:
    r = rand.uniform(3.000,3.544)
    # Calculamos un numero de iteraciones adecuado
    N = tiempo_transitorio(x0,r,logistica,N0)
    # Calculamos el conjunto atractor
    V = atractor(x0,r,logistica,N,N_cola)
    # Si los conjuntos obtenidos son distintos
    if i == 0 or not ((len(V) == len(V_ant)) and igual(V,V_ant)):
        # Calculamos el posible error
        err_x = error_x(x0,r,logistica,N,N_cola,V)
        # Mostramos
        print(i+1,"- Cuenca de atracción de x0 =",x0,"+-",err_x,"en r =",r)
        print("  >",V)
        V_ant = V
        i += 1
        
# APARTADO ii)
print("\n" + Formato.BOLD + "Apartado ii)" + Formato.RESET)

ext_izq, ext_der = [], []
err_r = 0.001
intervalo = False
for r in np.arange(3.544,4.000,err_r):
    # Calculamos el conjunto atractor
    V = atractor(x0,r,logistica,N0,N_cola)
    # Si no estamos en el intervalo y encontramos una cuenca de 8 elementos
    if not intervalo and len(V) == 8:
        ext_izq.append(r)
        intervalo = True
    # Si estamos en el intervalo y encontramos una cuenca que no tiene 8 elementos
    elif intervalo and len(V) != 8:
        ext_der.append(r-err_r)
        intervalo = False

print("  -",len(ext_izq),"intervalos de cuencas de atracción de 8 elementos con una sensibilidad de",err_r)
print("  >",end=' ')
for izq,der in zip(ext_izq,ext_der):
    print("[",izq,",",der,"]",end='')

# Tomamos un valor aleatorio del intervalo
for r in np.random.permutation(np.arange(3.544,4.000,0.001)):
    V = atractor(x0,r,logistica,N0,N_cola)
    if len(V) == 8:
        print("\nEj: Cuenca de atracción de 8 elementos en r =",r)
        print("  >",V)
        break
