"""
PRÁCTICA 6: TEOREMA DE LIOUVILLE
Belén Sánchez Centeno
Martín Fernández de Diego
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from matplotlib import animation

# FORMATO
class Formato:
    BOLD = "\033[1m"
    RESET = "\033[0m"

os.getcwd()

# q = variable de posición, dq0 = \dot{q}(0) = valor inicial de la derivada
# d = granularidad del parámetro temporal
def deriv(q, dq0, d):
   # dq = np.empty([len(q)])
   dq = (q[1:len(q)]-q[0:(len(q)-1)])/d
   dq = np.insert(dq, 0, dq0) # dq = np.concatenate(([dq0],dq))
   return dq

# Ecuación de un sistema dinámico continuo
# Ejemplo de oscilador simple
def Fejemplo(q):
    k = 1
    ddq = - k*q
    return ddq

# Oscilador no lineal
def F(q):
    ddq = -2*q*(q**2-1)
    return ddq

# Resolución de la ecuación dinámica \ddot{q} = F(q), obteniendo la órbita q(t)
# Los valores iniciales son la posición q0 := q(0) y la derivada dq0 := \dot{q}(0)
def orb(n, q0, dq0, F, args=None, d=0.001):
    # q = [0.0]*(n+1)
    q = np.empty([n+1])
    q[0] = q0
    q[1] = q0 + dq0*d
    for i in np.arange(2, n+1):
        args = q[i-2]
        q[i] = - q[i-2] + d**2*F(args) + 2*q[i-1]
    return q # np.array(q),

def periodos(q, d, max=True):
    # Si max == True, tomamos las ondas a partir de los máximos/picos
    # Si max == False, tomamos los las ondas a partir de los mínimos/valles
    epsilon = 5*d
    dq = deriv(q, dq0=None, d=d) #La primera derivada es irrelevante
    if max == True:
        waves = np.where((np.round(dq, int(-np.log10(epsilon))) == 0) & (q > 0))
    if max != True:
        waves = np.where((np.round(dq, int(-np.log10(epsilon))) == 0) & (q < 0))
    diff_waves = np.diff(waves)
    waves = waves[0][1:][diff_waves[0]>1]
    pers = diff_waves[diff_waves>1]*d
    return pers, waves



# APARTADO i)
print("\n" + Formato.BOLD + "Apartado i)" + Formato.RESET)

"""
Pintamos el espacio de fases
"""
def simplectica(q0, dq0, F, col=0, d=10**(-4), n=0, marker='-', plot=False):
    q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
    dq = deriv(q, dq0=dq0, d=d)
    p = dq/2
    if plot: plt.plot(q, p, marker, c=plt.get_cmap("winter")(col))

"""
Dada una ecuación diferencial F y un delta d
devuelve el espacio fásico completo
"""
def espacio_fasico_completo(F, d, plot=False):
    if plot:
        fig = plt.figure(figsize=(8, 5))
        fig.subplots_adjust(hspace=0.4, wspace=0.2)
        ax = fig.add_subplot(1, 1, 1)
    # Condiciones iniciales:
    seq_q0 = np.linspace(0., 1., num=10)
    seq_dq0 = np.linspace(0., 2, num=10)
    for i in range(len(seq_q0)):
        for j in range(len(seq_dq0)):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            col = (1+i+j*(len(seq_q0)))/(len(seq_q0)*len(seq_dq0))
            # ax = fig.add_subplot(len(seq_q0), len(seq_dq0), 1+i+j*(len(seq_q0)))
            simplectica(q0=q0, dq0=dq0, F=F, col=col, d=d, n=int(16/d), marker='o', plot=plot)
    if plot:
        ax.set_xlabel("q(t)", fontsize=12)
        ax.set_ylabel("p(t)", fontsize=12)
        # fig.savefig('Simplectic.png', dpi=250)
        plt.show()

# CÁLCULO DE ÓRBITAS
"""
Buscamos el delta de [10^-4,10^-3] con mayor granularidad en la órbita
"""
q0 = 0.
dq0 = 1.
fig, ax = plt.subplots(figsize=(12, 5))
plt.ylim(-2, 2)
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("t = n $\delta$", fontsize=12)
ax.set_ylabel("q(t)", fontsize=12)
iseq = np.array([3, 3.2, 3.4, 3.6, 3.8, 4])
horiz = 32
for i in iseq:
    d = 10**(-i)
    n = int(horiz/d)
    t = np.arange(n+1)*d
    q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
    plt.plot(t, q, 'o', markersize=0.5/i, label='$\delta$ ='+str(np.around(d, 4)), c=plt.get_cmap("winter")(i/np.max(iseq)))
    ax.legend(loc=3, frameon=False, fontsize=12)
# plt.savefig('Time_granularity.png', dpi=250)
plt.show()
"""
A simple vista, todos los delta parecen arrojar buenos niveles de granularidad.
Sin embargo, al hacer zoom en la gráfica, vemos que delta=10^-4 da mejores resultados.
"""

# ESPACIO FÁSICO
"""
Representamos el espacio fasico para el mejor delta
"""
d = 10**(-4)
espacio_fasico_completo(F, d, True);



# APARTADO ii)
print("\n" + Formato.BOLD + "Apartado ii)" + Formato.RESET)

"""
Dada la ecuación t=nd, como debe ser t=1/4 y d=10^-4 => n=t/d => n=2500
Dada la ecuacion n=int(horiz/d), como n=2500 y d=10^-4 => horiz~n*d => horiz=0.25
"""

"""
Dada una ecuación diferencial F, un delta d y un instante horiz
devuelve el valor del área de D_horiz
ConvexHull(...).volume devuelve el área de un recubrimiento convexo del conjunto.
El recubrimiento de un conjunto convexo tiene mayor area que el conjunto recubierto.
Además, como el conjunto resultante es convexo, el área queda determinada con a_max.
"""
def areaConvexa(F, d=10**(-4), horiz=10**(-4), plot=False):
    seq_q0 = np.linspace(0., 1., num=20)
    seq_dq0 = np.linspace(0., 2, num=20)
    q2 = np.array([])
    p2 = np.array([])
    
    if plot: ax = fig.add_subplot(1, 1, 1)
    
    for i in range(len(seq_q0)):
        for j in range(len(seq_dq0)):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            n = int(horiz/d)
            t = np.arange(n+1)*d
            q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
            dq = deriv(q, dq0=dq0, d=d)
            p = dq/2
            q2 = np.append(q2, q[-1])
            p2 = np.append(p2, p[-1])
            
            if plot:
                plt.xlim(-2.2, 2.2)
                plt.ylim(-1.2, 1.2)
                plt.rcParams["legend.markerscale"] = 6
                ax.set_xlabel("q(t)", fontsize=12)
                ax.set_ylabel("p(t)", fontsize=12)
                plt.plot(q[-1], p[-1], marker="o", markersize=10, c=plt.get_cmap("winter")(i/np.max(iseq)))

    X = np.array([q2,p2]).T
    hull = ConvexHull(X)
    
    if plot:
        convex_hull_plot_2d(hull)
        plt.show()
    
    return hull.volume

# CÁLCULO DEL ÁREA
area_t = areaConvexa(F,10**(-4),0.25)

# CÁLCULO DEL ERROR
iseq = np.linspace(3., 3.9, num=10)

"""
Dada la lista de exponentes [3., 3.1, ..., 3.9] para determinar el valor de delta
calcula el area en el instante 1/4 para cada delta
devuelve la máx diferencia de cada area con el area del delta con mayor granularidad
"""
areas_esp_fas = [areaConvexa(F,10**(-i),0.25) for i in iseq]
areas_inc = [abs(areas_esp_fas[i]-area_t) for i in range(len(areas_esp_fas)-1)]
area_max_error = np.max(areas_inc)

print("Área en D_1/4:", area_t, "+-", area_max_error)

# TEOREMA DE LIOUVILLE
tseq = np.linspace(0.0001, 0.25, num=26)

areas_0_t = [areaConvexa(F,10**(-4),t) for t in tseq]
print("Áreas de D_0 a D_1/4")
print(areas_0_t)
print(np.max(abs(np.max(areas_0_t) - 1), abs(np.min(areas_0_t) - 1)))



# APARTADO iii)
print("\n" + Formato.BOLD + "Apartado iii)" + Formato.RESET)

def animate(ft):
    seq_q0 = np.linspace(0., 1., num=10)
    seq_dq0 = np.linspace(0., 2, num=10)
    q2 = np.array([])
    p2 = np.array([])
    
    ax = fig.add_subplot(1, 1, 1)
    
    horiz = ft
    for i in range(len(seq_q0)):
        for j in range(len(seq_dq0)):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            d = 10**(-4)
            n = int(horiz/d)
            t = np.arange(n+1)*d
            q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
            dq = deriv(q, dq0=dq0, d=d)
            p = dq/2
            q2 = np.append(q2, q[-1])
            p2 = np.append(p2, p[-1])
            
            plt.xlim(-2.2, 2.2)
            plt.ylim(-1.2, 1.2)
            plt.rcParams["legend.markerscale"] = 6
            ax.set_xlabel("q(t)", fontsize=12)
            ax.set_ylabel("p(t)", fontsize=12)
            plt.plot(q[-1], p[-1], marker="o", markersize=10, c=plt.get_cmap("winter")(i/np.max(iseq)))
            
    return ax

def init():
    return animate(0.2)

# Representación: animación
fig = plt.figure(figsize=(6, 6))
ani = animation.FuncAnimation(fig, animate, np.arange(0.1, 5, 0.1), init_func=init, interval=48)
ani.save("ejemplo.gif", fps = 12)
plt.close(fig)
