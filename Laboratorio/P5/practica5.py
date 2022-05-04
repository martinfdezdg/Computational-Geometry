"""
PRÁCTICA 5: DEFORMACIÓN DE VARIEDADES DIFERENCIABLES
Belén Sánchez Centeno
Martín Fernández de Diego
"""

#import os
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d

from matplotlib import animation
#from mpl_toolkits.mplot3d.axes3d import Axes3D

"""
Dadas las coordenadas de x y z
devuelve la proyección de x sobre el eje z
z0 = 1 porque el polo extraído es el (0,0,1)
"""
def proj(x,z,z0=1,alpha=1):
    z0 = z*0+z0
    eps = 1e-16
    x_trans = x/(abs(z0-z)**alpha+eps)
    return(x_trans)
    # Nótese que añadimos un épsilon para evitar dividir entre 0
    
"""
Animación del APARTADO ii)
"""
def animate(t):
    z0 = -1
    xt = 2/(2*(1-t) + (1-z)*t + eps)*x
    yt = 2/(2*(1-t) + (1-z)*t + eps)*y
    zt = (-1)*t + z*(1-t)
    x2t = 2/(2*(1-t) + (1-z2)*t + eps)*x2
    y2t = 2/(2*(1-t) + (1-z2)*t + eps)*y2
    z2t = (-1)*t + z2*(1-t)
    
    ax = plt.axes(projection='3d')
    ax.set_zlim3d(-1,1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1, alpha=0.5, cmap='viridis', edgecolor='none')
    ax.plot(x2t, y2t, z2t, '-b', c="white", zorder=3)
    return ax

def init():
    return animate(0)



# FORMATO
class Formato:
    BOLD = "\033[1m"
    RESET = "\033[0m"



# APARTADO i)
print("\n" + Formato.BOLD + "Apartado i)" + Formato.RESET)

"""
2-esfera
definición en polares
"""
u = np.linspace(0.05, np.pi, 30)
v = np.linspace(0, 2 * np.pi, 60)

"""
2-esfera
transformación en paramétricas
"""
x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))

"""
gamma-curva
definición en paramétricas
"""
t2 = np.linspace(0.055, 1, 200)

x2 = abs(t2) * np.sin(40 * t2/2)**2
y2 = abs(t2) * -np.cos(40 * t2/2)**2
z2 = np.sqrt(1-x2**2-y2**2)

# Representación: 2-esfera + 2-esfera proyectada
z0 = -1

fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(hspace=0.4, wspace=0.2)

ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', alpha=0.5, edgecolor='none')
ax.plot(x2, y2, z2, '-b', c="white", zorder=3)

ax.set_title('2-sphere');
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.set_xlim3d(-8,8)
ax.set_ylim3d(-8,8)

ax.plot_surface(proj(x, z, z0), proj(y, z, z0), z*0+z0, rstride=1, cstride=1, cmap='viridis', alpha=0.5, edgecolor='purple')
ax.plot(proj(x2, z2, z0), proj(y2, z2, z0), z0, '-b', c="white", zorder=3)

ax.set_title('Stereographic projection');
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
fig.savefig('stereo2.png', dpi=250)
plt.close(fig) 



# APARTADO ii)
print("\n" + Formato.BOLD + "Apartado ii)" + Formato.RESET)

"""
2-esfera proyectada - familia paramétrica
"""
t = 0.1
eps = 1e-16

# Representación: animación
fig = plt.figure(figsize=(6, 6))
ani = animation.FuncAnimation(fig, animate, np.arange(0, 1, 0.05), init_func=init, interval=20)
ani.save("ejemplo.gif", fps = 5)
plt.close(fig)
