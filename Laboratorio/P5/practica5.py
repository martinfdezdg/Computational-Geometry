"""
PRÁCTICA 5: PCA Y ANALOGÍA
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
2-sphere
"""
u = np.linspace(0, np.pi, 30)
v = np.linspace(0, 2 * np.pi, 60)

x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))

t2 = np.linspace(0.001, 1, 200)
x2 = abs(t2) * np.sin(100 * t2/2)
y2 = abs(t2) * np.cos(100 * t2/2)
z2 = np.sqrt(1-x2**2-y2**2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.plot(x2, y2, z2, '-b', c="white", zorder=3)



# FORMATO
class Formato:
    BOLD = "\033[1m"
    RESET = "\033[0m"

# APARTADO i)
print("\n" + Formato.BOLD + "Apartado i)" + Formato.RESET)

"""
2-esfera proyectada
"""

"""
proyección
"""
def proj(x,z,z0=1,alpha=1):
    z0 = z*0+z0
    eps = 1e-16
    x_trans = x/(abs(z0-z)**alpha+eps)
    return(x_trans)
    #Nótese que añadimos un épsilon para evitar dividi entre 0!!

fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(hspace=0.4, wspace=0.2)

#c2 = np.sqrt(x2**2+y2**2)
#col = plt.get_cmap("hot")(c2/np.max(c2))

ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.plot(x2, y2, z2, '-b', c="white", zorder=3)

#ax.scatter(x2, y2, z2, '-b',c=col,zorder= 3,s=0.1)
ax.set_title('2-sphere');
#ax.text(0.5, 90, 'PCA-'+str(i), fontsize=18, ha='center')

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.set_xlim3d(-8,8)
ax.set_ylim3d(-8,8)
#ax.set_zlim3d(0,1000)

z0 = 1

ax.plot_surface(proj(x,z,z0=z0), proj(y,z,z0=z0), z*0+1, rstride=1, cstride=1, cmap='viridis', alpha=0.5, edgecolor='purple')
ax.plot(proj(x2, z2, z0=z0), proj(y2, z2, z0=z0), 1, '-b', c="white", zorder=3)

#ax.scatter(proj(x2,z2,z0=z0), proj(y2,z2,z0=z0), 1, '-b',c=col,zorder= 3,s=0.1)
ax.set_title('Stereographic projection');

plt.show()
fig.savefig('stereo2.png', dpi=250)   # save the figure to file
plt.close(fig) 



# APARTADO ii)
print("\n" + Formato.BOLD + "Apartado ii)" + Formato.RESET)

"""
2-esfera proyectada - familia paramétrica
"""

t = 0.1
eps = 1e-16

xt = 2/(2*(1-t) + (1-z)*t + eps)*x
yt = 2/(2*(1-t) + (1-z)*t + eps)*y
zt = (-1)*t + z*(1-t)
x2t = 2/(2*(1-t) + (1-z2)*t + eps)*x2
y2t = 2/(2*(1-t) + (1-z2)*t + eps)*y2
z2t = (-1)*t + z2*(1-t)

fig = plt.figure(figsize=(6, 6))
#fig.subplots_adjust(hspace=0.4, wspace=0.2)
ax = plt.axes(projection='3d')


ax.set_xlim3d(-8, 8)
ax.set_ylim3d(-8, 8)
ax.plot_surface(xt, yt, zt, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.plot(x2t,y2t, z2t, '-b', c="white", zorder=3)

plt.show()
#fig.savefig('stereo2.png')   # save the figure to file
plt.close(fig) 

"""
HACEMOS LA ANIMACIÓN
"""

def animate(t):
    xt = 2/(2*(1-t) + (1-z)*t + eps)*x
    yt = 2/(2*(1-t) + (1-z)*t + eps)*y
    zt = (-1)*t + z*(1-t)
    x2t = 2/(2*(1-t) + (1-z2)*t + eps)*x2
    y2t = 2/(2*(1-t) + (1-z2)*t + eps)*y2
    z2t = (-1)*t + z2*(1-t)
    
    ax = plt.axes(projection='3d')
    ax.set_zlim3d(-1,1)
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1, alpha=0.5, cmap='viridis', edgecolor='none')
    ax.plot(x2t,y2t, z2t, '-b', c="white", zorder=3)
    return ax

def init():
    return animate(0)

animate(np.arange(0, 1, 0.1)[1])
plt.show()

fig = plt.figure(figsize=(6, 6))
ani = animation.FuncAnimation(fig, animate, np.arange(0,1, 0.05), init_func=init, interval=20)
#ani.save("ejemplo.htm", fps = 5) 
ani.save("ejemplo.gif", fps = 5)
