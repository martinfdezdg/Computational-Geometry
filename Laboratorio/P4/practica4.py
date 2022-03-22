"""
PRÁCTICA 4: PCA Y ANALOGÍA
Belén Sánchez Centeno
Martín Fernández de Diego
"""

"""
Referencias:
    
    Fuente primaria del reanálisis
    https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.pressure.html
    
    Altura geopotencial en niveles de presión
    https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=59&tid=97457&vid=1498
    
    Temperatura en niveles de presión:
    https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=59&tid=97457&vid=4237

    Temperatura en niveles de superficie:
    https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=59&tid=97457&vid=1497
"""

import datetime as dt  # Python standard library datetime  module
import numpy as np
import math
import matplotlib.pyplot as plt
from netCDF4 import Dataset
#from scipy.io import netcdf as nc
from sklearn.decomposition import PCA

#workpath = "/Users/martin/Documents/Estudios/Matemáticas e Ingeniería Informática/2021-2022/GCom/Git/Computational-Geometry/Laboratorio/P4"

f = Dataset("air.2021.nc", "r", format="NETCDF4")
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
level = f.variables['level'][:].copy()
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()
air21 = f.variables['air'][:].copy()
air_units = f.variables['air'].units
#air_scale = f.variables['air'].scale_factor
#air_offset = f.variables['air'].add_offset
f.close()

f = Dataset("air.2022.nc", "r", format="NETCDF4")
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
air22 = f.variables['air'][:].copy()
f.close()


f = Dataset("hgt.2021.nc", "r", format="NETCDF4")
time21 = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
hgt21 = f.variables['hgt'][:].copy()
hgt_units = f.variables['hgt'].units
#hgt_scale = f.variables['hgt'].scale_factor
#hgt_offset = f.variables['hgt'].add_offset
f.close()


#f = nc.netcdf_file(workpath + "/" + files[0], 'r')
f = Dataset("hgt.2022.nc", "r", format="NETCDF4")
time22 = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
hgt22 = f.variables['hgt'][:].copy()
f.close()



# APARTADO i)

"""
Dada una matriz (latitud,longitud) con valores en el dominio de longitud [0,2pi]
devuelve la lista con valores en el dominio de longitud [-pi,pi]
"""
def lons_normal_ref(matrix):
    rows, _ = matrix.shape
    return [np.concatenate([matrix[i][72:], matrix[i][0:72]]) for i in range(rows)]

hgt21b = hgt21[:,level==500.,:,:].reshape(len(time21),len(lats)*len(lons))

n_components = 4

X = hgt21b
Y = hgt21b.transpose()
pca = PCA(n_components=n_components)

# Interpretar el siguiente resultado
pca.fit(X)
print(pca.explained_variance_ratio_)
out = pca.singular_values_
"""
Salida: [0.4724878  0.06072688 0.03592642 0.02815213]
La primera componente principal explica el 47% de la varianza del dataset.
En total, las cuatro primeras componentes principales explican entorno al 60% del dataset al entrenar X.
"""

# Interpretar el siguiente resultado
pca.fit(Y)
print(pca.explained_variance_ratio_)
out = pca.singular_values_
"""
Salida: [0.8877314  0.05177603 0.00543984 0.00357636]
La primera componente principal explica más del 88% de la varianza del dataset.
En total, las cuatro primeras componentes principales explican más del 94% del dataset al entrenar Y = tr(X).
"""

"""
Dado que es capaz de explicar más varianza con menos componentes,
se entrena el análisis de componentes principales —PCA— con Y = tr(X)
"""
Element_pca0 = pca.fit_transform(Y)
Element_pca0 = Element_pca0.transpose(1,0).reshape(n_components,len(lats),len(lons))

# Ejercicio de la práctica - Opción 1
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),
           fontsize=18, ha='center')
    plt.contour(lons, lats, lons_normal_ref(Element_pca0[i-1,:,:])) # Movemos España al medio
plt.show()


"""
Distribución espacial de la temperatura en el nivel de 500hPa, para el primer día
"""
plt.contour(lons, lats, lons_normal_ref(hgt21[0, level==500., :, :].reshape(len(lats), len(lons))))
plt.show()

"""
hgt21b = hgt21[:,level==500.,:,:].reshape(len(time21),len(lats)*len(lons))
air21b = air21[:,level==500.,:,:].reshape(len(time21),len(lats)*len(lons))

#air3 = air2.reshape(len(time),len(lats),len(lons))

n_components=4

X = hgt21b
Y = hgt21b.transpose()
pca = PCA(n_components=n_components)
Element_pca0 = pca.fit_transform(Y)
Element_pca0 = Element_pca0.transpose(1,0).reshape(n_components,len(lats),len(lons))

#Interpretar el siguiente resultado
pca.fit(Y)
print(pca.explained_variance_ratio_)
out = pca.singular_values_

#Interpretar el siguiente resultado
pca.fit(X)
print(pca.explained_variance_ratio_)
out = pca.singular_values_

#X = air21b
X = hgt21b
State_pca = pca.fit_transform(X)
Element_pca1 = pca.fit(X).components_
Element_pca1 = Element_pca1.reshape(n_components,len(lats),len(lons))
print(pca.explained_variance_ratio_)

#Ejercicio de la práctica - Opción 1
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),
           fontsize=18, ha='center')
    plt.contour(lons, lats, Element_pca0[i-1,:,:])
plt.show()

#Ejercicio de la práctica - Opción 2
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),
           fontsize=18, ha='center')
    plt.contour(lons, lats, Element_pca1[i-1,:,:])
plt.show()

#time_idx = 237
# Python and the reanalaysis are slightly off in time so this fixes that problem
# offset = dt.timedelta(hours=0)
# List of all times in the file as datetime objects
dt_time21 = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) for t in time21]
np.min(dt_time21)
np.max(dt_time21)

dt_time22 = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) for t in time22]
np.min(dt_time22)
np.max(dt_time22)
dia0 = dt.date(2022, 1, 11)
"""

# APARTADO ii)

# subsistema

def dist_euclidea(dia0, dia):
    dist = 0
    for lat in range(len(dia0[0])):
        for lon in range(len(dia0[0][0])):
            dia0_500 = 0.5*dia0[level==500.][0][lat][lon]
            dia0_1000 = 0.5*dia0[level==1000.][0][lat][lon]
            dia_500 = 0.5*dia[level==500.][0][lat][lon]         
            dia_1000 = 0.5*dia[level==1000.][0][lat][lon]
            dist += (dia0_500 - dia_500)**2 + (dia0_1000 - dia_1000)**2
    dist = math.sqrt(dist)
    return dist

