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
import matplotlib.pyplot as plt
from netCDF4 import Dataset
#from scipy.io import netcdf as nc
from sklearn.decomposition import PCA

#workpath = "/Users/martin/Documents/Estudios/Matemáticas e Ingeniería Informática/2021-2022/GCom/Git/Computational-Geometry/Laboratorio/P4"

"""
Dada una lista con longitudes [0,2pi]
devuelve la lista con longitud [-pi,pi]
"""
def lon_normal_ref(list):
    # La lista debe ser monótona
    return np.concatenate([list[72:]-360, list[0:72]])
    
"""
Dada una matriz (latitud,longitud) con valores en el dominio de longitud [0,2pi]
devuelve la lista con valores en el dominio de longitud [-pi,pi]
"""
def lons_normal_ref(matrix):
    rows, _ = matrix.shape
    return [np.concatenate([matrix[i][72:], matrix[i][0:72]]) for i in range(rows)]



# APARTADO i)

f = Dataset("hgt.2021.nc", "r", format="NETCDF4")
time = f.variables['time'][:].copy()
lons = lon_normal_ref(f.variables['lon'][:].copy())
lats = f.variables['lat'][:].copy()
level = f.variables['level'][:].copy()
hgt = f.variables['hgt'][:].copy()
#scale = f.variables['hgt'].scale_factor
#offset = f.variables['hgt'].add_offset
f.close()

"""
Distribución espacial de la temperatura en el nivel de 500hPa, para el primer día
"""
plt.contour(lons, lats, lons_normal_ref(hgt[0, level==500., :, :].reshape(len(lats),len(lons))))
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
