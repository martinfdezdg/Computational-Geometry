"""
DAVID SEIJAS PEREZ
PRACTICA 4
"""

import datetime as dt  # Python standard library datetime  module
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from sklearn.decomposition import PCA
import math

workpath = ""

f = Dataset(workpath + "air.2021.nc", "r", format="NETCDF4")
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
level = f.variables['level'][:].copy()
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()
air21 = f.variables['air'][:].copy()
air_units = f.variables['air'].units
f.close()

f = Dataset(workpath + "air.2022.nc", "r", format="NETCDF4")
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
air22 = f.variables['air'][:].copy()
f.close()

f = Dataset(workpath + "hgt.2021.nc", "r", format="NETCDF4")
time21 = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
hgt21 = f.variables['hgt'][:].copy()
hgt_units = f.variables['hgt'].units
f.close()

f = Dataset(workpath + "hgt.2022.nc", "r", format="NETCDF4")
time22 = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
hgt22 = f.variables['hgt'][:].copy()
f.close()



def apartado1():
    hgt21b = hgt21[:,level==500.,:,:].reshape(len(time21),len(lats)*len(lons))  #365x10512
    n_components = 4
    X = hgt21b
    Y = hgt21b.transpose()
    
    pca = PCA(n_components=n_components)
    Element_pca0 = pca.fit_transform(Y)
    Element_pca0 = Element_pca0.transpose(1,0).reshape(n_components,len(lats),len(lons))
    
    pesos = pca.components_
    print(pesos[0])
    print(pca.explained_variance_ratio_)
    
    pca.fit(X)
    pesos = pca.components_
    print(pesos[0])
    print(pca.explained_variance_ratio_)
    
    #Representacion espacial de las pca en (x, y)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i)
        ax.text(0.5, 90, 'PCA-'+str(i),
               fontsize=18, ha='center')
        plt.contour(lons, lats, Element_pca0[i-1,:,:])
    plt.show()
    
    
def dist_euclidea(a0, dia_aux):
    d = 0
    for i in range(len(a0[0])): #latitud
        for j in range(len(a0[0][0])): #longitud
            #Para cada elemento aplico su peso w_k segun su p_k
            #Al hacer a0[level == 500.] me quedo con la fila de level = 500, pero sigo teniendo matriz de dim 3 aunque en la dim 1 solo tengo una "fila"
            d += 0.5*((a0[level == 500.][0][i][j] - dia_aux[level == 500.][0][i][j])**2)
            d += 0.5*((a0[level == 1000.][0][i][j] - dia_aux[level == 1000.][0][i][j])**2)
    return math.sqrt(d)
    
    
def apartado2():
    #Subsistema de S
    hgt21c = hgt21[:,:,:,np.logical_or(340 < lons, lons < 20)]
    hgt21c = hgt21c[:,:,np.logical_and(30 < lats, lats < 50),:]

    dt_time22 = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) for t in time22]
    dia_a0 = dt.date(2022, 1, 11)
    index_a0 = dt_time22.index(dia_a0)
    
    #Dia a0 a estudiar
    a0 = hgt22[index_a0,:,:,:]
    a0 = a0[:,:,np.logical_or(340 < lons, lons < 20)]
    a0 = a0[:,np.logical_and(30 < lats, lats < 50),:]
    
    #Sacamos las distancias euclideas de cada dia con el a0 a estudiar y nos quedamos con los 4 más pequeñas
    n_dias = 4
    distancias = [(i, dist_euclidea(a0, hgt21c[i])) for i in range(len(hgt21c))]
    distancias = sorted(distancias, key=lambda x : x[1])
    dist_analogos = distancias[0:n_dias]
    print("Análogos a a0 según variable Z con su distancia:")
    print(dist_analogos)
    
    #Vemos cuales son los 4 días análogos (menor distancia euclídea) a a0
    dt_time21 = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) for t in time21]
    analogos = [dt_time21[dist_analogos[i][0]] for i in range(n_dias)]
    print("Días análogos a a0 según variable Z:")
    print(analogos)
    
    #Hallamos la media de la variable de estado T (para p_k=1000) de los analogos
    media = 0
    for i in range(n_dias):
        media = np.add(media, air21[dist_analogos[i][0]][level == 1000])
    media = media*(1/n_dias)
    print(media)
    
    #Hallamos el error absoluto medio de esta variable
    a0 = (air22[index_a0][level == 1000])*(-1) #variable T para dia a0
    eam = (np.sum(abs(np.add(media, a0))))/(73*144) #formula del eam
    print("Error absoluto medio de la temperatura:")
    print(eam)

print("------ APARTADO 1 ------")
apartado1()
print("------ APARTADO 2 ------")
apartado2()
