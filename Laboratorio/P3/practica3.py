"""
PRÁCTICA 3: DIAGRAMA DE VORONOI Y CLUSTERING
Belén Sánchez Centeno
Martín Fernández de Diego
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial import Voronoi, voronoi_plot_2d

"""
Dado un conjunto de puntos X
muestra la gráfica de los coeficientes de Silhouette para cada número de clusters
devuelve el número óptimo de clusters asociado al mayor coeficiente de Silhouette
"""
def plot_silhouette_kmeans(X):
    # Mostramos los coeficientes de Silhouette para cada k
    # y obtenemos el k asociado al mayor coeficiente de todos
    max_s = -1
    for k in range(2,16):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        labels = kmeans.labels_
        silhouette = metrics.silhouette_score(X, labels)
        # Decidimos computacinalmente el número óptimo de clusters
        if max_s < silhouette:
            max_s = silhouette
            max_k = k
        plt.plot(k, silhouette, 'o')
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Coeficiente de Silhouette (s)")
    plt.show()
    
    return max_k

"""
Dado un conjunto de puntos X y el número de vecindades óptimo n_clusters
muestra los clusters estimados por el algoritmo KMeans
devuelve la instancia de KMeans para el apartado iii)
"""
def plot_clusters_kmeans(X,n_clusters):
    # Tomamos el número de vecindades óptimo devuelto por el coeficiente de Silhouette
    # y volvemos a ejecutar KMeans para mostrar los clusters por colores
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # Mantenemos la misma proporción al resto en el tamaño de la gráfica
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    
    # Mostramos el teselado de Voronoi
    vor = Voronoi(centers)
    voronoi_plot_2d(vor,ax=ax)
    
    # Representamos el resultado con un plot
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=5)
    # Mostramos los centros
    plt.plot(centers[:,0],centers[:,1],'o', markersize=12, markerfacecolor="red")
    for i in range(len(centers)):
        plt.text(centers[i,0],centers[i,1],str(i),color='orange',fontsize=16,fontweight='black')
    # Configuramos los atributos de la gráfica con sus límites
    plt.title('Número de clusters fijado por KMeans: %d' % n_clusters)
    plt.xlim([np.min(X[:,0])-0.25,np.max(X[:,0])+0.25])
    plt.ylim([np.min(X[:,1])-0.25,np.max(X[:,1])+0.25])
    plt.show()
    
    return kmeans

"""
Dado un conjunto de puntos X, el tipo de métrica y la sensibilidad de búsqueda del umbral de distancia
muestra la gráfica de los coeficientes de Silhouette para cada umbral de distancia
devuelve el umbral de distancia óptimo asociado al mayor coeficiente de Silhouette
"""
def plot_silhouette_dbscan(X,metric,step=0.01):
    # Mostramos los coeficientes de Silhouette para cada épsilon
    # y obtenemos el épsilon asociado al mayor coeficiente de todos
    max_s = -1
    for epsilon in np.arange(0.11,0.4,step):
        # Utilizamos el algoritmo de DBSCAN para mínimo 10 elementos
        db = DBSCAN(eps=epsilon, min_samples=10, metric=metric).fit(X)
        labels = db.labels_
        # Aseguramos el valor de Silhouette si el número de clusters es 1
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        silhouette = metrics.silhouette_score(X, labels) if n_clusters_ != 1 else -1
        # Decidimos computacinalmente el número óptimo de clusters
        if max_s < silhouette:
            max_s = silhouette
            max_eps = epsilon
        plt.plot(epsilon, silhouette, 'o')
    plt.xlabel("Umbral de distancia (eps)")
    plt.ylabel("Coeficiente de Silhouette (s)")
    plt.show()
    
    return max_eps

"""
Dado un conjunto de puntos X, el tipo de métrica y el umbral de distancia
muestra los clusters estimados por el algoritmo DBSCAN con la métrica dada
"""
def plot_clusters_dbscan(X,metric,epsilon):
    # Tomamos el épsilon óptimo devuelto por el coeficiente de Silhouette
    # y volvemos a ejecutar DBSCAN para mostrar los clusters por colores
    db = DBSCAN(eps=epsilon, min_samples=10, metric=metric).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print("Número óptimo de vecindades: ", n_clusters_)
    
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    plt.figure(figsize=(8,4))
    for k, col in zip(unique_labels, colors):
        # Black used for noise.
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=5)
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=3)
    plt.title('Número de clusters estimado por DBSCAN: %d' % n_clusters_)
    plt.show()

def distancia_euclidea(punto,centro):
    return np.sqrt((punto[0]-centro[0])**2 + (punto[1]-centro[1])**2)

def distancia_manhattan(punto,centro):
    return abs(punto[0]-centro[0]) + abs(punto[1]-centro[1])

# FORMATO
class Formato:
    BOLD = "\033[1m"
    RESET = "\033[0m"

# Aquí tenemos definido el sistema X de 1000 elementos de dos estados
# construido a partir de una muestra aleatoria entorno a unos centros:
centers = [[-0.5, 0.5], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4, random_state=0)
#Si quisieramos estandarizar los valores del sistema, haríamos:
#from sklearn.preprocessing import StandardScaler
#X = StandardScaler().fit_transform(X)  

#Envolvente convexa, envoltura convexa o cápsula convexa 
hull = ConvexHull(X)
convex_hull_plot_2d(hull)

plt.plot(X[:,0],X[:,1],'ro', markersize=1)
plt.show()

# APARTADO i)
print("\n" + Formato.BOLD + "Apartado i)" + Formato.RESET)

max_k = plot_silhouette_kmeans(X)
print("Número óptimo de vecindades: ", max_k)
kmeans = plot_clusters_kmeans(X,max_k)

# APARTADO ii)
print("\n" + Formato.BOLD + "Apartado ii)" + Formato.RESET)

# Euclidean
euclidean_metric = 'euclidean'
euclidean_max_eps = plot_silhouette_dbscan(X,euclidean_metric)
print("Umbral de distancia euclidiana óptimo: ",euclidean_max_eps)
plot_clusters_dbscan(X,euclidean_metric,euclidean_max_eps)

# Manhattan
manhattan_metric = 'manhattan'
manhattan_max_eps = plot_silhouette_dbscan(X,manhattan_metric)
print("Umbral de distancia manhattan óptimo: ",manhattan_max_eps)
plot_clusters_dbscan(X,manhattan_metric,manhattan_max_eps)

# APARTADO iii)
print("\n" + Formato.BOLD + "Apartado iii)" + Formato.RESET)

centers = kmeans.cluster_centers_

punto1 = [0,0]
distancias_euclideas = [distancia_euclidea(punto1,centers[i]) for i in range(len(centers))]
cluster1 = distancias_euclideas.index(min(distancias_euclideas))
print("El punto ",punto1,"se encuentra en el cluster ",cluster1,"según la distancia euclídea")
distancias_manhattan = [distancia_manhattan(punto1,centers[i]) for i in range(len(centers))]
cluster1 = distancias_manhattan.index(min(distancias_manhattan))
print("El punto ",punto1,"se encuentra en el cluster ",cluster1,"según la distancia manhattan")
print("Comprobación: ",kmeans.predict([punto1])[0])

punto2 = [0,-1]
distancias_euclideas = [distancia_euclidea(punto2,centers[i]) for i in range(len(centers))]
cluster2 = distancias_euclideas.index(min(distancias_euclideas))
print("El punto ",punto2,"se encuentra en el cluster ",cluster2,"según la distancia euclídea")
distancias_manhattan = [distancia_manhattan(punto2,centers[i]) for i in range(len(centers))]
cluster2 = distancias_manhattan.index(min(distancias_manhattan))
print("El punto ",punto1,"se encuentra en el cluster ",cluster2,"según la distancia manhattan")
print("Comprobación: ",kmeans.predict([punto2])[0])

