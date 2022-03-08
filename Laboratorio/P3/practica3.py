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
Dado un dataframe
devuelve una rama del arbol de Huffman
"""
def huffman_branch(distr):
    states = np.array(distr['states'])
    probab = np.array(distr['probab'])
    state_new = np.array([''.join(states[[0,1]])])
    probab_new = np.array([np.sum(probab[[0,1]])])
    codigo = np.array([{states[0]: 0, states[1]: 1}])
    states =  np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
    probab =  np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
    distr = pd.DataFrame({'states': states, 'probab': probab, })
    distr = distr.sort_values(by='probab', ascending=True)
    distr.index=np.arange(0,len(states))
    branch = {'distr':distr, 'codigo':codigo}
    return(branch) 



# FORMATO
class Formato:
    BOLD = "\033[1m"
    RESET = "\033[0m"

# Aquí tenemos definido el sistema X de 1000 elementos de dos estados
# construido a partir de una muestra aleatoria entorno a unos centros:
centers = [[-0.5, 0.5], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
                            random_state=0)
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

print("Número óptimo de vecindades: ", max_k)

# Los clasificamos mediante el algoritmo KMeans
n_clusters = max_k
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
labels = kmeans.labels_

# Representamos el resultado con un plot
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(8,4))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=5)

plt.title('Fixed number of KMeans clusters: %d' % n_clusters)
plt.show()

#vor = Voronoi(X)
#voronoi_plot_2d(vor)

# APARTADO ii)
print("\n" + Formato.BOLD + "Apartado ii)" + Formato.RESET)

# euclidean
max_s = -1
for epsilon in np.arange(0.11,0.4,0.01):
    # Utilizamos el algoritmo de DBSCAN para mínimo 10 elementos
    db = DBSCAN(eps=epsilon, min_samples=10, metric='euclidean').fit(X)
    labels = db.labels_
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

print("Umbral de distancia óptimo: ", max_eps)

epsilon = max_eps
db = DBSCAN(eps=epsilon, min_samples=10, metric='euclidean').fit(X)
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
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=5)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=3)

plt.title('Estimated number of DBSCAN clusters: %d' % n_clusters_)
plt.show()

# manhattan
max_s = -1
for epsilon in np.arange(0.11,0.4,0.01):
    # Utilizamos el algoritmo de DBSCAN para mínimo 10 elementos
    db = DBSCAN(eps=epsilon, min_samples=10, metric='manhattan').fit(X)
    labels = db.labels_
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

print("Umbral de distancia óptimo: ", max_eps)

epsilon = max_eps
db = DBSCAN(eps=epsilon, min_samples=10, metric='euclidean').fit(X)
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
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=5)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=3)

plt.title('Estimated number of DBSCAN clusters: %d' % n_clusters_)
plt.show()




# APARTADO iii)
print("\n" + Formato.BOLD + "Apartado iii)" + Formato.RESET)
