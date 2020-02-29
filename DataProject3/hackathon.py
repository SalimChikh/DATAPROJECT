#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:45:59 2020

@author: salim
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
from sklearn.cluster import KMeans
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #para dibujos
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

os.chdir('/Users/salim/Desktop/EDEM/machine_learning/HACKATHON')
os.getcwd() #despues de cambiar de directorio verificamos si se ha hecho bien
data44 = pd.read_csv ("DATA_BANK11.csv", sep=',', decimal ='.')

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=[-1, 1]) 
data_rescaled = scaler.fit_transform(data)


#AJUST THE PCA WITH OUR DATA
pca = PCA().fit(data_rescaled)
#Grafica de la suma acumulativa de la varianza explicada
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Numero de componentes')
plt.ylabel('Varianza(%)')#para cada componente
plt.title('Big Bank Dataset variance')
plt.show()



# The optimate K 
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(data)
    distortions.append(kmeanModel.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion score')
plt.title('The Elbow Method showing the optimal k')
plt.show()

###############################################################


# K-means
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(data)
    distortions.append(kmeanModel.inertia_)

# Let´s plot our result
plt.figure(figsize=(6,6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion score')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Train the K-means model
kmeanModel = KMeans(n_clusters=2)
kmeanModel.fit(data)
y_kmeans = kmeanModel.predict(data)
centers = kmeanModel.cluster_centers_

kmeans_df = pd.DataFrame(centers)
kmeans_df.columns = data.columns

# We need data in two dimensions to plot the clusters.  We use PCA.   
X = StandardScaler().fit_transform(data)

pipe = Pipeline([('scaler', StandardScaler()), ('decomposition', PCA(n_components=2))])

X_transformed = pipe.fit_transform(X)
X_transformed.shape

pf_pca = pd.DataFrame(X_transformed, columns=['first_component', 'second_component'])

plt.scatter(pf_pca.first_component, pf_pca.second_component, alpha=0.9, c=y_kmeans, cmap="brg")
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')