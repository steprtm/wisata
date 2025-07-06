import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Load Dataset
df = pd.read_csv("D:\\University stuff\\upb\\S6\\DataMining]\\UAS\\wisata_indonesia_final.csv")

#Koordinat
coordinates = df[['latitude', 'longitude']]

#Visualisasi Awal Sebaran Lokasi
plt.figure(figsize=(8, 6))
plt.scatter(coordinates['longitude'], coordinates['latitude'], c='blue', alpha=0.5, s=50)
plt.title('Sebaran Lokasi Tempat Wisata di Indonesia')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

#K-Means Clustering (5 Cluster)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(coordinates)

#Visualisasi Hasil Clustering
plt.figure(figsize=(8, 6))
plt.scatter(coordinates['longitude'], coordinates['latitude'], c=df['cluster'], cmap='tab10', s=50)
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], c='red', marker='X', s=200, label='Centroid')
plt.title('Hasil Pengelompokan Lokasi Tempat Wisata dengan K-Means (5 Cluster)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()

#Ringkasan Jumlah Tempat Wisata per Cluster dan Provinsi
summary = df.groupby(['cluster', 'provinsi']).size().reset_index(name='jumlah_wisata')
print(summary)

