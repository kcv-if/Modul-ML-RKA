# Hierarchical Clustering

## Daftar Isi

-   [Daftar Isi](#daftar-isi)
-   [Definisi](#definisi)
-   [Cara Kerja](#cara-kerja)
-   [Kelebihan](#kelebihan)
-   [Kekurangan](#kekurangan)
-   [Implementasi](#implementasi)
-   [Referensi](#referensi)

## Definisi

**Hierarchical Clustering** adalah algoritma regresi **Unsupervised Learning** untuk mengelompokkan data ke dalam struktur berbentuk hierarki, biasanya divisualisasikan dalam bentuk dendrogram (diagram pohon).

<img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_agglomerative_dendrogram_001.png" />

Terdapat dua metode untuk melakukan hierarchical clustering, yakni:

### 1) Agglomerative Hierarchical Clustering

Metode ini menganggap semua data sebagai klaster tinggal yang akan digabungkan berdasarkan kemiripannya.

Saat memperbarui jarak antara klaster, terdapat berbagai macam stategi yang dinamakan _linkage method_, yakni:

-   Single

    Menggunakan jarak minimum antara titik dalam klaster terhadap klaster lain.

-   Complete

    Menggunakan jarak maksimum antara titik dalam klaster terhadap klaster lain.

-   Average

    Menggunaan jarak rata-rata antara semua titik dalam klaster terhadap klaster lain.

-   Ward

    Menggunakan varians intra-klaster antara sebuah klaster dengan klaster lain.

### 2) Divisive Hierarchical Clustering

Metode ini menganggap semua data berasal dari satu klaster yang sama lalu dipisah berdasarkan perbedaannya.

## Cara Kerja

1. Hitung jarak antara semua pasangan data.
2. Gabungkan dua klaster yang paling mirip (_agglomerative_) / Pisah klaster menjadi dua berdasarkan perbedaan paling jauh (_divisive_).
3. Perbarui matriks jarak antar klaster.
4. Ulangi sampai tersisa satu klaster besar.

## Kelebihan

-   **Tidak perlu menentukan jumlah klaster di awal**: mampu merepresentasikan hubungan yang lebih kompleks.
-   **Interpretasi mudah**: hanya perlu menambahkan transformasi fitur.
-   **Deterministik**: tidak akan menghasilkan hasil acak.

## Kekurangan

- **Berat untuk dikomputasi**: Kompleksitas waktu dan memori dapat mencapai $ O(n^2) $
- **Sensitif terhadap noise dan outlier**: outlier dan noise dapat menghasilkan penggabungan / pemisahan klaster yang salah.
- **Bergantung pada linkage**: hasil beragam berdasarkan linkage.

## Implementasi

Berikut adalah cara mengimplementasikan Agglomerative Clustering dengan library `scikit-learn`.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# 1️) Generate Sample Data
X, y = make_blobs(
    n_samples=200,        # total points
    centers=4,            # number of real clusters
    cluster_std=1.2,      # how spread out each cluster is
    random_state=42
)

# 2️) Visualize the Data
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], s=40, c='gray')
plt.title("Sample Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 3) Create Dendrogram
plt.figure(figsize=(8, 5))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# 4) Agglomerative Clustering
hc = AgglomerativeClustering(
    n_clusters=4,
    metric='euclidean',
    linkage='ward'
)
y_pred = hc.fit_predict(X)

# 5️) Visualize Clusters
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='rainbow')
plt.title("Hierarchical Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

## Referensi

-   [Scikit-Learn - Hierarchical Clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
-   [Scikit-Learn - Agglomerative Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
