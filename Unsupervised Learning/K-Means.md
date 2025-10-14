# K-Means

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Definisi](#definisi)
- [Cara Kerja](#cara-kerja)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Definisi

**K-Means** adalah 

## Cara Kerja


## Kelebihan
- **Sederhana dan mudah diimplementasikan**: Algoritma ini memiliki konsep dasar yang mudah dipahami dan langkah yang jelas sehingga penerapannya mudah dilakukan.
- **Cepat dan efisien**: Algoritma ini dapat memproses data dalam jumlah yang besar dalam waktu relatif singkat sehingga dapat digunakan pada data dengan jumlah yang besar.
- **Cocok untuk data dengan distribusi terpisah jelas**: Algoritma ini cocok untuk data yang klasternya berbentuk bulat, jaraknya saling berjauhan, dan ukurannya relatif seimbang karena K-Means mengelompokkan data berdasarkan jarak ke centroid, sehingga pola seperti itu memungkinkan pemisahan klaster yang optimal.

## Kekurangan
- **Harus menentukan jumlah klaster (k) di awal**: Algoritma ini memerlukan penentuan nilai k di awal, sedangkan jumlah klaster yang ideal sering kali belum diketahui, sehingga biasanya diperlukan metode pendukung seperti Elbow Method atau Silhouette Score untuk menentukannya.
- **Sensitif terhadap penentuan titik awal centroid**: Algoritma bisa berhenti pada solusi yang tidak optimal dan menghasilkan klaster yang tidak sesuai dengan pola asli data apabila penempatan centroid awal kurang tepat. 
- **Mudah terpengaruh oleh outlier**: Kemunculan data ekstrem dapat menggeser posisi centroid, sehingga hasil pengelompokan menjadi kurang akurat dan tidak mewakili distribusi data sebenarnya.

## Implementasi
```python
from sklearn.cluster import KMeans

# Data train
X_train = [
   [50, 1], 
   [60, 2], 
   [80, 3], 
   [100, 3], 
   [120, 3], 
   [150, 4]
]

# Data uji / test
X_test = [
   [130, 3],
   [160, 4]
]

# Inisialisasi dan melatih model K-Means
kmeans = KMeans(
    n_clusters=2,      # Jumlah klaster yang ingin dibentuk
    init="k-means++",  # Metode inisialisasi centroid
    n_init=10,         # Berapa kali algoritma dicoba untuk hasil terbaik
    max_iter=300,      # Iterasi maksimum
)

kmeans.fit(X_train)

# Menampilkan centroid hasil pelatihan
print(kmeans.cluster_centers_)

# Menentukan klaster dari data uji
y_pred = kmeans.predict(X_test)
print(y_pred)
```

## Referensi
[Scikit-Learn - KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
[Towards AI - What are the advantages and disadvantages of K-Means Clustering?](https://towardsai.net/p/l/what-are-the-advantages-and-disadvantages-of-k-means-clustering)