# K-Means

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Definisi](#definisi)
- [Cara Kerja](#cara-kerja)
- [Menghitung Jarak](#menghitung-jarak)
- [Menentukan jumlah K](#menentukan-jumlah-k)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Definisi

**K-Means Clustering** adalah algoritma Unsupervised Machine Learning yang mengelompokkan dataset tanpa label ke dalam beberapa klaster. Algoritma ini digunakan untuk mengorganisasi data ke dalam kelompok berdasarkan kemiripan antar data.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/16c76535-763c-40b7-9c7c-b819b1cdf81b" />

## Cara Kerja
1. Tentukan Jumlah K
2. Tentukan centroid awal sebanyak K
3. Hitung [Jarak](#menghitung-jarak) semua data ke centroid
4. Kelompokkan data dengan centroid terdekat
5. Temukan centroid baru dengan mencari Rata rata titik yang ada di dalam suatu cluster
6. Ulangi step 3 - 5 hingga tidak ada perubahan lagi

## Menghitung Jarak
Nah karena kita akan mencari kemiripan antar data dengan jaraknya, kita harus mengetahui cara untuk menghitung jaraknya. Ada beberapa rumus jarak yang umum untuk dipakai:

<img width="964" height="403" alt="image" src="https://github.com/user-attachments/assets/ea00eb41-80ba-485c-b31a-d8046418c9a6" />

## Menentukan Jumlah K
Salah satu tantangan utama dalam menggunakan K-Means adalah menentukan jumlah klaster (K) yang paling sesuai untuk data kita. Memilih K yang terlalu kecil dapat menggabungkan kelompok data yang seharusnya terpisah, sedangkan memilih K yang terlalu besar akan memecah kelompok data yang seharusnya menyatu.

1. Elbow Method

   Metode ini adalah yang paling umum digunakan karena intuisinya yang sederhana. Idenya adalah menjalankan algoritma K-Means untuk beberapa nilai K yang berbeda (misalnya, dari 1 hingga 10) dan menghitung Within-Cluster Sum of Squares (WCSS) atau Inertia untuk setiap nilai K.

   <img width="954" height="385" alt="image" src="https://github.com/user-attachments/assets/6cd7b8ee-8cc3-4192-b568-67fffb53e79f" />

   **Rumus WCSS**:
   
   <img width="687" height="466" alt="image" src="https://github.com/user-attachments/assets/7e36246d-5b8b-4cb7-b4e4-3d17dfe2ec89" />

   WCSS mengukur seberapa padat titik-titik dalam setiap cluster. Semakin kecil nilai WCSS, semakin baik kualitas clustering
   
2. Silhouette Score

   Silhouette Score mengukur seberapa baik setiap titik dikelompokkan ke dalam clusternya

   <img width="829" height="349" alt="image" src="https://github.com/user-attachments/assets/59483d90-08de-41d6-9266-13d394aac34e" />

3. Elbow Method vs Silhoutte Score

   <img width="809" height="484" alt="image" src="https://github.com/user-attachments/assets/7de107d1-e88c-4631-9b43-b46c09bfa50d" />

## Kelebihan
- **Sederhana dan mudah diimplementasikan**: Algoritma ini memiliki konsep dasar yang mudah dipahami dan langkah yang jelas sehingga penerapannya mudah dilakukan.
- **Cepat dan efisien**: Algoritma ini dapat memproses data dalam jumlah yang besar dalam waktu relatif singkat sehingga dapat digunakan pada data dengan jumlah yang besar.
- **Cocok untuk data dengan distribusi terpisah jelas**: Algoritma ini cocok untuk data yang klasternya berbentuk bulat, jaraknya saling berjauhan, dan ukurannya relatif seimbang karena K-Means mengelompokkan data berdasarkan jarak ke centroid, sehingga pola seperti itu memungkinkan pemisahan klaster yang optimal.

## Kekurangan
- **Harus menentukan jumlah klaster (k) di awal**: Algoritma ini memerlukan penentuan nilai k di awal, sedangkan jumlah klaster yang ideal sering kali belum diketahui, sehingga biasanya diperlukan metode pendukung seperti Elbow Method atau Silhouette Score untuk menentukannya.
- **Sensitif terhadap penentuan titik awal centroid**: Algoritma bisa berhenti pada solusi yang tidak optimal dan menghasilkan klaster yang tidak sesuai dengan pola asli data apabila penempatan centroid awal kurang tepat.

  <img width="844" height="478" alt="image" src="https://github.com/user-attachments/assets/4edd5c8b-ee6b-413b-9d5a-5d430d4e9644" />

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
