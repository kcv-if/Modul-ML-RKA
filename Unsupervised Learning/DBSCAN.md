# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Definisi](#definisi)
- [Cara Kerja](#cara-kerja)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Definisi

![dbscan](https://ml-explained.com/articles/dbscan-explained/dbscan.gif)

DBSCAN adalah algoritma clustering berbasis densitas yang digunakan untuk mengelompokkan titik atau data yang saling berdekatan dalam ruang fitur, serta mengidentifikasi titik atau data yang berada di area dengan kepadatan rendah sebagai noise (outlier). 

Algoritma ini tidak membutuhkan penentuan jumlah cluster sejak awal dan mampu menemukan kelompok data dengan bentuk yang beragam (tidak hanya bulat) karena bergantung pada tingkat kerapatan titik dalam ruang data.

---

## Cara Kerja

DBSCAN bekerja dengan prinsip bahwa cluster adalah kumpulan titik-titik yang memiliki kepadatan tinggi, sedangkan titik yang berada pada area kepadatan rendah dianggap sebagai noise (outlier).

Berikut ini adalah langkah kerja dari DBSCAN:

1. Menentukan parameter awal\
Dua parameter utama harus ditetapkan adalah sebagai berikut.
    - epsilon (ε): jarak maksimum untuk mencari tetangga di sekitar suatu titik.
    - minPts: jumlah minimal titik dalam radius ε agar sebuah titik dianggap memiliki kepadatan yang cukup.

2. Mengidentifikasi jenis titik\
![dbscan-parameter](https://miro.medium.com/v2/resize:fit:1400/1*arv3b3Um_Opu_zOECGwt6w.png)
Berdasarkan kombinasi ε dan minPts, setiap titik diklasifikasikan menjadi:
    - Core point: titik yang memiliki jumlah tetangga ≥ minPts dalam jarak ε.
    - Border point: titik yang berada di sekitar core point tetapi memiliki tetangga < minPts.
    - Noise point: titik yang tidak termasuk ke dalam cluster manapun dan tidak memenuhi kriteria sebagai core maupun border.

3. Membentuk Cluster
    - Algoritma memilih salah satu core point secara acak untuk memulai cluster.
    - Semua core point yang berada dalam jarak ε dari titik tersebut ditambahkan ke cluster.
    - Setiap core point baru yang tergabung digunakan untuk mencari core point lain di sekitarnya. Proses ini berlanjut hingga tidak ada lagi core point baru yang dapat ditambahkan (density-reachability).

4. Menambahkan Border Point
    - Setelah seluruh core point dalam satu cluster ditemukan, border point yang berada dalam jarak ε dari core point akan dimasukkan ke cluster.
    - Perlu diperhatikan bahwa border point hanya dapat bergabung ke cluster tetapi tidak dapat memperluas cluster lebih jauh.

5. Mengulangi Proses
    - Jika masih ada core point yang belum masuk cluster manapun, proses akan diulang untuk membentuk cluster baru.
    - Titik yang tidak terhubung dengan core point manapun akan menjadi noise.

---

## Kelebihan

- **Tidak perlu menentukan jumlah klaster di awal** seperti K-Means.
- **Mendeteksi outlier secara alami** , titik yang tidak termasuk cluster dikenali sebagai *noise*.
- **Mampu mengenali bentuk klaster arbitrer (non-spherical)**.
- **Stabil terhadap urutan data** karena hasil tidak tergantung pada inisialisasi acak.

---

## Kekurangan

- **Sulit digunakan untuk data berdimensi tinggi (curse of dimensionality)**.
- **Kurang efisien untuk dataset besar** karena perhitungan jarak antar semua titik bisa mahal secara komputasi.

---

## Implementasi

```python
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score

# 1) Generate Non-Spherical Sample Data
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

# 2) Visualize the Data
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], s=30, c='gray')
plt.title("Sample Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 3) Apply DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
y_pred = dbscan.fit_predict(X)

# 4) Visualize Clusters
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='rainbow', s=30)
plt.title("DBSCAN Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 5) Count Clusters and Noise
n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
n_noise = list(y_pred).count(-1)
print(f"Number of clusters formed: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# 6) Evaluate Clustering performance
sc = metrics.silhouette_score(X, y_pred)
print("Silhouette Coefficient: %0.2f" % sc)
```

<img width="556" height="468" alt="image" src="https://github.com/user-attachments/assets/6088273b-d2cb-4978-a134-a2660c2c0541" />


<img width="556" height="468" alt="image" src="https://github.com/user-attachments/assets/563213f9-73ca-4691-8991-455ab46ae6b7" />

<br>
- Number of clusters formed: 2 <br>
- Number of noise points: 0 <br>
- Silhouette Coefficient: 0.32 <br>

## Reference
https://www.geeksforgeeks.org/machine-learning/dbscan-clustering-in-ml-density-based-clustering/
