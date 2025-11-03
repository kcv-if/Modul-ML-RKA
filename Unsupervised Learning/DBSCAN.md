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

---

## Cara Kerja

---

## Kelebihan

- **Tidak perlu menentukan jumlah klaster di awal** seperti K-Means.
- **Mendeteksi outlier secara alami** — titik yang tidak termasuk cluster dikenali sebagai *noise*.
- **Mampu mengenali bentuk klaster arbitrer (non-spherical)**.
- **Stabil terhadap urutan data** karena hasil tidak tergantung pada inisialisasi acak.

---

## Kekurangan

- **Sensitif terhadap parameter ε dan MinPts** — hasil bisa sangat berbeda jika nilainya tidak sesuai.
- **Sulit digunakan untuk data berdimensi tinggi (curse of dimensionality)** — jarak antar titik menjadi kurang bermakna.
- **Kurang efisien untuk dataset besar** karena perhitungan jarak antar semua titik bisa mahal secara komputasi.

---

## Implementasi

```python
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 1) Generate Non-Spherical Sample Data
X, y = make_moons(n_samples=300, noise=0.12, random_state=42)

# 2) Visualize the Data
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], s=30, c='gray')
plt.title("Sample Data (Two Moons)")
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
```

<img width="567" height="468" alt="image" src="https://github.com/user-attachments/assets/7cae7d26-868e-44c7-9a7b-5ea41b789ef9" />

<img width="567" height="468" alt="image" src="https://github.com/user-attachments/assets/59e8d0b2-687f-4656-ba28-91148d02122d" />
Number of clusters formed: 1
Number of noise points: 4

