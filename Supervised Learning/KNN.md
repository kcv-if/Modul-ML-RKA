# K-Nearest Neighbors (KNN)

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Definisi](#definisi)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Definisi

KNN (K-Nearest Neighbors) merupakan sebuah algoritma supervised learning yang memberikan prediksi dengan melihat kelas mayoritas (terbanyak) yang mengelilingi data yang ingin kita prediksi.

<img src="https://machinelearningknowledge.ai/wp-content/uploads/2018/08/Value-of-K.gif">

KNN mengambil sebanyak $k$ (hyperparameter) tetangga terdekat, kemudian menentukan label kelas untuk data baru dengan melakukan voting mayoritas. Untuk menhitung jarak antara dua titik, misal $x$ dengan $x'$, terdapat berbagai metrik seperti berikut.

**Manhattan (L1 Distance)**

$$d(x, x') = \sum_{i = 1}^n \left| x_i - x_i' \right|$$

**Euclidean (L2 Distance)**

$$d(x, x') = \sqrt{ \sum_{i = 1} ^ n ( x_i - x_i' ) ^ 2 }$$

**Minkowski (Generalisasi dari L1 & L2)**

$$d(x, x') = \left[ \sum_{i = 1} ^ n ( x_i - x_i' ) ^ p \right] ^ { \frac{1}{p} }$$

Diberikan fitur latih $X \in \mathbb{R}^{n \times d}$, vektor label $Y \in \mathbb{Z} ^ n$ dan data uji $x \in \mathbb{R} ^ {m \times d}$ berikut cara menerapkan KNN:
1. Tentukan nilai $k$.
2. Tentukan metrik jarak yang akan digunakan.
3. Untuk setiap data uji $x$ pada $X$, hitung jarak antara $x$ dan seluruh data latih $X$.
4. Pilih $k$ tetangga terdekat berdasarkan hasil perhitungan jarak.
5. Prediksi label $y$ untuk $x$ berdasarkan voting mayoritas dari label $Y$ milik $k$ tetangga terdekat.



## Kelebihan

- Implementasi mudah

	KNN sangat sederhana untuk diimplementasikan karena tidak memerlukan proses pelatihan model yang kompleks. Algoritma ini hanya perlu menyimpan data latih dan perhitungan jarak saat prediksi.
	
- Tidak mengandung parameter

	Ketika membuat prediksi, KNN hanya perlu menghitung jarak antara data dengan data latih yang diberikan tanpa menggunakan parameter yang perlu dilatih.

## Kekurangan

- Komputasi berat
	
	Karena KNN menghitung jarak pada seluruh data relatif terhadap data input, algoritma ini sangat lambat untuk dataset besar atau berdimensi tinggi. Untuk Naive KNN, time complexity `O(ND)` dimana `N` adalah banyak data dan `D` adalah banyak dimensi.

- Sensitif terhadap nilai $k$
	- Jika nilai $k$ terlalu kecil, maka KNN akan mengalami overfitting.
	- Jika nilai $k$ terlalu besar, maka KNN akan mengalami underfitting.

## Implementasi

Berikut adalah cara mengimplementasikan KNN dengan library `scikit-learn`.

```python
from sklearn.neighbors import KNeighborsClassifier

# Matriks fitur
X_train = [
	[0.1, -0.3],
	[-0.2, 1.5],
	[1, 3],
	[-0.6, 1.3],
	[2.3, -4.7],
	[0.4, 0.9],
	[-0.1, 1.3],
	# ...
]

# Vektor label
y_train = [
	0,
	1,
	0,
	1,
	0,
	1,
	0,
	# ...
]

# Data uji
X_test = [
    [0.1, 0.5],
    # ...
]

# Jumlah tetangga (k)
k = 5

knn = KNeighborsClassifier(
    n_neighbors=k,      # Jumlah tetangga
    weights='uniform',  # Pembobotan jarak
    metric='minkowski'  # Metrik jarak
    p=2,                # Nilai pangkat (untuk metrik minkowski)
)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(pred)
```

## Referensi

- [Scikit-Learn](https://scikit-learn.org/stable/modules/neighbors.html#classification)