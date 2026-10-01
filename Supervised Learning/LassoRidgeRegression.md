# Lasso & Ridge Regression

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Definisi](#definisi)
- [Cara Kerja](#cara-kerja)
- [Kelebihan dan Kekurangan](#kelebihan-dan-kekurangan)
- [Perbandingan Ridge vs Lasso](#perbandingan-ridge-vs-lasso)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Definisi

Lasso dan Ridge Regression adalah dua teknik regularisasi yang umum digunakan dalam Machine Learning. Keduanya merupakan pengembangan dari Linear Regression biasa yang dirancang untuk mengatasi kelemahan model, terutama ketika menghadapi data dengan banyak fitur atau fitur yang saling berkorelasi tinggi. 

### Apa itu Regularisasi?
Regularisasi adalah teknik untuk mencegah overfitting dengan menambahkan penalti pada koefisien model agar tidak menjadi terlalu besar. Dengan cara ini, model dipaksa untuk tetap sederhana dan stabil, sehingga performanya tetap baik pada data baru.

### Lasso Regression
Lasso Regression (Least Absolute Shrinkage and Selection Operator), yang juga dikenal sebagai L1 Regularization adalah teknik regresi linear yang menambahkan penalty ke fungsi loss untuk mencegah overfitting. Penalty ini didasarkan pada nilai absolut koefisien.

Fungsi loss standar (mean squared error/MSE) dimodifikasi dengan menambahkan term regularisasi:

$$\text{Loss} = \text{MSE} + \lambda \sum_{i=1}^{n} |w_i|$$

Di mana:
- $MSE$ = Mean Squared Error (error standar dari prediksi)
- $\lambda$ = Parameter regularisasi yang mengontrol kekuatan penalty
- $w_i$ = Koefisien atau weights dari model
- $\sum |w_i|$ = Jumlah nilai absolut dari semua koefisien

### Ridge Regression
Ridge Regression, yang juga dikenal sebagai L2 Regularization, adalah teknik yang digunakan dalam regresi linear untuk mencegah overfitting dengan menambahkan penalty term ke fungsi loss. Penalty ini proporsional terhadap kuadrat dari besarnya koefisien (weights).

Fungsi loss standar (mean squared error/MSE) dimodifikasi dengan menambahkan term regularisasi:

$$\text{Loss} = \text{MSE} + \lambda \sum_{i=1}^{n} w_i^2$$

Di mana:
- $MSE$ = Mean Squared Error (error standar dari prediksi)
- $\lambda$ = Parameter regularisasi yang mengontrol kekuatan penalty
- $w_i$ = Koefisien atau weights dari model
- $\sum(w_i^2)$ = Jumlah kuadrat dari semua koefisien

## Cara Kerja
1. **Kumpulkan & siapkan data**
    - Buat matriks fitur $X \in \mathbb{R}^{n \times p}$, di mana $n$ = jumlah sampel dan $p$ = jumlah fitur, serta vektor target $y$.
    - **Standarisasi fitur**: Sangat penting karena penalti regularisasi sensitif terhadap skala.
    - **Encoding fitur kategorikal**: Jika ada data kategori, ubah menjadi numerik menggunakan metode encoding, seperti One-Hot Encoding atau metode lainnya agar model bisa memprosesnya.
    
2. **Tentukan Fungsi Loss**
    - Ridge: menambahkan penalti kuadrat koefisien.
    - Lasso: menambahkan penalti nilai absolut koefisien.

3. **Optimasi Parameter**
    - Ridge: Ada solusi matematis langsung (closed-form), bisa dihitung dengan rumus matriks.
    - Lasso: Tidak ada solusi langsung, perlu algoritma iteratif seperti Coordinate Descent, yaitu mencoba memperbarui setiap koefisien secara bergantian hingga fungsi loss minimal.

4. **Tentukan Hyperparameter $\lambda$ (alpha)**
    - $\lambda$ mengontrol kekuatan penalti regularisasi.
    - Gunakan cross-validation untuk menemukan nilai $\lambda$ yang memberikan keseimbangan terbaik antara akurasi dan generalisasi.
5. **Prediksi**
    - Setelah model dilatih, prediksi dilakukan dengan $\hat{y} = X_{\text{baru}} \cdot \hat{\beta}$.

## Kelebihan dan Kekurangan
### Ridge Regression
**Kelebihan:**
- **Stabil terhadap multicollinearity**: Memberikan estimasi koefisien yang lebih konsisten ketika fitur saling berkorelasi.
- **Menurunkan varians model**: Dengan menambahkan sedikit bias, Ridge membantu mengurangi varians sehingga MSE lebih rendah.
- **Tetap menggunakan semua fitur asli**: Berbeda dengan PCA atau metode dimensionality reduction, Ridge mempertahankan semua variabel sehingga hasilnya tetap mudah diinterpretasi.

**Kekurangan:**
- **Memperkenalkan bias**: Penalti yang diterapkan bisa menyebabkan underestimation terhadap pengaruh fitur yang sebenarnya kuat.
- **Pemilihan parameter α (lambda) tidak selalu mudah**: Membutuhkan tuning yang tepat, biasanya dengan cross-validation.
- **Tidak melakukan feature selection**: Semua fitur tetap digunakan, bahkan jika beberapa sebenarnya tidak relevan.


### Lasso Regression
**Kelebihan:**
- **Melakukan feature selection otomatis**: Koefisien yang tidak penting akan menjadi nol, sehingga model lebih sederhana dan efisien.
- **Mengurangi overfitting melalui regularisasi**: Membatasi besarnya koefisien membuat model lebih general dan tidak terlalu bias terhadap data training.
- **Meningkatkan interpretabilitas model**: Model yang lebih ringkas lebih mudah dijelaskan terutama dalam bidang seperti kesehatan dan keuangan.
- **Efektif untuk high-dimensional data**: Cocok untuk kasus dengan jumlah fitur yang sangat banyak (misalnya data genetik atau citra).

**Kekurangan:**
- **Bisa memilih fitur secara acak jika banyak yang saling berkorelasi**: Hal ini menyebabkan bias dalam pemilihan fitur.
- **Sensitif terhadap skala fitur**: Jika tidak dilakukan standarisasi, performanya bisa menurun drastis.
- **Rentan terhadap outlier**: Nilai ekstrem dapat memengaruhi penalti dan membuat koefisien tidak stabil.
- **Model bisa tidak konsisten**: Perubahan kecil pada data dapat menghasilkan fitur terpilih yang berbeda-beda.
- **Butuh tuning α (lambda)**: Harus dicari nilai terbaik melalui cross-validation agar hasilnya optimal.

## Perbandingan Ridge vs Lasso

| Aspek | Ridge Regression | Lasso Regression |
|---|---|---|
| Nama lain | L2 Regularization | L1 Regularization |
| Fungsi loss | $\text{MSE} + \lambda \sum w_i^2$ | $\text{MSE} + \lambda \sum \lvert w_i \rvert$ |
| Jenis penalti | Kuadrat koefisien | Nilai absolut koefisien |
| Efek pada koefisien | Mengecil mendekati nol, tetapi tidak benar-benar nol | Sebagian koefisien menjadi tepat nol |
| Feature selection | Tidak, semua fitur tetap dipakai | Ya, otomatis |
| Metode penyelesaian | Closed-form (rumus matriks) | Iteratif (Coordinate Descent) |
| Multicollinearity | Menangani dengan baik, koefisien fitur berkorelasi saling berbagi bobot | Cenderung memilih salah satu fitur dan mengabaikan yang lain |
| Interpretabilitas | Model memuat semua fitur | Model lebih ringkas karena fitur tidak penting dibuang |
| Sensitivitas terhadap outlier | Terpengaruh | Terpengaruh, dan dapat mengubah fitur yang terpilih |
| Perlu standarisasi | Ya | Ya |
| Perlu tuning $\lambda$ | Ya | Ya |
| Cocok digunakan saat | Sebagian besar fitur relevan dan saling berkorelasi | Fitur sangat banyak dan hanya sebagian yang berpengaruh |

## Implementasi
Berikut adalah cara mengimplementasikan Lasso & Ridge Regression dengan library `scikit-learn`.

Beberapa hal yang diperhatikan pada implementasi ini:
- **Standarisasi** dengan `StandardScaler`, karena penalti regularisasi sensitif terhadap skala fitur (luas dan jumlah kamar memiliki skala yang sangat berbeda).
- **`Pipeline`** agar standarisasi otomatis diterapkan juga pada data uji.
- **`RidgeCV` dan `LassoCV`** untuk memilih nilai `alpha` terbaik secara otomatis melalui cross-validation.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV

# Data train
X_train = [
   [50, 1], 
   [60, 2], 
   [80, 3], 
   [100, 3], 
   [120, 3], 
   [150, 4]
]
y_train = [100, 120, 150, 180, 200, 250]

# Data uji / test
X_test = [
   [130, 3],
   [160, 4]
]

# Ridge Regression
# alphas: daftar kandidat parameter regularisasi yang diuji
ridge_model = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=3)
)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
print("Prediksi Ridge:", ridge_pred)
print("Alpha Ridge terpilih:", ridge_model[-1].alpha_)
print("Koefisien Ridge:", ridge_model[-1].coef_)

# Lasso Regression
# alpha dicari otomatis oleh LassoCV melalui cross-validation
lasso_model = make_pipeline(
    StandardScaler(),
    LassoCV(cv=3)
)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
print("Prediksi Lasso:", lasso_pred)
print("Alpha Lasso terpilih:", lasso_model[-1].alpha_)
print("Koefisien Lasso:", lasso_model[-1].coef_)
```

> **Catatan:** Data contoh hanya terdiri dari 6 baris sehingga hasil cross-validation kurang stabil. Pada data yang lebih besar, pencarian `alpha` otomatis akan jauh lebih bermakna. Jika ingin menentukan `alpha` secara manual, gunakan `Ridge(alpha=...)` dan `Lasso(alpha=...)` di dalam pipeline yang sama.

## Referensi
- [Geeks for Geeks - Ridge Regression vs Lasso Regression](https://www.geeksforgeeks.org/machine-learning/ridge-regression-vs-lasso-regression/)
- [Medium - Lasso and Ridge Regularization Simply Explained](https://medium.com/nerd-for-tech/lasso-and-ridge-regularization-simply-explained-d551ee1e47b7)
- [Scikit-Learn - Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [Scikit-Learn - Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
