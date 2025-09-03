# Linear Regression

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Definisi](#definisi)
- [Cara Kerja](#cara-kerja)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Definisi
**Linear Regression** adalah algoritma regresi **Supervised Learning** yang digunakan untuk memprediksi nilai output berdasarkan hubungan linear dengan input. Model ini mengasumsikan bahwa ada hubungan lurus antara input dan output, di mana output akan berubah secara proporsional terhadap perubahan input.

## Cara Kerja

1. **Siapkan data**

   * Bentuk **matriks desain** $X \in \mathbb{R}^{n \times p}$ (n = jumlah sampel, p = jumlah fitur).
   * Tambahkan kolom 1 untuk **intercept** jika memakai konstanta ($\beta_0$).

2. **Formulasi fungsi loss (OLS)**

   * Tujuan: meminimalkan **Sum of Squared Errors (SSE)**

     $$
     \mathcal{L}(\beta) = \|y - X\beta\|_2^2
     $$

3. **Estimasi parameter**

   * **Solusi tertutup (Normal Equation)** jika $X^\top X$ dapat diinvers:

     $$
     \hat{\beta} = (X^\top X)^{-1} X^\top y
     $$
   * Praktik umum (seperti di `scikit-learn`): gunakan dekomposisi numerik (QR/SVD) yang lebih stabil daripada menghitung invers secara eksplisit.
   * Untuk **Ridge** (regularisasi L2):

     $$
     \hat{\beta}_{\text{ridge}} = (X^\top X + \alpha I)^{-1} X^\top y
     $$
   * Alternatif skala-besar: **(Stochastic) Gradient Descent** sampai konvergen.

4. **Prediksi**

   * Setelah $\hat{\beta}$ didapat, prediksi:

     $$
     \hat{y} = X_{\text{baru}} \hat{\beta}
     $$

5. **Evaluasi & Diagnostik**

   * Ukur performa (MAE/MSE/RMSE/$R^2$).
   * Cek **residual**: sebaran acak (tidak berpola), varians relatif konstan (homoskedastis).
   * Jika ada pola non-linear, pertimbangkan rekayasa fitur atau model non-linear.

> Pseudocode singkat

```text
X ← add_intercept(X)              # opsional jika pakai konstanta
β ← argmin || y − Xβ ||^2         # OLS (via QR/SVD/gradient descent)
ŷ ← X_new · β
evaluate(ŷ, y_true)               # RMSE, R², dst.
```

## Kelebihan
* **Sederhana & cepat**: training sangat cepat, cocok untuk baseline.
* **Mudah diinterpretasikan**: setiap koefisien menjelaskan besaran pengaruh fitur terhadap target.
* **Jumlah data tidak harus banyak**: bisa bekerja baik meski data tidak terlalu besar.
* **Analitik & inferensi**: memudahkan analisis hubungan antar variabel.

## Kekurangan
* **Asumsi linearitas**: hubungan harus (kurang lebih) linier, sehinnga pola non-linear sulit ditangkap tanpa rekayasa fitur.
* **Sensitif terhadap outlier**: beberapa titik ekstrem dapat sangat memengaruhi garis terbaik.
* **Multikolinearitas**: fitur yang saling berkorelasi tinggi membuat koefisien tidak stabil.
* **Heteroskedastisitas**: varians error yang tidak konstan menurunkan kualitas inferensi.
* **Butuh praproses**: fitur kategorikal perlu encoding. missing value harus ditangani.

## Implementasi

Berikut adalah cara mengimplementasikan Linear Regression dengan library `scikit-learn`.

```python
from sklearn.linear_model import LinearRegression

# Data train
X_train = [[1], [2], [3], [4], [5], [6]]
y_train = [2, 2.5, 4.5, 3, 5, 4.7]

# Data uji / test
X_test = [[7], [8]]

# Inisialisasi & melatih model
model = LinearRegression(fit_intercept=True) 
model.fit(X_train, y_train)

# fit_intercept=True -> model menggunakan konstanta (b) pada persamaan (w*x + b)
# fit_intercept=False -> b = 0, sehingga persamaan hanya (w*x), garis dipaksa lewat titik (0,0)

# Prediksi nilai target untuk data uji
y_pred = model.predict(X_test)
print(y_pred)
```

## Referensi


