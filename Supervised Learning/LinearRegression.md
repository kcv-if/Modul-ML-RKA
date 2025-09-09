# Linear Regression

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Pendahuluan](#pendahuluan)
- [Definisi](#definisi)
- [Cara Kerja](#cara-kerja)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Pendahuluan

Bayangin kamu ingin **menebak harga rumah** hanya dari **luas bangunan**. Kamu punya beberapa contoh data: luas dan harganya. Kalau semua contoh itu digambar sebagai titik di kertas (X = luas, Y = harga), maka tugas kita adalah **menarik satu garis lurus** yang paling cocok untuk melewati kumpulan titik tersebut. Garis ini membantu kita **memperkirakan harga rumah lain** yang belum diketahui harganya.

**Kapan cocok?**

* Saat hubungan X dan Y **kurang lebih searah** (kalau X naik, Y ikut naik/turun dengan pola relatif rata).
* Saat kamu ingin **memahami pengaruh** tiap faktor (ex: setiap luas nambah 1 m², harga rumah naik berapa?).

**Kapan kurang cocok?**

* Saat pola hubungan **berkelok/kompleks** (non-linear kuat) atau ada **banyak outlier** (data ekstrem yang menyimpang).
* Saat fitur saling **tumpang tindih** kuat (multikolinearitas), membuat koefisien tidak stabil.

## Definisi

**Linear Regression** adalah algoritma regresi **Supervised Learning** untuk memprediksi nilai output kontinu berdasarkan hubungan linear dengan input. Model mengasumsikan perubahan pada input akan diikuti perubahan pada output secara **proporsional**.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20231021153930/gh.png">

Secara statistik, Linear Regression memetakan hubungan linier antara fitur \$\mathbf{X}\$ dan target kontinu \$y\$:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p
$$

dengan \$\beta\$ adalah koefisien yang dipelajari dari data. Parameter biasanya diestimasi menggunakan **Ordinary Least Squares (OLS)**, yaitu meminimalkan jumlah kuadrat selisih antara nilai aktual dan prediksi:

$$
\min_{\beta} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

### 1) Simple Linear Regression

Untuk kasus **satu fitur (satu variabel input)**, hubungan paling dasar ditulis:

$$
\hat{y} = m x + b
$$

**Di mana:**

* $\hat{y}$ = **nilai prediksi** (variabel terikat / dependent)
* $x$ = **input** (variabel bebas / independent)
* $m$ = **kemiringan (slope)** → seberapa banyak $\hat{y}$ berubah saat $x$ naik 1 unit
* $b$ = **intersep (intercept)** → nilai $\hat{y}$ ketika $x = 0$

**Best-fit line** adalah garis yang memilih nilai $m$ dan $b$ sehingga **prediksi** sedekat mungkin dengan **nilai aktual**. Umumnya dicari dengan **Ordinary Least Squares (OLS)**, yaitu meminimalkan jumlah **kuadrat error**.

> **Catatan:** dalam statistik sering dipakai $\beta_1$ untuk slope dan $\beta_0$ untuk intercept, sehingga $m \equiv \beta_1$ dan $b \equiv \beta_0$.

### 2) Multiple Linear Regression

Ketika fitur **lebih dari satu**, garis pada 2D “melebar” menjadi **bidang/hiperbidang**:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p
$$

dengan $\beta$ adalah koefisien yang dipelajari dari data. Estimasi parameter umumnya memakai **OLS**:

$$
\min_{\beta} \sum_{i=1}^n \big(y_i - \hat{y}_i\big)^2
$$

* **Simple Linear Regression**: hanya **satu** fitur (garis di bidang 2D).
* **Multiple Linear Regression**: **banyak** fitur (garis/hiperbidang di dimensi lebih tinggi).

> **Kesimpulan** kita mencari garis/hiperbidang yang paling “mendekati” data sehingga total **kuadrat selisih** (error) terkecil.

## Cara Kerja

1. **Kumpulkan & siapkan data**

   * Bentuk **matriks desain** \$X \in \mathbb{R}^{n \times p}\$ (n = jumlah sampel, p = jumlah fitur).
   * Tambahkan kolom 1 untuk **intercept** jika pakai konstanta (\$\beta\_0\$).
   * Rapikan data: tangani nilai hilang, standariskan skala bila perlu, dan **encode** fitur kategorikal.

2. **Rumuskan fungsi loss (OLS)**

   * Tujuan: meminimalkan **Sum of Squared Errors (SSE)**
     \$\mathcal{L}(\beta)=\lVert y - X\beta \rVert\_2^2\$.

3. **Estimasi parameter**

   * **Solusi tertutup (Normal Equation)** jika \$X^\top X\$ bisa diinvers:
     \$\hat{\beta}=(X^\top X)^{-1}X^\top y\$.
   * Praktik umum (mis. `scikit-learn`): gunakan **QR/SVD** (lebih stabil) daripada invers langsung.
   * **Ridge (L2)**:
     \$\hat{\beta}\_{\text{ridge}}=(X^\top X+\alpha I)^{-1}X^\top y\$ (lebih stabil saat fitur saling berkorelasi).
   * Skala besar: **(Stochastic) Gradient Descent** sampai konvergen.

4. **Prediksi**

   * Setelah \$\hat{\beta}\$ didapat: \$\hat{y}=X\_{\text{baru}}\hat{\beta}\$.

5. **Evaluasi & diagnostik**

   * Metrik umum: **MAE**, **MSE/RMSE**, **R²**.
   * Cek **residual** (error per titik): sebaran **acak** (tidak berpola) dan varians relatif **konstan** (homoskedastis).
   * Jika residual berpola melengkung, pertimbangkan **fitur non-linear** (mis. \$x^2\$) atau model non-linear.

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
- [Scikit-Learn - Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)