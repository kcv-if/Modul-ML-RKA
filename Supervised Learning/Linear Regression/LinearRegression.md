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

Jika kita ingin **menebak harga rumah** hanya dari **luas bangunan**. kita punya beberapa contoh data: luas dan harganya. Kalau semua contoh itu digambar sebagai titik di kertas (X = luas, Y = harga), maka tugas kita adalah **menarik satu garis lurus** yang paling cocok untuk melewati kumpulan titik tersebut. Garis ini membantu kita **memperkirakan harga rumah lain** yang belum diketahui harganya.

**Istilah dasar:**

- **Predictor / feature** ($x$): variabel input, misalnya luas bangunan.
- **Target / response** ($y$): nilai yang ingin diprediksi, misalnya harga rumah.
- **Training data**: kumpulan pasangan $(x, y)$ yang dipakai model untuk belajar.
- **Parameter** ($\beta$): angka-angka yang dipelajari model dari data (di sini: kemiringan dan titik potong garis).

**Kapan cocok?**

- Saat hubungan X dan Y **kurang lebih searah** (kalau X naik, Y ikut naik/turun dengan pola relatif rata).
- Saat kita ingin **memahami pengaruh** tiap faktor (ex: setiap luas nambah 1 m², harga rumah naik berapa?).

**Kapan kurang cocok?**

- Saat pola hubungan **berkelok/kompleks** (non-linear kuat). Untuk kasus ini lihat modul [Polynomial Regression](PolynomialRegression.md).
- Saat ada **banyak outlier** (data ekstrem yang menyimpang).
- Saat fitur saling **tumpang tindih** kuat (multikolinearitas), membuat koefisien tidak stabil.

## Definisi

**Linear Regression** adalah algoritma regresi **Supervised Learning** untuk memprediksi nilai output kontinu berdasarkan hubungan linear dengan input. Model mengasumsikan perubahan pada input akan diikuti perubahan pada output secara **proporsional**.

[![](https://camo.githubusercontent.com/d090b6e3bbaa687912184df8f09ffcfb9838434c3f6013346a788fc5c00dd411/68747470733a2f2f6d656469612e6765656b73666f726765656b732e6f72672f77702d636f6e74656e742f75706c6f6164732f32303233313032313135333933302f67682e706e67)](https://camo.githubusercontent.com/d090b6e3bbaa687912184df8f09ffcfb9838434c3f6013346a788fc5c00dd411/68747470733a2f2f6d656469612e6765656b73666f726765656b732e6f72672f77702d636f6e74656e742f75706c6f6164732f32303233313032313135333933302f67682e706e67)

Secara statistik, Linear Regression memetakan hubungan linear antara fitur $\mathbf{X}$ dan target kontinu $y$:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p
$$

dengan $\beta$ adalah koefisien (parameter) yang dipelajari dari data.

### 1) Simple Linear Regression

Untuk kasus **satu fitur (satu variabel input)**, hubungan paling dasar ditulis:

$$
\hat{y} = \beta_0 + \beta_1 x
$$

**Di mana:**

- $\hat{y}$ = **nilai prediksi** (variabel terikat / dependent)
- $x$ = **input** (variabel bebas / independent)
- $\beta_1$ = **kemiringan (slope)** → seberapa banyak $\hat{y}$ berubah saat $x$ naik 1 unit
- $\beta_0$ = **intersep (intercept)** → nilai $\hat{y}$ ketika $x = 0$

**Best-fit line** adalah garis yang memilih nilai $\beta_0$ dan $\beta_1$ sehingga **prediksi** sedekat mungkin dengan **nilai aktual**.

> **Catatan:** beberapa sumber memakai $m$ untuk slope dan $b$ untuk intercept, sehingga $m \equiv \beta_1$ dan $b \equiv \beta_0$. Modul ini konsisten memakai $\beta$.

### 2) Multiple Linear Regression

Ketika fiturnya **lebih dari satu**, maka garis pada 2D melebar menjadi **bidang/hiperbidang**:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p
$$

- **Simple Linear Regression**: hanya **satu** fitur (garis di bidang 2D).
- **Multiple Linear Regression**: **banyak** fitur (bidang/hiperbidang di dimensi lebih tinggi).

Untuk $n$ sampel dan $p$ fitur, semua data dan model bisa ditulis ringkas dalam **bentuk matriks**:

$$
\mathbf{y} = X\beta + \epsilon
$$

$$
\mathbf{y} = \begin{pmatrix} y_1 \\ \vdots \\ y_n \end{pmatrix}, \qquad
X = \begin{pmatrix}
1 & x_{1,1} & \cdots & x_{1,p} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{n,1} & \cdots & x_{n,p}
\end{pmatrix}, \qquad
\beta = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_p \end{pmatrix}
$$

dengan $X \in \mathbb{R}^{n \times (p+1)}$ (kolom pertama berisi angka 1 untuk intercept $\beta_0$), $\beta \in \mathbb{R}^{p+1}$, dan $\epsilon$ adalah **error** (selisih antara nilai aktual dan garis/bidang).

### 3) Loss Function

Selisih antara nilai aktual dan prediksi disebut **residual**: $e_i = y_i - \hat{y}_i$. Agar model "sedekat mungkin" dengan data, kita meminimalkan **Sum of Squared Errors (SSE)**:

$$
\mathcal{L}(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \lVert \mathbf{y} - X\beta \rVert_2^2
$$

atau versi rata-ratanya, **Mean Squared Error (MSE)**: $\text{MSE} = \frac{1}{n}\,\text{SSE}$. Metode ini disebut **Ordinary Least Squares (OLS)**.

**Kenapa error dikuadratkan?**

- Error positif dan negatif **tidak saling menghapus**.
- Error besar **dihukum lebih berat** daripada error kecil.
- Fungsinya mulus sehingga mudah diturunkan (dibutuhkan untuk mencari parameter terbaik).

> **Hasil penurunan:** alasan yang lebih mendasar untuk loss kuadrat berasal dari asumsi bahwa error berdistribusi normal, $\epsilon \sim N(0, \sigma^2)$. Dengan asumsi ini, memaksimalkan likelihood data (**Maximum Likelihood Estimation**) ternyata setara dengan meminimalkan SSE. Penurunan lengkapnya ada di [CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf).

## Cara Kerja

Ringkasan alurnya:

1. **Siapkan data**: bentuk matriks desain $X$ dan target $\mathbf{y}$.
2. **Rumuskan loss function**: SSE/MSE.
3. **Estimasi parameter**: cari $\hat{\beta}$ yang meminimalkan loss (closed form atau gradient descent).
4. **Prediksi** data baru.
5. **Evaluasi & diagnostik** hasilnya.

### 1) Siapkan data

- Bentuk **matriks desain** $X$ dengan $n$ sampel dan $p$ fitur. Setelah kolom 1 untuk **intercept** ($\beta_0$) ditambahkan, ukurannya menjadi $X \in \mathbb{R}^{n \times (p+1)}$.
- Rapikan data: tangani nilai hilang (missing values), dan **encode** fitur kategorikal.
- Pisahkan data menjadi **training set** (untuk melatih) dan **test set** (untuk menguji pada data yang belum pernah dilihat model).
- Lakukan **standardization** fitur bila perlu (rumus di bawah).

**Standardization** mengubah tiap kolom fitur menjadi rata-rata 0 dan simpangan baku 1:

$$
z = \frac{x - \mu}{\sigma}
$$

dengan $\mu$ (rata-rata) dan $\sigma$ (simpangan baku) dihitung **hanya dari training set**, lalu angka yang sama dipakai untuk test set. Target $y$ tidak ikut distandardisasi. Standardization **sangat penting untuk gradient descent** (lihat bagian di bawah), sedangkan untuk closed form solution tidak wajib.

### 2) Rumuskan loss function

Tujuannya meminimalkan SSE: $\mathcal{L}(\beta) = \lVert \mathbf{y} - X\beta \rVert_2^2$ (atau MSE, hasil $\hat{\beta}$-nya sama).

### 3) Estimasi parameter

Ada dua keluarga cara untuk mencari $\hat{\beta}$:

| | Closed form solution | Gradient descent |
|---|---|---|
| **Cara** | Dihitung langsung dalam sekali jalan | Iteratif: memperbaiki $\beta$ sedikit demi sedikit |
| **Contoh** | Normal equation, solver berbasis SVD | Batch GD, Stochastic GD, Mini-batch GD |
| **Dipakai di praktik** | Ya, default untuk linear regression biasa (termasuk scikit-learn) | Untuk data sangat besar, data streaming, dan sebagai fondasi model lain |

#### A) Closed form solution (Normal Equation)

Jika $X^\top X$ bisa diinvers, parameter yang meminimalkan SSE adalah:

$$
\hat{\beta} = (X^\top X)^{-1} X^\top \mathbf{y}
$$

> **Hasil penurunan:** turunkan $\mathcal{L}(\beta)$ terhadap $\beta$ dan samakan dengan nol: $\nabla_\beta \mathcal{L} = -2X^\top(\mathbf{y} - X\beta) = 0$, sehingga $X^\top X \beta = X^\top \mathbf{y}$ (inilah yang disebut **normal equations**). Detailnya ada di [CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf).

Syaratnya, kolom-kolom $X$ saling bebas linear (tidak ada fitur yang merupakan kombinasi fitur lain). Jika tidak, $X^\top X$ tidak bisa diinvers.

#### B) Apa yang sebenarnya dilakukan scikit-learn

`LinearRegression` di scikit-learn **tidak** menghitung $(X^\top X)^{-1}$ secara literal. Ia memakai `scipy.linalg.lstsq`, yaitu solver least squares berbasis **SVD (Singular Value Decomposition)**. Ini tetap solusi langsung (tanpa iterasi), dan hasilnya sama dengan normal equation pada kasus normal.

**SVD** menguraikan matriks $X$ menjadi tiga matriks yang lebih sederhana:

$$
X = U \Sigma V^\top
$$

Secara intuitif: $V^\top$ memutar ruang fitur, $\Sigma$ (matriks diagonal berisi **singular values**) meregangkan atau memampatkan sepanjang tiap sumbu, lalu $U$ memutar hasilnya. Untuk $X$ yang kolomnya saling bebas, solusi least squares menjadi:

$$
\hat{\beta} = V \Sigma^{-1} U^\top \mathbf{y}
$$

> **Hasil penurunan:** ini setara dengan normal equation, karena $X^\top X = V \Sigma^2 V^\top$.

**Kenapa ini lebih disukai daripada menghitung invers langsung?**

- **Stabilitas numerik.** Membentuk $X^\top X$ **mengkuadratkan condition number**: $\kappa(X^\top X) = \kappa(X)^2$. Condition number mengukur seberapa sensitif solusi terhadap galat kecil (pembulatan). Makin besar, makin tidak stabil. SVD bekerja langsung pada $X$ sehingga menghindari pengkuadratan ini.
- **Fitur redundan.** Jika ada singular value yang (hampir) nol, misalnya dua kolom identik, normal equation gagal ("singular matrix"). Solver berbasis SVD tetap memberi solusi, yaitu solusi dengan norma $\beta$ terkecil.

Beberapa detail lain di scikit-learn:

- Dengan `fit_intercept=True`, data **dipusatkan dulu** (rata-rata dikurangkan), koefisien dicari pada data terpusat, lalu intercept dihitung belakangan: $\hat{\beta}_0 = \bar{y} - \bar{\mathbf{x}}^\top \hat{\beta}$. Hasilnya setara dengan menambahkan kolom 1 pada $X$.
- Untuk data **sparse**, scikit-learn memakai `scipy.sparse.linalg.lsqr` (solver iteratif).
- Dengan `positive=True` (koefisien dipaksa tidak negatif), dipakai `scipy.optimize.nnls`.
- **Ridge (L2)**: $\hat{\beta}_{\text{ridge}} = (X^\top X + \alpha I)^{-1} X^\top \mathbf{y}$ lebih stabil saat fitur saling berkorelasi (dibahas di modul [Lasso & Ridge Regression](LassoRidgeRegression.md)).

#### C) Gradient Descent

**Posisi gradient descent.** Untuk linear regression biasa pada data berukuran sedang, closed form solution adalah pilihan standar dan gradient descent jarang dipakai. Gradient descent tetap penting karena:

1. Dipakai saat **data sangat besar** atau **data datang terus-menerus** (streaming), ketika solusi langsung terlalu berat.
2. Menjadi **fondasi pelatihan model lain** (logistic regression, neural network) yang tidak punya closed form solution. Linear regression adalah contoh paling bersih untuk memahaminya, karena loss-nya berbentuk mangkuk (convex).

**Intuisi.** Bayangkan kita berdiri di lereng bukit dengan mata tertutup dan ingin sampai ke lembah. kita rasakan arah turun dari tempat berdiri, ambil satu langkah ke arah itu, lalu ulangi. "Tinggi bukit" adalah nilai loss, posisi kita adalah nilai parameter $\beta$, dan ukuran langkahnya adalah **learning rate** $\eta$.

**Enam tahap gradient descent:**

1. **Tentukan model dan loss function**: $\hat{y} = X\beta$ dan $J(\beta) = \frac{1}{n}\lVert X\beta - \mathbf{y} \rVert_2^2$ (MSE).
2. **Beri nilai awal** $\beta$ (biasanya nol atau angka acak kecil).
3. **Hitung prediksi dan loss** dengan $\beta$ saat ini.
4. **Hitung gradien** $\nabla J(\beta)$, yaitu turunan loss terhadap tiap parameter. Gradien menunjuk arah **naik** paling curam.
5. **Update parameter** ke arah sebaliknya (turun):

$$
\beta \leftarrow \beta - \eta \, \nabla J(\beta)
$$

6. **Ulangi tahap 3 sampai 5** sampai berhenti.

Untuk MSE pada linear regression, gradiennya:

$$
\nabla J(\beta) = \frac{2}{n} X^\top (X\beta - \mathbf{y})
$$

> **Hasil penurunan:** gradien ini didapat dengan menurunkan $J(\beta)$ terhadap $\beta$ (aturan rantai pada kuadrat). Detailnya ada di [CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf). CS229 memakai loss $\frac{1}{2}\sum(\cdot)^2$ sehingga konstanta di depan gradiennya berbeda; itu hanya soal skala dan setara dengan mengubah learning rate.

**Kriteria berhenti** (salah satu): jumlah iterasi maksimum tercapai, loss hampir tidak berubah lagi (di bawah tolerance), atau gradien sudah hampir nol.

Dalam bentuk pseudocode:

```
β ← nilai awal
ulangi:
    g ← (2/n) · Xᵀ(Xβ − y)      # gradien
    β ← β − η · g               # update
sampai kriteria berhenti terpenuhi
```

**Contoh hitungan kecil.** Agar hanya ada satu parameter, pakai model $\hat{y} = \beta x$ (tanpa intercept) dengan data $(x, y)$: $(1, 2), (2, 4), (3, 6)$. Jawaban idealnya $\beta = 2$. Gradiennya $\nabla J = \frac{2}{3}\sum (\beta x_i - y_i)x_i \approx 9.33\beta - 18.67$. Dengan $\eta = 0.1$ dan nilai awal $\beta = 0$:

| Iterasi | $\beta$ | Gradien | $\beta$ baru |
|---|---|---|---|
| 1 | 0 | −18.67 | 1.867 |
| 2 | 1.867 | −1.244 | 1.991 |
| 3 | 1.991 | −0.083 | 1.999 |

Langkah awal besar karena lerengnya curam, lalu makin kecil karena gradiennya mengecil menjelang dasar lembah.

**Learning rate ($\eta$).** Pada contoh di atas, jarak ke jawaban benar dikalikan faktor tetap tiap iterasi:

$$
\beta_{\text{baru}} - 2 = (\beta - 2)\,(1 - \eta \cdot 9.33)
$$

> **Hasil penurunan aljabar:** substitusikan gradien ke aturan update, lalu kurangkan 2 di kedua sisi. Angka 9.33 adalah $\frac{2}{n}\sum x_i^2$, yaitu **kelengkungan** (curvature) loss.

| $\eta$ | Faktor $1 - 9.33\eta$ | Perilaku |
|---|---|---|
| 0.1 | 0.07 | menuju jawaban dengan cepat |
| 0.01 | 0.91 | menuju jawaban, tapi lambat |
| 0.25 | −1.33 | **divergen**: jarak malah membesar tiap iterasi |

Jadi gradient descent aman jika besar faktornya kurang dari 1. Untuk kasus umum, syaratnya $\eta < \dfrac{2}{\lambda_{\max}}$, dengan $\lambda_{\max}$ adalah nilai eigen terbesar dari $H = \frac{2}{n}X^\top X$ (kelengkungan loss terbesar). Makin tajam lembahnya, makin kecil learning rate yang masih aman.

**Tiga varian gradient descent.** Keenam tahap di atas sama; yang berbeda hanya **data yang dipakai untuk menghitung gradien**:

| Varian | Data per update | Karakteristik |
|---|---|---|
| **Batch GD** | Seluruh training set | Stabil dan deterministik, tetapi berat untuk data besar |
| **Stochastic GD (SGD)** | 1 sampel | Update cepat dan murah, tetapi berfluktuasi di sekitar optimum |
| **Mini-batch GD** | Sekelompok kecil sampel | Kompromi keduanya; paling umum di deep learning |

**Konvergensi.** Loss linear regression berbentuk mangkuk (**convex**), sehingga hanya ada **satu minimum global** dan tidak ada local minimum yang menjebak. Selama learning rate cukup kecil, gradient descent pasti menuju solusi yang sama dengan closed form solution.

**Sensitivitas SGD terhadap outlier.** Pada SGD, satu update memakai satu sampel $(x_i, y_i)$ (dengan $x_i$ adalah satu baris $X$). Dengan loss per sampel $\frac{1}{2}(x_i^\top\beta - y_i)^2$, update-nya $\beta \leftarrow \beta - \eta\,(x_i^\top\beta - y_i)\,x_i$, dan residual sampel itu dikalikan:

$$
1 - \eta \lVert x_i \rVert^2
$$

Sampel yang nilai fiturnya ekstrem punya $\lVert x_i \rVert^2$ sangat besar, sehingga faktornya bisa jauh melewati batas aman dan membuat SGD **divergen**, padahal learning rate yang sama aman untuk batch GD. Batch GD memakai rata-rata gradien seluruh data, jadi satu outlier hanya berkontribusi $1/n$. Aturan praktisnya, $\eta \lVert x_i \rVert^2 < 2$ untuk sampel terburuk adalah batas yang konservatif (bukan ambang yang tajam). Standardization perlu, tetapi **tidak cukup** untuk menghilangkan masalah ini; learning rate yang lebih kecil atau penanganan outlier tetap dibutuhkan.

**Peran feature scaling.** Gradient descent sensitif terhadap skala fitur. Jika satu fitur berskala jauh lebih besar dari yang lain, kelengkungan loss pada arah itu sangat besar sehingga learning rate yang aman menjadi sangat kecil (dan arah lain konvergen sangat lambat). Karena itu, lakukan **standardization** sebelum memakai gradient descent.

**Closed form vs gradient descent:**

| Aspek | Closed form (SVD / normal equation) | Gradient descent |
|---|---|---|
| Hyperparameter | Tidak ada | Learning rate $\eta$, jumlah iterasi |
| Hasil | Eksak (sampai presisi numerik) | Pendekatan; bergantung pada $\eta$ dan kriteria berhenti |
| Feature scaling | Tidak wajib | Sangat disarankan |
| Data sangat besar | Berat (komputasi dan memori) | Bisa ditangani dengan SGD/mini-batch |
| Kegagalan | Fitur redundan pada normal equation | Divergen jika $\eta$ terlalu besar |

### 4) Prediksi

Setelah $\hat{\beta}$ didapat: $\hat{y} = X_{\text{baru}}\hat{\beta}$. Pastikan data baru diproses dengan cara yang sama seperti training data (termasuk standardization dengan $\mu$ dan $\sigma$ dari training set).

### 5) Evaluasi & diagnostik

- Metrik umum: **MAE**, **MSE/RMSE**, **R²**. Hitung pada **test set**, bukan pada training set, agar mencerminkan performa pada data baru.
- Cek **residual** (error per titik): sebaran **acak** (tidak berpola) dan varians relatif **konstan** (homoskedastis).
- Jika residual berpola melengkung, pertimbangkan **fitur non-linear** (mis. $x^2$, lihat modul [Polynomial Regression](PolynomialRegression.md)) atau model non-linear.

> Pseudocode singkat

```
X ← add_intercept(X)              # kolom 1 untuk β0
β ← argmin || y − Xβ ||²          # closed form (SVD/normal equation) atau gradient descent
ŷ ← X_new · β
evaluate(ŷ, y_true)               # RMSE, R², dst.
```

## Kelebihan

- **Sederhana & cepat**: training sangat cepat, cocok untuk baseline.
- **Mudah diinterpretasikan**: setiap koefisien menjelaskan besaran pengaruh fitur terhadap target.
- **Jumlah data tidak harus banyak**: bisa bekerja baik meski data tidak terlalu besar.
- **Analitik & inferensi**: memudahkan analisis hubungan antar variabel.
- **Tidak ada local minimum**: loss-nya convex sehingga solusi terbaiknya dijamin global dan bisa dihitung langsung (closed form).

## Kekurangan

- **Asumsi linearitas**: hubungan harus (kurang lebih) linear, sehingga pola non-linear sulit ditangkap tanpa rekayasa fitur.
- **Sensitif terhadap outlier**: beberapa titik ekstrem dapat sangat memengaruhi garis terbaik.
- **Multikolinearitas**: fitur yang saling berkorelasi tinggi membuat koefisien tidak stabil.
- **Heteroskedastisitas**: varians error yang tidak konstan menurunkan kualitas inferensi.
- **Butuh praproses**: fitur kategorikal perlu encoding, missing value harus ditangani, dan standardization diperlukan bila memakai gradient descent.

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

# Parameter hasil training
print("beta_0 (intercept):", model.intercept_)
print("beta_1 (koefisien):", model.coef_)

# Prediksi nilai target untuk data uji
y_pred = model.predict(X_test)
print(y_pred)
```

> **Catatan:** `LinearRegression` memakai closed form solution (berbasis SVD). Untuk versi gradient descent, scikit-learn menyediakan `SGDRegressor`. Gunakan `penalty=None` agar hasilnya setara OLS murni (secara default `SGDRegressor` memakai regularisasi L2), dan lakukan standardization fitur (misalnya dengan `StandardScaler`) sebelum melatihnya. Learning rate awalnya diatur lewat parameter `eta0`.

## Referensi

- [Scikit-Learn - Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Scikit-Learn - SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
- [SciPy - scipy.linalg.lstsq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html)
- [CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf)
- [Lecture 6: Multiple Linear Regression, Polynomial Regression and Model Selection](https://harvard-iacs.github.io/2018-CS109A/lectures/lecture-6/)
