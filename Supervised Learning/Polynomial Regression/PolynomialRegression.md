# Polynomial Regression

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Pendahuluan](#pendahuluan)
- [Definisi](#definisi)
- [Cara Kerja](#cara-kerja)
- [Memilih Derajat Polinomial](#memilih-derajat-polinomial)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Pendahuluan

Di modul [Linear Regression](LinearRegression.md), kita menarik **satu garis lurus** untuk memprediksi target dari fitur. Tapi bagaimana kalau datanya **melengkung**?

Contohnya hubungan antara **tenaga mesin (horsepower)** dan **konsumsi bahan bakar (mpg)** pada dataset *Auto MPG*: makin besar horsepower, mpg makin turun, tapi penurunannya tidak lurus, melainkan melandai. Garis lurus akan "meleset" di sebagian besar titik. Solusinya, kita ganti garis dengan **kurva polinomial**, misalnya parabola, yang bisa mengikuti lengkungan data.

**Kapan cocok?**

- Saat pola hubungan X dan Y **melengkung** (non-linear).
- Saat kamu ingin memperbaiki Linear Regression **tanpa mengganti algoritmanya**, cukup menambah fitur baru.

**Kapan kurang cocok?**

- Saat pola hubungannya sangat rumit atau tidak mulus (melonjak, berpola periodik), derajat polinomial yang dibutuhkan bisa terlalu tinggi dan model menjadi **overfitting**.
- Saat kamu perlu memprediksi **jauh di luar rentang data latih** (ekstrapolasi), karena kurva polinomial bisa melonjak tidak realistis.

## Definisi

**Polynomial Regression** adalah algoritma regresi **Supervised Learning** untuk memprediksi nilai output kontinu ketika hubungan antara variabel input dan output tidak linear, melainkan mengikuti suatu fungsi polinomial.

![](https://imgs.search.brave.com/66eIvMgihDhas0DESXoFAbU_3SylnvokAfUmK0a-0yA/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly93d3cu/dHV0b3JpYWxzcG9p/bnQuY29tL21hY2hp/bmVfbGVhcm5pbmcv/aW1hZ2VzL2xpbmVh/cl92c19wb2x5bm9t/aWFsX3JlZ3Jlc3Np/b24uanBn)


### 1) Univariate Polynomial Regression

Jika hanya terdapat **satu fitur input** $x$, maka model polinomial berderajat $M$ dirumuskan sebagai berikut.

$$
\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_M x^M
$$

**Di mana:**

- $\hat{y}$ = **nilai prediksi** (variabel terikat / dependent)
- $x$ = **input** (variabel bebas / independent)
- $M$ = **derajat (degree)** polinomial
- $\beta_0$ = **intersep (intercept)**
- $\beta_j$ = **koefisien** untuk suku $x^j$

Kasus khusus: $M = 1$ adalah **linear** (garis lurus), $M = 2$ **kuadratik** (parabola), dan $M = 3$ **kubik**.

### 2) Multivariate Polynomial Regression

Jika terdapat **lebih dari satu fitur**, maka kombinasi polinomial antar variabel (**cross term** / interaction term) dapat ditambahkan ke dalam fungsi polinomial. Berikut contoh dari multivariate polynomial regression dengan 2 fitur, yakni $a$ dan $b$, berderajat 2:

$$
\hat{y} = \beta_0 + \beta_1 a + \beta_2 b + \beta_3 a^2 + \beta_4 a b + \beta_5 b^2
$$

dengan $\beta$ adalah koefisien yang dipelajari dari data.

**Perhatian:** jumlah suku (termasuk intercept) untuk $p$ fitur dan derajat $M$ adalah $\binom{p + M}{M}$. Jumlah ini **membengkak sangat cepat**. Misalnya 8 fitur dengan derajat 3 sudah menghasilkan 164 fitur baru (di luar intercept).

### 3) Kenapa tetap disebut "linear"?

Nama "Linear" pada Linear Regression merujuk pada hubungan **terhadap parameter $\beta$**, bukan terhadap $x$. Pada polynomial regression, kita cukup mendefinisikan **fitur baru**:

$$
z_1 = x, \quad z_2 = x^2, \quad \ldots, \quad z_M = x^M
$$

sehingga modelnya menjadi $\hat{y} = \beta_0 + \beta_1 z_1 + \beta_2 z_2 + \cdots + \beta_M z_M$. Ini adalah **Linear Regression biasa** pada fitur $z$. Secara umum bisa ditulis dengan **feature map** $\varphi(x) = (1, x, x^2, \ldots, x^M)^\top$:

$$
\hat{y} = \beta^\top \varphi(x)
$$

Akibatnya, loss function-nya tetap kuadratik terhadap $\beta$, dan semua cara mencari parameter di modul [Linear Regression](LinearRegression.md) (closed form maupun gradient descent) berlaku sama persis.

**Kontras:** model seperti $\hat{y} = \beta_1 \cos(\beta_2 x)$ **bukan** linear terhadap parameter, karena $\beta_2$ berada di dalam fungsi non-linear. Model seperti ini tidak bisa diselesaikan dengan cara di atas.

> **Hasil penurunan:** aturan update dan solusi least squares dengan feature map $\varphi(x)$ memiliki bentuk yang sama dengan Linear Regression, hanya saja $x$ diganti $\varphi(x)$. Detailnya ada di [CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf).

## Cara Kerja

Ringkasan alurnya:

1. **Siapkan data** dan pisahkan menjadi training, validation, dan test set.
2. **Transformasi fitur** menjadi fitur polinomial sampai derajat $M$.
3. **Standardization** tiap kolom fitur hasil transformasi.
4. **Bentuk design matrix**.
5. **Rumuskan loss function** (OLS).
6. **Estimasi parameter** (sama dengan Linear Regression).
7. **Prediksi**.

### 1) Siapkan data

- Tangani nilai hilang (missing values), dan **encode** fitur kategorikal.
- Pisahkan data menjadi **training set**, **validation set**, dan **test set** (peran masing-masing dijelaskan di bagian [Memilih Derajat Polinomial](#memilih-derajat-polinomial)). Lakukan pemisahan **sebelum** transformasi dan standardization agar tidak ada informasi dari data validation/test yang bocor ke proses training.

### 2) Transformasi fitur

Ciptakan fitur polinomial beserta interaksi antar fitur hingga derajat $M$ (mis. $x, x^2, \ldots, x^M$ untuk satu fitur).

### 3) Standardization

Fitur hasil pemangkatan punya skala yang sangat berbeda (misalnya $x$ bernilai 100, tapi $x^3$ bernilai 1.000.000). Karena itu lakukan **standardization**:

$$
z = \frac{x - \mu}{\sigma}
$$

dengan aturan penting berikut:

- Standardization dilakukan pada **fitur** (kolom $X$), **bukan** pada target $y$.
- Tiap kolom ($x$, $x^2$, $x^3$, dst.) distandardisasi **terpisah**, **setelah** dipangkatkan.
- $\mu$ dan $\sigma$ dihitung **hanya dari training set**, lalu angka yang sama dipakai untuk validation dan test set.
- Kolom konstanta 1 untuk intercept ditambahkan **setelah** standardization (tidak ikut distandardisasi).

### 4) Bentuk design matrix

Untuk $n$ sampel dengan satu fitur, **design matrix** berisi satu kolom untuk tiap pangkat:

$$
\mathbf{y} = \begin{pmatrix} y_1 \\ \vdots \\ y_n \end{pmatrix}, \qquad
X = \begin{pmatrix}
1 & x_1 & x_1^2 & \cdots & x_1^M \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_n & x_n^2 & \cdots & x_n^M
\end{pmatrix}, \qquad
\beta = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_M \end{pmatrix}
$$

sehingga $X \in \mathbb{R}^{n \times (M+1)}$ (dengan kolom pertama berisi angka 1 untuk intercept). Setelah standardization, kolom-kolom $x, x^2, \ldots$ diganti dengan versi terstandardisasinya.

### 5) Rumuskan loss function (OLS)

Sama seperti Linear Regression, tujuannya meminimalkan **Sum of Squared Errors (SSE)**:

$$
\mathcal{L}(\beta) = \lVert \mathbf{y} - X\beta \rVert_2^2
$$

### 6) Estimasi parameter

Karena modelnya linear terhadap $\beta$, **cara mencari parameter sama persis dengan Linear Regression** (lihat modul [Linear Regression](LinearRegression.md), bagian *Estimasi parameter*):

- **Closed form solution**: normal equation $\hat{\beta} = (X^\top X)^{-1} X^\top \mathbf{y}$, atau solver berbasis SVD (yang dipakai scikit-learn).
- **Gradient descent** (batch, stochastic, mini-batch), untuk data yang sangat besar.
- **Ridge (L2)** untuk menstabilkan koefisien (lihat modul [Lasso & Ridge Regression](LassoRidgeRegression.md)).

Ada beberapa hal **khusus polynomial regression** yang perlu diperhatikan:

1. **Fitur saling berkorelasi.** Kolom $x, x^2, x^3, \ldots$ pada dasarnya saling berkaitan, dan korelasinya tetap tinggi meskipun sudah distandardisasi. Akibatnya **condition number** $\kappa(X)$ membesar seiring derajat $M$. Normal equation dengan invers eksplisit **mengkuadratkan** condition number ($\kappa(X^\top X) = \kappa(X)^2$) sehingga lebih rentan pada derajat tinggi, sedangkan solver berbasis SVD jauh lebih tahan. Itulah salah satu alasan scikit-learn tidak menghitung invers secara langsung.
2. **Gradient descent butuh standardization.** Tanpa standardization, kolom berpangkat tinggi memiliki skala sangat besar sehingga learning rate yang aman menjadi sangat kecil. Jika memakai learning rate biasa, nilai loss bisa meledak sampai tak hingga (`inf`/`nan`). Bahkan setelah standardization, konvergensi bisa **lambat** karena kolom yang berkorelasi menciptakan arah dengan kelengkungan (curvature) loss yang sangat kecil.
3. **Koefisien sulit ditafsirkan sendiri-sendiri.** Karena kolom-kolomnya berkorelasi kuat, koefisien individual bisa besar dan saling menutupi (misalnya satu positif besar, satu negatif besar). Yang bermakna adalah **kurva hasilnya**, bukan koefisien satu per satu.

### 7) Prediksi

Untuk data baru $x_{\text{baru}}$, terapkan **transformasi dan standardization yang sama** (dengan $\mu$ dan $\sigma$ dari training set), lalu hitung $\hat{y} = \varphi(x_{\text{baru}})^\top \hat{\beta}$.

> Pseudocode singkat

```
X_poly ← PolynomialFeatures(X, derajat M)      # transformasi fitur
X_poly ← standardize(X_poly)                   # μ, σ dari training set saja
X_poly ← add_intercept(X_poly)                 # kolom 1 untuk β0
β ← argmin || y − X_poly β ||²                 # closed form (SVD) atau gradient descent
ŷ ← transform(X_new) · β                       # transformasi & scaler yang sama
```

## Memilih Derajat Polinomial

Derajat $M$ adalah **hyperparameter**: nilainya tidak dipelajari otomatis oleh OLS, melainkan harus kita pilih. Pemilihannya krusial karena menentukan kompleksitas model.

### Underfitting dan Overfitting

| Kondisi | Derajat $M$ | Error pada training set | Error pada validation set | Bentuk kurva |
|---|---|---|---|---|
| **Underfitting** | terlalu kecil | tinggi | tinggi | terlalu kaku, tidak menangkap pola |
| **Good Fit** | sedang | rendah | **terendah** | mengikuti pola data |
| **Overfitting** | terlalu besar | sangat rendah | tinggi | meliuk-liuk mengikuti noise |

**Overfitting** adalah fenomena ketika model terlalu kompleks sehingga ikut "menghafal" noise pada data latih alih-alih pola sebenarnya, sehingga performanya turun pada data baru. Tandanya: **error training kecil tapi error pada data baru besar**.

Perhatikan bahwa error pada training set akan **terus turun** (secara teori tidak pernah naik) seiring bertambahnya derajat, sehingga **tidak boleh dipakai untuk memilih $M$**. Error pada validation set biasanya berbentuk huruf **U**: turun dulu, lalu naik saat model mulai overfitting.

> ini dikenal sebagai **bias-variance tradeoff**. Derajat rendah punya bias tinggi tapi variance rendah (underfitting); derajat tinggi punya bias rendah tapi variance tinggi (overfitting). Detailnya ada di [CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf).

### Training, Validation, dan Test Set

| Data | Fungsi |
|---|---|
| **Training set** | Melatih model (mencari $\beta$) untuk tiap kandidat derajat $M$ |
| **Validation set** | Membandingkan kandidat derajat dan **memilih $M$ terbaik** |
| **Test set** | Mengukur performa akhir model terpilih. **Dipakai sekali saja, di akhir** |

Test set tidak boleh dipakai untuk memilih $M$, karena jika begitu estimasi performanya menjadi terlalu optimis (informasi test set ikut mempengaruhi model).

**Prosedur memilih derajat:**

1. Pisahkan data menjadi training, validation, dan test set.
2. Untuk tiap kandidat $M = 1, 2, 3, \ldots$: latih model pada training set, lalu hitung error pada training dan validation set.
3. Pilih $M$ dengan **error validation terendah**.
4. Evaluasi model terpilih **sekali** pada test set.

### Cross Validation

Satu validation set saja bisa menyesatkan, karena hasilnya bergantung pada pembagian data (dan model bisa ikut overfitting ke validation set tersebut). Solusinya adalah **K-fold cross validation**:

1. Bagi data latih menjadi $K$ bagian (fold) berukuran sama.
2. Secara bergantian, pakai 1 fold sebagai validation set dan $K-1$ fold sisanya sebagai training set (sebanyak $K$ kali).
3. Rata-ratakan error dari $K$ percobaan tersebut:

$$
\text{CV}(M) = \frac{1}{K} \sum_{k=1}^{K} \text{Error}_k
$$

Pilih $M$ dengan nilai $\text{CV}$ terendah. Cara ini lebih andal, terutama saat data sedikit.

> **Catatan:** derajat terbaik bisa berbeda tergantung pembagian data. Karena itu, laporkan hasil dengan hati-hati dan hindari menyimpulkan "derajat X selalu terbaik". Detail cross validation ada di [CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf).

Selain memilih derajat, overfitting juga bisa dikurangi dengan **regularization** (lihat modul [Lasso & Ridge Regression](LassoRidgeRegression.md)).

## Kelebihan

- **Menangkap pola non-linear**: mampu merepresentasikan hubungan yang lebih kompleks.
- **Sederhana**: hanya perlu menambahkan transformasi fitur.
- **Kompatibel dengan Linear Regression**: setelah transformasi, tetap dapat diproses dengan algoritma regresi linear (closed form maupun gradient descent).

## Kekurangan

- **Overfitting**: derajat polinomial yang terlalu tinggi membuat model overfit terhadap data latih.
- **Harus memilih derajat**: derajat $M$ adalah hyperparameter yang perlu dipilih lewat validation set atau cross validation.
- **Ekstrapolasi buruk**: prediksi di luar rentang data dapat bersifat tidak realistis (kurva bisa melonjak atau anjlok tajam di tepi).
- **Sensitif terhadap skala**: nilai polinomial bisa besar, sehingga perlu dilakukan standardization fitur.
- **Fitur membengkak**: pada multivariate, jumlah fitur bertambah sangat cepat seiring derajat dan jumlah fitur.
- **Multikolinearitas**: kolom $x, x^2, \ldots$ saling berkorelasi, sehingga condition number membesar dan koefisien tidak stabil.
- **Kurang interpretatif**: semakin tinggi orde, semakin sulit menjelaskan arti setiap parameter.

## Implementasi

Berikut adalah cara mengimplementasikan Polynomial Regression dengan library `scikit-learn`. `PolynomialFeatures` membuat fitur polinomial, `StandardScaler` melakukan standardization, dan `LinearRegression` mencari parameternya. Ketiganya digabung dalam satu `Pipeline` agar transformasi dan scaler yang sama otomatis dipakai saat prediksi.

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# Data train: [luas bangunan, jumlah kamar] -> harga
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

# Pipeline: transformasi polinomial -> standardization -> linear regression
model = make_pipeline(
   PolynomialFeatures(
      degree=2,                # Derajat polinomial (M)
      interaction_only=False,  # False: pangkat murni (a^2, b^2) DAN interaksi (a*b) ikut dibuat
                               # True : hanya interaksi (a*b), tanpa pangkat murni
      include_bias=False       # False: kolom konstanta tidak dibuat (intercept diurus LinearRegression)
   ),
   StandardScaler(),           # tiap kolom fitur hasil transformasi distandardisasi terpisah
   LinearRegression()
)

# fit: statistik (mean & std) StandardScaler dihitung dari data train saja
model.fit(X_train, y_train)

# predict: transformasi dan scaler yang sama otomatis dipakai untuk data uji
y_pred = model.predict(X_test)
print(y_pred)

# Fitur yang dihasilkan PolynomialFeatures
print(model.named_steps["polynomialfeatures"].get_feature_names_out(["luas", "kamar"]))
```

> **Catatan:** data contoh di atas sangat kecil (6 sampel dengan 6 parameter, sehingga model bisa mencocokkan data latih secara sempurna) dan hanya untuk mengilustrasikan penggunaan API. Pada praktiknya, pilih derajat polinomial lewat validation set atau cross validation (misalnya dengan `cross_val_score` atau `GridSearchCV` pada parameter `polynomialfeatures__degree`), seperti dijelaskan di bagian [Memilih Derajat Polinomial](#memilih-derajat-polinomial).

## Referensi

- [Scikit-Learn - Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Scikit-Learn - Polynomial Features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [Scikit-Learn - StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Scikit-Learn - make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)
- [CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf)
- [Lecture 6: Multiple Linear Regression, Polynomial Regression and Model Selection](https://harvard-iacs.github.io/2018-CS109A/lectures/lecture-6/)
- [Polynomial regression (CSE 446: Machine Learning, University of Washington)](https://courses.cs.washington.edu/courses/cse446/22wi/schedule/week2L4_annotated.pdf)
