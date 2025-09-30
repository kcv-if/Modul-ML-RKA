# Polynomial Regression

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Pendahuluan](#pendahuluan)
- [Definisi](#definisi)
- [Cara Kerja](#cara-kerja)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Definisi

**Polynomial Regression** adalah algoritma regresi **Supervised Learning** untuk memprediksi nilai output kontinu ketika hubungan antara variabel input dan output tidak linear, melainkan mengikuti suatu fungsi polinomial.

<img src="https://indiansuccessstories.com/wp-content/uploads/2024/04/polynomial-regression.jpg" />

Secara statistik, Polynomial Regression memetakan hubungan polinomial antara fitur \$\mathbf{X}\$ dan target kontinu \$y\$ seperti berikut.

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 ^ 2 + \cdots + \beta_p x_p ^ p
$$

dengan \$\beta\$ adalah koefisien yang dipelajari dari data. Parameter biasanya diestimasi menggunakan **Ordinary Least Squares (OLS)**, yaitu meminimalkan jumlah kuadrat selisih antara nilai aktual dan prediksi:

$$
\min_{\beta} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

### 1) Univariate Polynomial Regression

Jika hanya terdapat **satu fitur input**, maka modelnya dapat dirumuskan sebagai berikut.

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 ^ 2 + \cdots + \beta_p x_p ^ p 
$$

**Di mana:**

* $\hat{y}$ = **nilai prediksi** (variabel terikat / dependent)
* $x$ = **input** (variabel bebas / independent)
* $m$ = **kemiringan (slope)** → seberapa banyak $\hat{y}$ berubah saat $x$ naik 1 unit
* $b$ = **intersep (intercept)** → nilai $\hat{y}$ ketika $x = 0$

### 2) Multivariate Polynomial Regression

Jika terdapat **lebih dari satu fitur**, maka kombinasi polinomial antar variabel dapat ditambahkan ke dalam fungsi polinomial. Berikut contoh dari multivariate polynomial regression dengan 2 fitur, yakni $a$ dan $b$.

$$
\hat{y} = \beta_0 + \beta_1 a + \beta_2 b + \beta_3 a ^ 2 + \beta_4 a b + \beta_5 b ^ 2
$$

dengan $\beta$ adalah koefisien yang dipelajari dari data. Estimasi parameter umumnya tetap menggunakan OLS.

## Cara Kerja

1. **Kumpulkan & siapkan data**

   * Bentuk **matriks desain** \$X \in \mathbb{R}^{n \times p}\$ (n = jumlah sampel, p = jumlah fitur).
   * Tambahkan kolom 1 untuk **intercept** jika pakai konstanta (\$\beta\_0\$).
   * Rapikan data: tangani nilai hilang, standariskan skala bila perlu, dan **encode** fitur kategorikal.

2. **Transformasi Fitur**

   * Ciptakan fitur polinomial beserta interaksi antar fitur hingga derajat tertentu.

3. **Rumuskan fungsi loss (OLS)**

   * Tujuan: meminimalkan **Sum of Squared Errors (SSE)**
     \$\mathcal{L}(\beta)=\lVert y - X\beta \rVert\_2^2\$.

4. **Estimasi parameter**

   * **Solusi tertutup (Normal Equation)** jika \$X^\top X\$ bisa diinvers:
     \$\hat{\beta}=(X^\top X)^{-1}X^\top y\$.
   * Praktik umum (mis. `scikit-learn`): gunakan **QR/SVD** (lebih stabil) daripada invers langsung.
   * **Ridge (L2)**:
     \$\hat{\beta}\_{\text{ridge}}=(X^\top X+\alpha I)^{-1}X^\top y\$ (lebih stabil saat fitur saling berkorelasi).
   * Skala besar: **(Stochastic) Gradient Descent** sampai konvergen.

5. **Prediksi**

   * Setelah \$\hat{\beta}\$ didapat: \$\hat{y}=X\_{\text{baru}}\hat{\beta}\$.

## Kelebihan
* **Menangkap pola non-linear**: mampu merepresentasikan hubungan yang lebih kompleks.
* **Sederhana**: hanya perlu menambahkan transformasi fitur.
* **Kompatibel dengan Linear Regression**: setelah transformasi, tetap dapat diproses dengan algoritma regresi linear.

## Kekurangan
* **Overfitting**: derajat polinomial yang terlalu tinggi membuat model overfit terhadap data latih.
* **Ekstrapolasi buruk**: prediksi di luar rentang data dapat bersifat tidak realistis.
* **Sensitif terhadap skala**: nilai polinomial bisa besar, sehingga perlu dilakukan standardisasi fitur.
* **Kurang interpretatif**: semakin tinggi orde, semakin sulit menjelaskan arti setiap parameter.

## Implementasi

Berikut adalah cara mengimplementasikan Linear Regression dengan library `scikit-learn`.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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

# Transformasi polinomial (orde 2)
poly = PolynomialFeatures(
   degree=2,               # Derajat polinomial
   interaction_only=False, # Hanya menggunakan interaksi saja
   include_bias=False      # Menambahkan bias
)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Inisialisasi & melatih model
model = LinearRegression() 
model.fit(X_train_poly, y_train)

# Prediksi nilai target untuk data uji
y_pred = model.predict(X_test_poly)
print(y_pred)
```

## Referensi
- [Scikit-Learn - Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Scikit-Learn - Polynomial Features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)