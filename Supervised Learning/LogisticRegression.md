# Logistic Regression

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Definisi](#definisi)
- [Cara Kerja](#cara-kerja)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Definisi

**Logistic Regression** merupakaan algoritma klasifikasi **Supervised Learning** yang memprediksi label berdasarkan fungsi logistik atau sigmoid.

<img src="https://th.bing.com/th/id/R.78378909ca1d863ef69e3228d31e63d0?rik=bY3uCI478E54YQ&riu=http%3a%2f%2fhelloacm.com%2fwp-content%2fuploads%2f2016%2f03%2flogistic-regression-example.jpg&ehk=a5oA2TWPMyLvVF%2fx4zh8EAYHa%2f%2bA3HNwz4NdbhoEuEY%3d&risl=&pid=ImgRaw&r=0">

di mana fungsi logistik atau sigmoid dapat dirumuskan menjadi sebagai berikut.

$$
\hat y = \frac{ 1 }{ 1 + \exp \left[ - \left( \beta_0 + \sum ^ n _ { i = 1 } \beta_i x_i \right) \right] }
$$

di mana $x$ adalah masukan dengan panjang $n$ dan $\beta$ merupakan parameter yang dipelajari.

## Cara Kerja

<!-- 1. **Siapkan data**
    - Bentuk **matriks desain** $X \in \mathbb{R}^{n \times p}$ (n = jumlah sampel, p = jumlah fitur).
    - Tambahkan kolom 1 untuk **intercept** jika memakai konstanta ($\beta_0$).

2. **Hitung loss**
    
    Minimalisir **Negative Log-Likelihood (Cross Entropy Loss)**, yaitu:
    
    $$
    \begin{align*}
    - \log \left[ L( \beta ) \right ] &=  - \sum ^ n _ { i = 1 } \left [ y_i \log \hat y_i + ( 1 - y_i ) \log ( 1 - \hat y_i ) \right ] \\
    &= - \sum ^ n _ { i = 1 } \left [ y_i \log \hat y_i + \log ( 1 - \hat y_i ) - y_i \log ( 1 - \hat y_i ) \right ] \\
    &= - \sum ^ n _ { i = 1 } \left [ \log ( 1 - \hat y_i ) + y_i \log \frac{ \hat y_i }{ 1 - \hat y_i } \right ] \\
    \end{align*}
    $$ 

3. **Optimasi parameter**

    Perbarui parameter dengan gradien dari loss.
    $$ 
    \begin{align*}
    \frac{ \partial \log \left[ L ( \beta ) \right ] }{ \partial \beta } &= - \sum ^ n _ { i = 1 } \left [\frac{ 1 }{ ( 1 - \hat y_i ) \ln (10) } +  \right ] \\ 
    &= \text{magic here} \\
    &= \sum ^ n _ { i = 1 } ( y_i - \hat y_i ) x_{ij}  
    \end{align*}
    $$

4. **Prediksi**

5. **Evaluasi & Diagnostik** -->

## Kelebihan
- **Performa baik pada data linear**: Logistic Regression bekerja secara optimal apabila data relatif dapat dipisahkan secara linear. 
- **Lebih tahan terhadap overfitting pada data dengan dimensi rendah**: Logistic Regression relatif stabil, meskipun pada dataset berdimensi tinggi bisa tetap overfit. Untuk mengatasi hal tersebut dapat menggunakan teknik Regularisasi (L1/L2).
- **Interpretasi jelas**: Logistic Regression memberikan informasi tidak hanya seberapa besar pengaruh variabel (ukuran koefisien), tetapi juga arah hubungan (positif/negatif).
- **Efisien dan mudah digunakan**: Logistic Regression mudah diimplementasikan, diinterpretasikan, dan sangat efisien untuk dilatih.

## Kekurangan
- **Asumsi linearitas**: Logistic Regression mengasumsikan hubungan linier antara variabel independen dan variabel dependen, padahal data pada dunia nyata sering kali non-linear dan tidak rapi.
- **Kurang cocok untuk data berdimensi tinggi**: Jika jumlah fitur lebih banyak daripada jumlah observasi, model cenderung untuk overfit.

## Implementasi

## Referensi

- [Scikit-Learn - Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)