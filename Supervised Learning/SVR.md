# Support Vector Regression (SVR)

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Definisi](#definisi)
- [Cara Kerja](#cara-kerja)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Perbandingan SVM vs SVR](#perbandingan-svm-vs-svr)
- [Referensi](#referensi)

## Definisi
Support Vector Regression (SVR) adalah turunan dari Support Vector Machine (SVM) yang digunakan untuk **regresi**, bukan klasifikasi.  
Alih-alih mencari garis pemisah antar kelas, SVR berusaha menemukan sebuah fungsi (garis/kurva) yang dapat memprediksi nilai keluaran (output) secara **kontinu**.  

Perbedaannya dengan regresi linear biasa adalah SVR tidak hanya fokus meminimalkan error, tapi juga menjaga agar model tetap sederhana dengan margin tertentu (ε-insensitive).

<img width="700" alt="svr" src="https://github.com/user-attachments/assets/ad1e0470-62a5-402a-8513-92edf4f76aee" />  


## Cara Kerja
1. **Epsilon-Insensitive Tube**  
   SVR memperkenalkan konsep **ε-tube**. Selama prediksi masih berada dalam jarak ε dari nilai asli, itu dianggap tidak error. Hanya prediksi yang berada di luar margin ini yang dihitung error.

   <img width="700" alt="epsilon-tube" src="https://scikit-learn.org/stable/_images/sphx_glr_plot_svm_regression_001.png" />  

   - Titik dalam tube (±ε) → tidak dihukum.  
   - Titik di luar tube → dihukum (jadi support vectors).

2. **Fungsi Tujuan (Objective Function)**  
   Sama seperti SVM, SVR mencari **hyperplane** optimal berupa garis regresi:  

   $$f(x) = w \cdot x + b$$  

   Optimisasi:  
   - **Minimalkan** `½ ||w||²` (agar model sederhana).  
   - Dengan constraint bahwa prediksi berada dalam margin `ε` atau diberi penalti jika melampaui.  

3. **Support Vectors**  
   Hanya titik data yang berada di luar ε-tube yang berpengaruh pada model.
   <img width="1426" height="803" alt="image" src="https://github.com/user-attachments/assets/a962a081-e3af-47a3-9946-7b7d8ed1dc47" />

## Kelebihan
* **Tahan terhadap overfitting** berkat prinsip margin.  
* **Fleksibel** dengan pilihan kernel (linear, polynomial, RBF, dll).  
* **Efektif di data berdimensi tinggi**, sama seperti SVM.  
* **Kontrol error dengan ε**: bisa mengatur toleransi error sesuai kebutuhan.  

## Kekurangan
* **Lebih lambat** dibanding regresi linear biasa, terutama untuk dataset besar.  
* **Sulit tuning**: performa sangat bergantung pada parameter (`C`, `ε`, `gamma`).  
* **Tidak seintuitif regresi linear** dalam interpretasi koefisien.  

## Implementasi

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Dataset contoh: California housing (regresi)
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inisialisasi model SVR dengan kernel RBF
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

# Training
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
print("MSE :", mean_squared_error(y_test, y_pred))
print("R^2 :", r2_score(y_test, y_pred))
```

## Referensi
- https://scikit-learn.org/stable/modules/svm.html#regression
- https://youtu.be/kPw1IGUAoY8?si=JTKJsCn3LxprXQr2
- https://youtu.be/EESZtSOdhEQ?si=TwNnS3wgg00xBG9c
