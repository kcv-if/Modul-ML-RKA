# Linear Regression

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Definisi](#definisi)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Definisi

**Linear Regression** adalah algoritma regresi **Supervised Learning** yang digunakan untuk memprediksi nilai output berdasarkan hubungan linear dengan input. Model ini mengasumsikan bahwa ada hubungan lurus antara input dan output, di mana output akan berubah secara proporsional terhadap perubahan input.

## Kelebihan

## Kekurangan

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


