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

## Definisi

## Cara Kerja


## Kelebihan

## Kekurangan
* **Komputasi mahal**: proses training bisa sangat lambat pada dataset besar dengan banyak sampel maupun fitur.
* **Pemilihan kernel sulit**: performa SVM sangat bergantung pada pemilihan kernel dan parameter (C, gamma).
* **Kurang cocok untuk data berskala besar**: dibanding algoritma sederhana, SVM bisa lebih boros memori dan waktu.
* **Sulit diinterpretasikan**: tidak semudah regresi linear dalam menjelaskan pengaruh tiap fitur.
* **Sensitif terhadap noise**: terutama jika kelas tidak terpisah jelas atau banyak outlier.

## Implementasi

Berikut adalah cara mengimplementasikan SVM dengan library `scikit-learn`.

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Dataset contoh: Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split data train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inisialisasi model SVM dengan kernel RBF
model = SVC(kernel='rbf', C=1.0, gamma='scale')

# Training
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
print("Akurasi:", accuracy_score(y_test, y_pred))
```

## Referensi
- 
