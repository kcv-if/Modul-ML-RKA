# Support Vector Machine (SVM)

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Definisi](#definisi)
- [Cara Kerja](#cara-kerja)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Definisi
Support Vector Machine (SVM) adalah salah satu algoritma supervised learning yang sangat kuat dan serbaguna. Bayangkan kamu memiliki data dari dua kelompok yang berbeda, misalnya titik biru dan titik merah. Tujuan SVM adalah menemukan sebuah garis (atau bidang, jika datanya 3D) yang menjadi pemisah terbaik di antara kedua kelompok tersebut.

<img width="1600" height="1000" alt="image" src="https://github.com/user-attachments/assets/98487189-27ff-4959-8de6-6323c02ccc88" />

Pemisah ini disebut hyperplane. SVM tidak hanya sekadar mencari garis pemisah, tetapi mencari garis yang memiliki margin (jarak) paling lebar dari titik terdekat di setiap kelompok. Titik-titik terdekat inilah yang disebut support vectors, karena merekalah yang menentukan posisi si garis pemisah.

## Cara Kerja

## Kelebihan

## Kekurangan
* **Komputasi mahal**: proses training bisa sangat lambat pada dataset besar dengan banyak sampel maupun fitur.
* **Pemilihan kernel sulit**: performa SVM sangat bergantung pada pemilihan kernel dan parameter (C, gamma).
* **Kurang cocok untuk data berskala besar**: dibanding algoritma sederhana, SVM bisa lebih boros memori dan waktu.
* **Sulit diinterpretasikan**: tidak semudah regresi linear dalam menjelaskan pengaruh tiap fitur.
* **Sensitif terhadap noise**: terutama jika kelas tidak terpisah jelas atau banyak outlier.

## Implementasi

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
