# Decision Tree Regressor

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Definisi](#definisi)
- [Tabel perbandingan](#Tabel-Perbandingan)
- [Cara Kerja](#cara-kerja)
- [Metode Pemisahan](#metode-pemisahan)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Definisi
Decision Tree adalah model prediktif berbentuk seperti pohon yang digunakan dalam machine learning untuk membantu pengambilan keputusan. Model ini memetakan berbagai pilihan dan hasil yang mungkin berdasarkan fitur-fitur dalam data.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/788afd65-ea70-4f8a-88d2-3e2e2b8ebdb0" />

#### Struktur dasar Decision Tree
- Root Node: Titik awal yang mewakili seluruh dataset.
- Branches: Jalur yang menghubungkan antar node, menunjukkan alur keputusan.
- Internal Nodes: Titik di mana keputusan dibuat berdasarkan fitur tertentu.
- Leaf Nodes: Titik akhir yang menunjukkan hasil atau prediksi akhir.

#### Tipe" Decision Tree
- **Classification Tree:**  
  Digunakan untuk memprediksi hasil kategorikal seperti spam atau bukan spam. Tipe ini membagi data berdasarkan fitur-fitur tertentu untuk mengklasifikasikan data ke dalam kategori yang telah ditentukan sebelumnya.
- **Regression Tree:**  
  Digunakan untuk memprediksi hasil kontinu seperti harga rumah. Tipe ini memberikan prediksi berupa nilai numerik berdasarkan fitur-fitur input.

## Tabel Perbandingan

| Aspek                  | Decision Tree Classifier                     | Decision Tree Regressor                           |
| ---------------------- | -------------------------------------------- | ------------------------------------------------- |
| **Output**             | Label kategori (diskrit)                     | Nilai numerik (kontinu)                           |
| **Contoh Masalah**     | Spam/Not Spam, Sakit/Sehat, Sentimen Pos/Neg | Harga rumah, Suhu udara, Nilai saham              |
| **Kriteria Split**     | Gini Impurity, Entropy                       | MSE, MAE, Poisson Deviance                        |
| **Prediksi Leaf Node** | Kelas mayoritas dari sampel di node          | Rata-rata nilai target pada node                  |
| **Tujuan Split**       | Membuat node homogen (satu kelas dominan)    | Membuat node dengan varian target sekecil mungkin |
| **Evaluasi Kinerja**   | Akurasi, Precision, Recall, F1-Score         | MSE, RMSE, MAE, R² Score                          |

## Cara Kerja
1) **Mulai dari Root Node**  
  Semua data ditempatkan di node akar.
2) **Evaluasi Fitur untuk Split.**  
  Setiap fitur diuji dengan berbagai threshold untuk melihat bagaimana data bisa dipisahkan dengan meminimalkan error.
3) **Pilih Split Terbaik.**  
  Split dipilih berdasarkan pengurangan error terbesar (misalnya: penurunan MSE).
4) **Buat cabang dan Ulangi.**  
  Proses berlanjut secara rekursif sampai kondisi berhenti tercapai (max depth, min samples, atau error minimum).
5) **Prediksi di Leaf Node.**
  Nilai prediksi di leaf node adalah rata-rata nilai target dari semua data di node tersebut.

## Metode Pemisahan
Berbeda dengan klasifikasi (yang menggunakan Gini/Entropy), regresi menggunakan ukuran error:

1. **Mean Squared Error (MSE)**
   [
   MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y})^2
   ]

   * (y_i) = nilai target sebenarnya
   * (\hat{y}) = rata-rata target pada node

   Semakin kecil MSE, semakin baik split.

2. **Mean Absolute Error (MAE)**
   Menggunakan selisih absolut daripada kuadrat. Lebih robust terhadap outlier.

3. **Poisson Deviance** (khusus untuk data count/positif)
   Cocok jika target berupa jumlah kejadian (misalnya jumlah klik).

#### Contoh Implementasi Split (Scikitlearn):
```
DecisionTreeRegressor(criterion='squared_error')  # MSE
DecisionTreeRegressor(criterion='absolute_error') # MAE
DecisionTreeRegressor(criterion='poisson')        # Poisson
```

## Kelebihan
1) **Serbaguna.**  
  Dapat dipakai untuk **klasifikasi** dan **regresi**.
2) **Tanpa Skala Fitur.**  
  Tidak perlu normalisasi/standarisasi fitur numerik terlebih dahulu.
3) **Menangani Hubungan Non-Linear.**  
  Mampu menangkap hubungan kompleks dan non-linear antara fitur dan hasil secara efektif.
4) **Interpretabilitas Tinggi.**  
  Struktur pohon yang jelas memudahkan pengguna memahami alasan di balik setiap keputusan yang diambil.

## Kekurangan
1) Overfitting , Decision Tree bisa tumbuh terlalu dalam dan rumit, menyesuaikan setiap detail termasuk noise (data yang tidak mewakili pola sebenarnya).misal jika ada 1 data outlier, pohon bisa membuat cabang khusus hanya untuk data itu = memicu overfitting
2) Perubahan kecil pada data (menambah/menghapus 1 sampel) bisa menghasilkan struktur pohon yang sangat berbeda.
3) Pembelajaran serakah (greedy) ,Pohon dibentuk dengan memilih split terbaik di tiap langkah secara lokal (greedy) sehingga tidak menjamin struktur pohon terbaik secara keseluruhan (global optimal).
4) Bias kelas , kelas mayoritas dapat mendominasi pembelahan (split), menyebabkan bias.

## Implementasi
```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Buat dataset sintetis regresi (100 sampel, 1 fitur)
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.1 

# Visualisasi dataset
plt.figure(figsize=(6, 4))
plt.scatter(X, y, color="blue", label="Data")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Synthetic Regression Dataset")
plt.legend()
plt.show()

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Buat model Decision Tree Regressor
dt_reg = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=4,
    min_samples_leaf=2,
    random_state=42
)

# Training dan Prediksi
dt_reg.fit(X_train, y_train)
y_pred = dt_reg.predict(X_test)

# Evaluasi model
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Visualisasi hasil prediksi vs data asli
X_range = np.linspace(0, 5, 500).reshape(-1, 1)
y_range_pred = dt_reg.predict(X_range)

plt.figure(figsize=(6, 4))
plt.scatter(X, y, color="blue", label="Data Asli")
plt.plot(X_range, y_range_pred, color="red", linewidth=2, label="Prediksi Tree")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Decision Tree Regression Fit")
plt.legend()
plt.show()

# Visualisasi pohon
plt.figure(figsize=(12, 8))
plot_tree(dt_reg, filled=True, rounded=True, feature_names=['x'], fontsize=10)
plt.show()
```

## Referensi
- [Scikit-Learn - DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)