# Artificial Neural Network (ANN)

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Definisi](#definisi)
- [Cara Kerja](#cara-kerja)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Definisi

**Artificial Neural Network (ANN)** merupakan sebuah algoritma _supervised learning_ yang dapat digunakan untuk masalah regresi dan klasifikasi. Berbeda dengan model lainnya, ANN adalah sebuah model yang meniru cara kerja neuron dalam otak manusia, di mana banyak unit sederhana (neuron) saling terhubung dan bekerja sama untuk mempelajari pola dari data.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20230410104038/Artificial-Neural-Networks.webp" />

Secara matematis, ANN dapat dirumuskan menjadi seperti berikut:

$$ f : \mathbb{R}^m \to \mathbb{R}^n $$

dimana $m$ adalah ukuran input dan $n$ adalah ukuran output.

Komponen dari ANN meliputi:

### Neuron

Neuron merupakan sebuah unit dasar dalam neural network. Setiap neuron menerima satu atau lebih sinyal masukan $x_i$ dan melakukan penjumlahan berbobot dengan masukan tersebut. Cara kerjanya mirip dengan Linear Regression, yakni:

$$ y = \sum w_i x_i + b $$

dimana $x_i$ merupakan input ke-$i$, $w_i$ adalah bobot dari input $x_i$, dan $b$ adalah konstanta atau bias.

**Contoh Sederhana**

Misalkan sebuah neuron menerima dua input, $x_1 = 2$ dan $x_2 = 3$, dengan bobot $w_1 = 0.5$ dan $w_2 = -1$, serta bias $b = 1$. Maka keluaran neuron tersebut (sebelum melewati activation) adalah:

$$ y = (0.5 \times 2) + (-1 \times 3) + 1 = 1 - 3 + 1 = -1 $$

Dalam neuron, parameter $w$ dan $b$ merupakan hal yang akan dipelajari dan disempurnakan oleh model selama proses training, agar keluaran neuron sesuai dengan pola yang ingin dipelajari dari data.

### Layer

Layer merupakan kumpulan neuron pada tahap yang sama. Sehingga nilai keluaran setiap neuron dapat dirumuskan seperti berikut.

$$ y_j = \sum w_{ij} x_i + b_j $$

dimana $y_j$ adalah keluaran neuron ke-$j$, $w_{ij}$ merupakan bobot dari input ke-$i$ yang terhubung pada neuron ke-$j$, dan $b_j$ adalah konstanta atau bias pada neuron ke-$j$.

Terdapat berbagai jenis layer serta tujuannya, yakni:

- **Input**: Mendapatkan fitur mentah.
- **Hidden**: Mentransformasi fitur.
- **Output**: Memberikan prediksi akhir.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20240601001059/FNN.jpg" />

### Activation

Activation memperbolehkan model untuk mempelajari data yang bersifat nonlinear dengan "mengaktivasikan" keluaran dari neuron. Activation dapat dirumuskan seperti berikut.

$$ z = f(x) $$

dimana $f(x)$ merupakan fungsi aktivasi. Beberapa contoh dari fungsi aktivasi meliputi:

<img width="900" height="700" alt="image" src="https://github.com/user-attachments/assets/9fb0de86-5ea9-48d0-ba6b-f732c9725e2a" />

**Mengapa Activation Diperlukan?**

Tanpa fungsi aktivasi, keluaran tiap neuron hanyalah kombinasi linear dari inputnya (`w⋅x + b`). Jika neuron-neuron ini ditumpuk menjadi banyak layer tanpa activation, hasil akhirnya tetap saja berupa kombinasi linear, sebab penjumlahan berbobot dari kombinasi linear tetaplah kombinasi linear. Artinya, sebanyak apa pun layer yang ditambahkan, model hanya akan sekuat regresi linear biasa dan tidak mampu mempelajari pola yang non-linear (misalnya memisahkan data yang berbentuk lingkaran di dalam lingkaran).

Fungsi aktivasi (seperti ReLU atau sigmoid) menyisipkan non-linearitas di antara layer-layer tersebut, sehingga ANN benar-benar mampu memodelkan hubungan yang kompleks dan non-linear, bukan sekadar garis lurus.

### Neural Network

Neural network merupakan sebuah model yang terdiri dari berbagai layer. Masukan akan dikirimkan ke layer pertama dan diteruskan hingga ke layer paling akhir. Hasil dari layer paling akhir merupakan prediksi dari model ANN.

## Cara Kerja

1. **Kumpulkan & siapkan data**

   * Bentuk **matriks desain** $X \in \mathbb{R}^{n \times p}$ (n = jumlah sampel, p = jumlah fitur).
   * Tambahkan kolom 1 untuk **intercept** jika memakai konstanta ($\beta_0$).
   * Rapikan data: tangani nilai hilang, standariskan skala bila perlu, dan **encode** fitur kategorikal.

2. **Hitung Loss**

   Hitung loss sesuai dengan tujuan model (regresi atau klasifikasi). Loss ini mengukur seberapa jauh prediksi model dari nilai target/label sebenarnya; semakin kecil loss, semakin baik prediksi model.

3. **Estimasi parameter**

   Lakukan _backpropagation_ dari gradien yang didapatkan dari loss. Secara intuitif, backpropagation menghitung seberapa besar kontribusi kesalahan (error) di output terhadap tiap bobot di layer-layer sebelumnya, dengan merambatkan error tersebut dari output kembali ke arah input. Setelah gradien didapatkan, dilakukan optimisasi parameter untuk memperkecil error tersebut. Beberapa metode optimisasi parameter meliputi:

   * **Stochastic Gradient Descent**: Memperbarui bobot secara bertahap menggunakan gradien.
   * **Limited-memory BFGS (LBFGS)**: Optimizer berbasis quasi-Newton method, lebih stabil dan konvergen lebih cepat untuk dataset kecil.
   * **Adam**: Optimizer adaptif yang menyesuaikan learning rate berdasarkan rata-rata dan variansi gradien.

4. **Prediksi**

   Teruskan masukan dari layer paling awal hingga paling akhir (proses ini disebut *forward pass*) untuk mendapatkan prediksi.

## Kelebihan
- **Generalization**

   ANN dapat melakukan generalisasi dari data latih ke data yang belum pernah dilihat, sehingga mampu menangani tugas baru tanpa retraining penuh.

- **Parallel Processing**

   Mampu memproses beberapa input secara bersamaan, efisien untuk tugas yang memerlukan komputasi paralel.

- **Fault Tolerance / Robustness**

   Tetap bisa berfungsi meski terdapat beberapa komponen yang gagal, karena pemrosesan tersebar di banyak node.

- **Non-linear Problem Solving**

   ANN dapat memodelkan hubungan kompleks dan non-linear antara input dan output, berkat adanya fungsi aktivasi.

## Kekurangan
- **Memerlukan Data yang Banyak**

   Untuk menghasilkan model yang akurat, ANN biasanya membutuhkan dataset yang besar dan berkualitas tinggi.

- **Interpretability / Black Box**

   Sulit memahami dasar keputusan yang dibuat ANN, sehingga tidak selalu dapat diandalkan, terutama di bidang seperti kesehatan dan keuangan.

- **Memerlukan Waktu dan Sumber Daya Tinggi**

   Pelatihan ANN bisa sangat lambat dan membutuhkan komputasi tinggi, terutama untuk jaringan besar dan data yang banyak.

- **Overfitting**

   Model dapat menghafal data latih sehingga buruk dalam generalisasi. Hal ini mengurangi performa pada data nyata yang belum pernah dilihat.

## Implementasi

Berikut adalah cara mengimplementasikan ANN untuk klasifikasi dengan library `scikit-learn`.

```python
from sklearn.neural_network import MLPClassifier

# Data train
X_train = [
    [1.2, 2.3, 0.7],
    [2.1, 1.8, 1.5],
    [3.4, 3.1, 0.2],
    [4.7, 2.4, 1.1],
    [5.5, 5.2, 0.9],
    [6.3, 3.7, 1.4],
    # ...
]
y_train = [0, 0, 1, 0, 1, 1]

# Data uji / test
X_test = [
    [2.0, 2.1, 1.0],
    [5.8, 4.9, 1.3]
    # ...
]

# Inisialisasi & melatih model
model = MLPClassifier(
   hidden_layer_sizes=(5,),   # Ukuran hidden layer
   activation='relu',         # Fungsi aktivasi
   solver='adam',             # Metode optimisasi
   max_iter=200,              # Jumlah iterasi maksimum
   random_state=42,           # Random state
)
model.fit(X_train, y_train)

# Prediksi nilai target untuk data uji
y_pred = model.predict(X_test)
print(y_pred)
```

## Referensi
- [Scikit-Learn - Neural Network (Supervised)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised)
- [GeeksforGeeks (Artifical Neural Networks and its Applications)](https://www.geeksforgeeks.org/artificial-intelligence/artificial-neural-networks-and-its-applications/)
