# Artificial Neural Networks

> Berbeda dengan [ANN di Supervised Learning](../Supervised%20Learning/ANN.md), di sini ANN diimplementasikan menggunakan framework _deep learning_ (PyTorch atau TensorFlow) yang lebih fleksibel untuk membangun, melatih, dan mengoptimalkan jaringan.

## Daftar Isi

- [Konsep Umum](#konsep-umum)
- [Komponen ANN](#komponen-ann)
- [Cara Kerja](#cara-kerja)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Konsep Umum

<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*92lRrkeeE-vWVKrYELkN_g.jpeg" />

**Artificial Neural Network (ANN)**, sering juga disebut **Multilayer Perceptron (MLP)**, adalah jaringan yang terdiri dari beberapa layer neuron yang saling terhubung penuh (_fully connected_). Setiap neuron di suatu layer terhubung ke semua neuron di layer berikutnya.

ANN mempelajari pola dari data melalui dua proses: **forward propagation** (menghasilkan prediksi) dan **backward propagation** (memperbaiki bobot berdasarkan kesalahan).

## Komponen ANN

### Neuron

Setiap neuron melakukan dua langkah:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b     (jumlah berbobot + bias)
a = f(z)                              (fungsi aktivasi)
```

- **Bobot (w)**: menentukan seberapa penting suatu input
- **Bias (b)**: menggeser output agar model lebih fleksibel
- **Fungsi aktivasi (f)**: menambahkan sifat non-linear

### Fungsi Aktivasi

Tanpa fungsi aktivasi, sebanyak apa pun layer yang ditumpuk, hasilnya tetap setara dengan satu model linear.

| Fungsi | Rumus | Range | Penggunaan umum |
| --- | --- | --- | --- |
| ReLU | `max(0, z)` | [0, ∞) | Hidden layer (default) |
| Sigmoid | `1 / (1 + e⁻ᶻ)` | (0, 1) | Output klasifikasi biner |
| Tanh | `(eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)` | (−1, 1) | Hidden layer (lebih jarang dipakai saat ini) |
| Softmax | `eᶻⁱ / Σ eᶻʲ` | (0, 1), total = 1 | Output klasifikasi multikelas |
| Linear | `z` | (−∞, ∞) | Output regresi |

### Memilih Output Layer dan Loss Function

| Tugas | Neuron output | Aktivasi output | Loss |
| --- | --- | --- | --- |
| Regresi | 1 | Linear | MSE / MAE |
| Klasifikasi biner | 1 | Sigmoid | Binary Cross-Entropy |
| Klasifikasi multikelas | Jumlah kelas | Softmax | Categorical Cross-Entropy |

## Cara Kerja

1. **Inisialisasi bobot dan bias**\
   Bobot diberi nilai acak kecil di awal, lalu disesuaikan selama pelatihan.

2. **Forward propagation**\
   Input dikalikan bobot, ditambah bias, lalu dilewatkan ke fungsi aktivasi. Proses ini berlanjut layer demi layer hingga output menghasilkan prediksi.

3. **Perhitungan loss**\
   Prediksi dibandingkan dengan nilai sebenarnya menggunakan _loss function_ untuk mengukur besarnya kesalahan.

4. **Backpropagation**\
   Gradien loss terhadap setiap bobot dan bias dihitung menggunakan _chain rule_, dimulai dari layer output kembali ke layer input.

5. **Update parameter**\
   Optimizer (SGD, Adam, RMSProp) memperbarui bobot ke arah yang mengurangi loss:
   `w_baru = w_lama − learning_rate × gradien`

6. **Iterasi (epoch)**\
   Langkah 2–5 diulang selama beberapa epoch hingga loss tidak lagi turun secara signifikan.

7. **Prediksi**\
   Model dengan bobot hasil pelatihan digunakan untuk memprediksi data baru.

## Kelebihan

- **Mampu memodelkan hubungan non-linear** antar fitur yang tidak bisa ditangkap model linear
- **Fleksibel**: jumlah layer, neuron, dan fungsi aktivasi dapat disesuaikan dengan kebutuhan
- **Skalabel**: dapat dilatih pada data besar dengan bantuan GPU/TPU
- **Dasar dari arsitektur lain**: konsep ANN dipakai di CNN, RNN, maupun Transformer (misalnya pada layer _fully connected_)

## Kekurangan

- **Butuh data cukup banyak** agar tidak _overfitting_
- **Banyak hyperparameter** (jumlah layer, neuron, learning rate, epoch) yang harus di-_tuning_
- **Sensitif terhadap skala fitur**, sehingga data perlu dinormalisasi/distandarisasi terlebih dahulu
- **Sulit diinterpretasi** (_black box_)
- Pada data tabular, sering kali **kalah dari model berbasis tree** (Random Forest, XGBoost)

## Implementasi

Studi kasus: memprediksi **pendapatan penjualan es krim** berdasarkan hari dan suhu (regresi).

Arsitektur: `Input (2) → Hidden (8, ReLU) → Output (1, Linear)`

> Data di bawah hanya 10 baris untuk keperluan demonstrasi. Pada kasus nyata, data perlu dibagi menjadi _train_ dan _test_ agar evaluasi tidak bias.

### TensorFlow / Keras

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers

# Data
day = [2, 6, 1, 3, 2, 5, 7, 4, 3, 1]
temperature = [22, 33, 20, 25, 24, 30, 35, 28, 26, 21]
revenue = [1.51, 2.22, 1.37, 1.77, 1.64, 2.04, 2.42, 1.90, 1.75, 1.45]

X = np.column_stack((day, temperature)).astype("float32")
y = np.array(revenue, dtype="float32")

# Standarisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Membangun model
model = keras.Sequential([
    keras.Input(shape=(2,)),
    layers.Dense(8, activation="relu", name="hidden_layer"),
    layers.Dense(1, activation="linear", name="output_layer"),  # linear untuk regresi
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss="mse", metrics=["mae"])
model.summary()

# Training
history = model.fit(X_scaled, y, epochs=200, batch_size=len(X), verbose=0)

# Plot loss
plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

# Prediksi & evaluasi
predictions = model.predict(X_scaled).flatten()
print("Prediksi:", predictions)
print("MSE:", mean_squared_error(y, predictions))
```

### PyTorch

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)

# Data
day = [2, 6, 1, 3, 2, 5, 7, 4, 3, 1]
temperature = [22, 33, 20, 25, 24, 30, 35, 28, 26, 21]
revenue = [1.51, 2.22, 1.37, 1.77, 1.64, 2.04, 2.42, 1.90, 1.75, 1.45]

X = np.column_stack((day, temperature)).astype(np.float32)
y = np.array(revenue, dtype=np.float32)

# Standarisasi fitur & konversi ke tensor
scaler = StandardScaler()
X_tensor = torch.tensor(scaler.fit_transform(X), dtype=torch.float32)
y_tensor = torch.tensor(y).view(-1, 1)

# Membangun model
class IceCreamSalesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),   # hidden layer
            nn.ReLU(),
            nn.Linear(8, 1),   # output layer (linear untuk regresi)
        )

    def forward(self, x):
        return self.net(x)

model = IceCreamSalesModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
losses = []
for epoch in range(200):
    predictions = model(X_tensor)           # 1. forward propagation
    loss = criterion(predictions, y_tensor) # 2. hitung loss

    optimizer.zero_grad()                   # 3. reset gradien
    loss.backward()                         # 4. backpropagation
    optimizer.step()                        # 5. update bobot

    losses.append(loss.item())

# Plot loss
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

# Prediksi & evaluasi
model.eval()
with torch.no_grad():
    predictions = model(X_tensor)
print("Prediksi:", predictions.flatten().numpy())
print("MSE:", criterion(predictions, y_tensor).item())
```

**Perbedaan utama kedua framework:** Keras menyembunyikan _training loop_ di dalam `model.fit()`, sedangkan PyTorch menuliskannya secara eksplisit sehingga setiap langkah (forward → loss → backward → update) terlihat jelas.

## Referensi

- [GeeksforGeeks: Artificial Neural Networks and its Applications](https://www.geeksforgeeks.org/artificial-intelligence/artificial-neural-networks-and-its-applications/)
- [Implementing Neural Networks in TensorFlow (and PyTorch)](https://medium.com/@shreya.rao/implementing-neural-networks-in-tensorflow-and-pytorch-3c1f097e412a)
- [Dive into Deep Learning: Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/index.html)
- [TensorFlow Playground (visualisasi interaktif ANN)](https://playground.tensorflow.org/)
