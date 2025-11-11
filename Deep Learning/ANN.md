# Artificial Neural Networks

> Berbeda dengan [ANN di Supervised Learning](../Supervised%20Learning/ANN.md), implementasinya menggunakan kerangka kerja _deep learning_ (seperti PyTorch atau TensorFlow) yang memberikan fleksibilitas lebih tinggi dalam membangun, melatih, dan mengoptimalkan jaringan.

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Konsep Umum](#konsep-umum)
- [Cara Kerja](#cara-kerja)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Konsep Umum

<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*92lRrkeeE-vWVKrYELkN_g.jpeg" />

**_Artificial Neural Network_ (ANN)** dalam konteks deep learning merupakan model yang terdiri dari sejumlah lapisan (_layers_) yang mampu mempelajari pola kompleks dari data melalui _forward propagration_ dan _backward propagation_.
Setiap layer berisi neuron buatan yang bekerja secara paralel untuk mengubah representasi input menjadi bentuk yang lebih abstrak.

## Cara Kerja
1. **Inisialisasi Bobot dan Bias**\
Setiap koneksi antar neuron diberikan bobot acak awal dan setiap neuron memiliki bias. Nilai-nilai ini akan disesuaikan selama proses pelatihan agar model menghasilkan prediksi yang akurat.

2. **Forward Propagation**\
Data masukan dikalikan dengan bobot, dijumlahkan dengan bias, kemudian dilewatkan melalui fungsi aktivasi. Proses ini berlanjut dari layer input hingga output untuk menghasilkan prediksi.

3. **Perhitungan Error (Loss Computation)**\
Nilai hasil model dibandingkan dengan nilai target sebenarnya menggunakan fungsi loss untuk menghitung seberapa besar kesalahan prediksi.

4. **Backpropagation**\
Kesalahan yang dihitung digunakan untuk menyesuaikan bobot dan bias dengan cara menghitung gradien terhadap setiap parameter menggunakan turunan berantai.

5. Optimisasi Parameter\
Optimizer (misalnya SGD, Adam, atau RMSProp) memperbarui bobot dan bias berdasarkan gradien agar nilai loss berkurang secara bertahap.

6. Iterasi Pelatihan (Epochs)\
Langkah 2–5 diulangi selama beberapa epoch hingga model mencapai konvergensi atau nilai loss tidak menurun lagi secara signifikan.

7. Prediksi Akhir\
Setelah pelatihan selesai, model menggunakan bobot dan bias yang telah dioptimalkan untuk memprediksi output baru dari data yang belum pernah dilihat sebelumnya.

## Kelebihan

- **Fleksibilitas Arsitektur**: Framework deep learning memungkinkan pembuatan arsitektur jaringan yang kompleks dan custom sesuai kebutuhan
- **Kemampuan Pembelajaran Non-linear**: Dapat mempelajari hubungan non-linear antar fitur yang tidak dapat diselesaikan oleh model linier konvensional
- **Feature Learning Otomatis**: Tidak memerlukan feature engineering manual, karena jaringan dapat belajar representasi fitur secara otomatis
- **Skalabilitas**: Dapat menangani dataset yang sangat besar dengan performa yang baik ketika dilatih dengan GPU/TPU

## Kekurangan

- **Kebutuhan Data Besar**: Memerlukan dataset yang besar untuk mencapai performa optimal dan menghindari overfitting
- **Komputasi Intensif**: Membutuhkan resource komputasi yang tinggi, terutama untuk jaringan yang dalam dan dataset besar
- **Waktu Pelatihan Lama**: Proses training dapat memakan waktu berjam-jam hingga berhari-hari tergantung kompleksitas model
- **Black Box Model**: Sulit untuk menginterpretasi bagaimana model membuat keputusan, terutama pada jaringan yang sangat dalam

## Implementasi

Berikut ini adalah contoh implementasi dengan menggunakan TensorFlow.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Data
day = [2, 6, 1, 3, 2, 5, 7, 4, 3, 1]
temperature = [22, 33, 20, 25, 24, 30, 35, 28, 26, 21]
revenue = [1.51, 2.22, 1.37, 1.77, 1.64, 2.04, 2.42, 1.90, 1.75, 1.45]

# Combine features
X_train = np.column_stack((day, temperature))
y_train = np.array(revenue)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = Sequential([
    Dense(2, input_dim=2, activation='relu', name='hidden_layer'),
    Dense(1, activation='relu', name='output_layer')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

# Display model architecture
model.summary()

# Training
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    verbose=0,
    batch_size=len(X_train)
)

# Plot training loss over epochs
plt.plot(history.history['loss'])
plt.title('Model Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Make predictions
predictions = model.predict(X_train_scaled)
print("Predicted Revenues on Training Data:", predictions)

# Evaluate the model
mse = mean_squared_error(y_train, predictions)
print("Mean Squared Error on Training Data:", mse)

```

Berikut ini adalah contoh implementasi dengan menggunakan PyTorch.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Data 
day = [2, 6, 1, 3, 2, 5, 7, 4, 3, 1]
temperature = [22, 33, 20, 25, 24, 30, 35, 28, 26, 21]
revenue = [1.51, 2.22, 1.37, 1.77, 1.64, 2.04, 2.42, 1.90, 1.75, 1.45]

# Convert to numpy array
X = np.array(list(zip(day, temperature)), dtype=np.float32)
y = np.array(revenue, dtype=np.float32)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Build the neural network
class IceCreamSalesModel(nn.Module):
    def __init__(self):
        super(IceCreamSalesModel, self).__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x

model = IceCreamSalesModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    # Forward pass
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)
    losses.append(loss.item())
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot training loss
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Make predictions
with torch.no_grad():
    predictions = model(X_tensor)
print("Predicted Revenue:", predictions.numpy())

# Evaluate the model
mse_value = criterion(predictions, y_tensor).item()
print("Mean Squared Error:", mse_value)
```

## Referensi

- [GeeksforGeeks (Artifical Neural Networks and its Applications)](https://www.geeksforgeeks.org/artificial-intelligence/artificial-neural-networks-and-its-applications/)
- [Implementing Neural Networks in TensorFlow (and PyTorch)](https://medium.com/@shreya.rao/implementing-neural-networks-in-tensorflow-and-pytorch-3c1f097e412a)