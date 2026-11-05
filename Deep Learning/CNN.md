# Convolutional Neural Networks

## Daftar Isi

- [Pengertian](#pengertian)
- [Mengapa CNN?](#mengapa-cnn)
- [Lapisan](#lapisan)
- [Alur Arsitektur CNN](#alur-arsitektur-cnn)
- [Arsitektur Populer](#arsitektur-populer)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Pengertian

**Convolutional Neural Network (CNN)** adalah arsitektur deep learning yang dirancang untuk memproses **data berstruktur spasial** seperti gambar dan video. CNN menggunakan operasi **konvolusi** untuk mengekstraksi fitur, mulai dari fitur sederhana (tepi, warna) hingga fitur kompleks (bentuk, objek).

Konvolusi dapat diterapkan pada berbagai dimensi data:

| Dimensi | Contoh data |
| --- | --- |
| 1D | Time series, sinyal audio, teks |
| 2D | Gambar |
| 3D | Video, citra medis (CT/MRI), data volumetrik |

## Mengapa CNN?

Bayangkan gambar berukuran 224×224 piksel RGB dimasukkan ke ANN biasa. Input-nya menjadi 224 × 224 × 3 = **150.528 nilai**. Satu hidden layer berisi 1.000 neuron saja sudah membutuhkan **±150 juta bobot**. Selain itu, ANN tidak mempertimbangkan bahwa piksel yang berdekatan saling berkaitan.

CNN mengatasi masalah ini dengan dua ide:

- **Local connectivity**: setiap neuron hanya melihat sebagian kecil area gambar (sesuai ukuran kernel).
- **Parameter sharing**: kernel yang sama digeser ke seluruh gambar, sehingga jumlah bobot jauh lebih sedikit. Fitur yang sama (misalnya tepi vertikal) dapat dikenali di posisi mana pun.

## Lapisan

### Konvolusi

Kernel (filter) berukuran kecil digeser di atas input. Pada setiap posisi, dilakukan perkalian elemen demi elemen lalu dijumlahkan, menghasilkan satu nilai pada **feature map**. Satu layer konvolusi biasanya memiliki banyak kernel, dan setiap kernel mempelajari fitur yang berbeda.

![](https://poloclub.github.io/cnn-explainer/assets/figures/convlayer_detailedview_demo.gif)

Hyperparameter dalam konvolusi:

| Kernel Size | Stride | Padding |
| --- | --- | --- |
| Ukuran kernel (contoh: 3×3, 5×5) | Jarak pergeseran kernel setiap langkah | Penambahan nilai (umumnya 0) di tepi input untuk menjaga ukuran output |
| ![](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed_small.gif) | ![](https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/no_padding_strides.gif) | ![](https://kionkim.github.io/assets/full_padding_no_strides.gif) |

**Rumus ukuran output:**

```
Output = (W − K + 2P) / S + 1
```

dengan W = ukuran input, K = ukuran kernel, P = padding, S = stride.

Contoh: input 32×32, kernel 5×5, padding 0, stride 1 → (32 − 5 + 0) / 1 + 1 = **28×28**.

Setelah konvolusi, output biasanya dilewatkan ke fungsi aktivasi **ReLU**.

### Pooling

Pooling mereduksi ukuran spasial feature map dengan mengambil **nilai maksimum** (_max pooling_) atau **rata-rata** (_average pooling_) di dalam suatu jendela. Pooling tidak memiliki bobot yang dipelajari.

Manfaatnya: mengurangi komputasi dan jumlah parameter di layer berikutnya, serta membuat model lebih tahan terhadap pergeseran posisi objek yang kecil.

![](https://www.researchgate.net/publication/333593451/figure/fig2/AS:765890261966848@1559613876098/Illustration-of-Max-Pooling-and-Average-Pooling-Figure-2-above-shows-an-example-of-max.png)

### Batch Normalization

Batch normalization menormalisasi output suatu layer menggunakan **rata-rata dan variansi yang dihitung dari seluruh sampel dalam satu batch** (pada CNN, dihitung per channel). Hasilnya kemudian diskalakan ulang dengan dua parameter yang dapat dipelajari (γ dan β).

![](https://miro.medium.com/v2/resize:fit:1358/0*BeRdzOsXQba-bFKb.png)

Manfaatnya: training menjadi lebih **stabil dan cepat**, memungkinkan learning rate yang lebih besar, serta memberi sedikit efek regularisasi. Batch normalization awalnya diusulkan untuk mengatasi _internal covariate shift_ (perubahan distribusi input setiap layer selama training), meskipun penelitian selanjutnya menunjukkan bahwa manfaat utamanya berasal dari permukaan loss yang menjadi lebih "mulus".

### Dropout

Dropout secara acak mengubah output sebagian neuron menjadi 0 dengan probabilitas tertentu (contoh: `p = 0.5`) **hanya saat training**. Saat inferensi, semua neuron aktif.

![](https://nagadakos.github.io/images/dropout.gif)

Tujuannya adalah mencegah **overfitting**: model dipaksa untuk tidak bergantung pada neuron tertentu saja, sehingga fitur yang dipelajari lebih umum.

### Flatten & Fully Connected

Feature map hasil ekstraksi diubah menjadi vektor 1D (**flatten**), lalu diteruskan ke layer **fully connected** (sama seperti [ANN](ANN.md)) untuk menghasilkan output akhir, misalnya probabilitas setiap kelas menggunakan **softmax**.

![](https://www.researchgate.net/profile/Rafeek-Mamdouh/publication/352969239/figure/fig2/AS:1041742841270274@1625382256986/Diagram-Schema-of-a-basic-architecture-CNN-consists-of-two-parts-Feature-Extraction.png)

## Alur Arsitektur CNN

CNN secara umum terbagi menjadi dua bagian:

```
                ┌───────── Feature Extraction ─────────┐   ┌──── Classification ────┐
Input Gambar → [Conv → BatchNorm → ReLU → Pool] × N → Flatten → FC → Dropout → FC → Softmax
```

Semakin dalam layer-nya, ukuran spasial feature map semakin kecil, tetapi jumlah channel (fitur) semakin banyak.

## Arsitektur Populer

| Arsitektur | Tahun | Kontribusi |
| --- | --- | --- |
| LeNet-5 | 1998 | CNN pertama yang sukses, digunakan untuk klasifikasi digit tulisan tangan |
| AlexNet | 2012 | Memenangkan ImageNet dengan selisih besar; mempopulerkan ReLU, dropout, dan training dengan GPU |
| VGG | 2014 | Menunjukkan bahwa tumpukan kernel kecil (3×3) yang dalam bekerja dengan baik |
| ResNet | 2015 | _Skip connection_ yang memungkinkan jaringan sangat dalam (100+ layer) tetap bisa dilatih |

Dalam praktiknya, arsitektur-arsitektur ini sering digunakan melalui **transfer learning**: menggunakan model yang sudah dilatih pada dataset besar (seperti ImageNet), lalu di-_fine-tune_ untuk kasus kita.

## Implementasi

Contoh LeNet-5 sederhana untuk klasifikasi digit MNIST (input 1×28×28, 10 kelas) di PyTorch:

```python
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 1×28×28 → 6×28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → 6×14×14
            nn.Conv2d(6, 16, kernel_size=5),             # → 16×10×10
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → 16×5×5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # → 400
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),  # softmax sudah termasuk di nn.CrossEntropyLoss
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

Implementasi lengkap (data loading, training, dan evaluasi):

- [TensorFlow](<https://www.kaggle.com/code/danieladhitthana/hands-on-cnn-club-dev-ai-bem-fteic-tensorflow#Hands-On-CNN-Club-Dev-AI-BEM-FTEIC-(TensorFlow)>)
- [PyTorch](https://www.kaggle.com/code/danieladhitthana/hands-on-cnn-club-dev-ai-bem-fteic-pytorch)

## Referensi

- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n: Deep Learning for Computer Vision](https://cs231n.github.io/)
- [CNN Explainer (visualisasi interaktif)](https://poloclub.github.io/cnn-explainer/)
- [Dive into Deep Learning: Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html)
