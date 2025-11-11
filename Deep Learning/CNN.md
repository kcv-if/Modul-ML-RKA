# Convolutional Neural Networks

## Daftar Isi

-   [Pengertian](#daftar-isi)
-   [Lapisan](#lapisan)
-   [Implementasi](#implementasi)
-   [Referensi](#referensi)

## Pengertian

Convolution Neural Network (CNN) merupakan salah satu jenis arsitektur deep learning yang memiliki kemampuan untuk memproses data spasial, seperti citra (gambar) dan video, dengan memanfaatkan operasi konvolusi untuk mengekstraksi fitur-fitur dari data tersebut. Terdapat 3 jenis konvolusi yang biasa digunakan pada CNN, seperti:
| Dimensi | Contoh |
| ------- | ---------------- |
| 1D | Time series, NLP |
| 2D | Gambar |
| 3D | Video, mesh |

## Lapisan

Lapisan yang umumnya ditemukan dalam CNN terdiri dari:

### Konvolusi

Lapisan ini melakukan penjumlahan berbobot dari input dalam kernel yang bergeser. Hal ini memungkinkan model untuk mengekstraksi fitur penting seperti tepi, tekstur dan bentuk.

![](https://poloclub.github.io/cnn-explainer/assets/figures/convlayer_detailedview_demo.gif)

Beberapa hyperparameter dalam konvolusi meliputi:

| Kernel Size                                                                                                                                       | Stride                                                                                             | Padding                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Ukuran dari kernel                                                                                                                                | Jarak pergeseran kernel                                                                            | Penambahan data spasial untuk mempertahankan ukuran keluaran.     |
| ![](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed_small.gif) | ![](https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/no_padding_strides.gif) | ![](https://kionkim.github.io/assets/full_padding_no_strides.gif) |

### Pooling

Pooling bertujuan untuk mereduksi dimensi spasial dari gambar lewat agregasi, seperti mendapatkan nilai maksimum atau rata-rata dalam kernel. Hal ini membantu dalam mereduksi jumlah parameter model dan meringankan komputasi.

![](https://www.researchgate.net/publication/333593451/figure/fig2/AS:765890261966848@1559613876098/Illustration-of-Max-Pooling-and-Average-Pooling-Figure-2-above-shows-an-example-of-max.png)

### Batch Normalization

Batch normalization melakukan normalisasi pada dimensi batch, dimana setiap gambar dalam batch dinormalisasikan secara terpisah.

![](https://miro.medium.com/v2/resize:fit:1358/0*BeRdzOsXQba-bFKb.png)

Hal ini bertujuan untuk mengatasi _internal covariate shift_, dimana distribusi activation berubah karena distribusi data berubah.

### Dropout

Dropout secara sengaja mengubah output dari beberapa neuron dalam suatu lapisan menjadi 0.

![](https://wenkangwei.github.io/images/DL/dropout.jpg)

Hal ini bertujuan untuk menghindari overfitting dengan cara "memberhentikan" neuron-neuron tersebut untuk belajar.

### Fully Connected

Fully Connected bertujuan untuk menghubungkan hasil lapisan sebelumnya dengan lapisan output.

![](https://www.researchgate.net/profile/Rafeek-Mamdouh/publication/352969239/figure/fig2/AS:1041742841270274@1625382256986/Diagram-Schema-of-a-basic-architecture-CNN-consists-of-two-parts-Feature-Extraction.png)

## Implementasi

Berikut adalah contoh implementasi dari sebuah digit classifier (LeNet5) dalam:

-   [Tensorflow](<https://www.kaggle.com/code/danieladhitthana/hands-on-cnn-club-dev-ai-bem-fteic-tensorflow#Hands-On-CNN-Club-Dev-AI-BEM-FTEIC-(TensorFlow)>)
-   [PyTorch](https://www.kaggle.com/code/danieladhitthana/hands-on-cnn-club-dev-ai-bem-fteic-pytorch)

## Referensi

-   [Deep Learning book](https://www.deeplearningbook.org/)
-   [CS231 Deep Learning for Computer Vision](https://cs231n.github.io/)
-   [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
