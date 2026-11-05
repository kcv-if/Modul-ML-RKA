# Deep Learning

## Daftar Isi

- [Pengenalan](#pengenalan)
- [Machine Learning vs Deep Learning](#machine-learning-vs-deep-learning)
- [Konsep Dasar](#konsep-dasar)
- [Arsitektur Jaringan](#arsitektur-jaringan)
- [Framework](#framework)
- [Referensi](#referensi)

## Pengenalan

**Deep Learning (DL)** adalah cabang dari machine learning yang menggunakan **jaringan saraf tiruan berlapis banyak** (_deep neural networks_) untuk mempelajari pola dari data. Kata _deep_ merujuk pada banyaknya _hidden layer_ di antara input dan output.

Keunggulan utama DL adalah kemampuannya mempelajari **representasi fitur secara bertingkat**. Contoh pada gambar wajah:

```
Piksel → Tepi & garis → Bentuk (mata, hidung) → Wajah utuh
(layer awal)            (layer tengah)            (layer akhir)
```

Setiap layer membangun fitur yang lebih abstrak dari fitur layer sebelumnya, sehingga model tidak perlu diberi tahu secara manual fitur apa yang penting.

Contoh penerapan DL: pengenalan wajah, deteksi objek, terjemahan mesin, _speech-to-text_, chatbot (LLM), hingga diagnosis citra medis.

## Machine Learning vs Deep Learning

| Aspek | Machine Learning (klasik) | Deep Learning |
| --- | --- | --- |
| Feature engineering | Dilakukan manual oleh manusia | Dipelajari otomatis oleh model |
| Kebutuhan data | Cukup dengan data kecil–menengah | Butuh data besar |
| Komputasi | Ringan, cukup CPU | Berat, umumnya butuh GPU/TPU |
| Interpretabilitas | Relatif mudah dijelaskan | Sulit (_black box_) |
| Cocok untuk | Data tabular terstruktur | Data tidak terstruktur (gambar, teks, suara) |

**Feature engineering**: pada ML klasik, manusia mengekstrak fitur terlebih dahulu sebelum masuk ke model. Pada DL, ekstraksi fitur dan klasifikasi dilakukan oleh satu jaringan.

<img width="1611" height="930" alt="ML vs DL feature engineering" src="https://github.com/user-attachments/assets/212b367e-57b8-4ab6-8c76-94c2edd44da7" />

**Jumlah data**: performa ML klasik cenderung mentok ketika data bertambah, sedangkan performa DL terus meningkat seiring bertambahnya data.

![ML vs DL jumlah data](https://images.datacamp.com/image/upload/f_auto,q_auto:best/v1542231691/data_bw94vh.png)

> **Catatan:** DL tidak selalu lebih baik. Untuk data tabular berukuran kecil–menengah, model seperti Random Forest atau XGBoost sering kali lebih akurat, lebih cepat, dan lebih mudah dijelaskan.

## Konsep Dasar

Istilah yang akan sering muncul di materi selanjutnya:

| Istilah | Penjelasan singkat |
| --- | --- |
| **Neuron** | Unit komputasi: menghitung `w·x + b`, lalu melewatkannya ke fungsi aktivasi |
| **Layer** | Kumpulan neuron. Terdiri dari _input layer_, _hidden layer_, dan _output layer_ |
| **Fungsi aktivasi** | Memberi sifat non-linear (contoh: ReLU, Sigmoid, Softmax) |
| **Loss function** | Mengukur seberapa salah prediksi (contoh: MSE untuk regresi, Cross-Entropy untuk klasifikasi) |
| **Backpropagation** | Menghitung gradien loss terhadap setiap bobot menggunakan _chain rule_ |
| **Optimizer** | Memperbarui bobot berdasarkan gradien (contoh: SGD, Adam) |
| **Epoch / Batch** | Epoch = satu putaran penuh seluruh data; batch = potongan data yang diproses sekaligus |

## Arsitektur Jaringan

Setiap arsitektur dirancang untuk jenis data tertentu:

| Arsitektur | Cocok untuk | Contoh penerapan |
| --- | --- | --- |
| [Artificial Neural Networks (ANN/MLP)](ANN.md) | Data tabular / vektor fitur | Prediksi harga, klasifikasi sederhana |
| [Convolutional Neural Networks (CNN)](CNN.md) | Data spasial (gambar, video) | Klasifikasi gambar, deteksi objek |
| Recurrent Neural Networks (RNN, LSTM, GRU) | Data sekuensial | Time series, teks |
| Transformer | Data sekuensial skala besar | LLM (GPT, Claude), terjemahan, Vision Transformer |

## Framework

Dua framework yang paling banyak digunakan:

- **PyTorch**: populer di riset dan industri, gaya penulisan kode lebih "pythonic" dan fleksibel.
- **TensorFlow / Keras**: API Keras sangat ringkas, cocok untuk membuat model dengan cepat.

Materi ini menyertakan contoh kode dalam keduanya.

## Referensi

- [IBM: What is deep learning?](https://www.ibm.com/think/topics/deep-learning)
- [Dive into Deep Learning (d2l.ai)](https://d2l.ai/)
- [Deep Learning Book (Goodfellow et al.)](https://www.deeplearningbook.org/)
- [3Blue1Brown: Neural Networks (video)](https://www.3blue1brown.com/topics/neural-networks)
