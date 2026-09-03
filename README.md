
<div align="center">

# Modul Machine Learning (Rekayasa Kecerdasan Artifisial)


<br/>

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

<br/>

</div>

<p align="justify">
Repositori ini berisi kumpulan modul pembelajaran machine learning yang disusun sebagai referensi bagi mahasiswa program studi Rekayasa Kecerdasan Artifisial. Setiap modul mencakup penjelasan konseptual, implementasi dengan Python, dan contoh kasus untuk masing-masing algoritma.
</p>

<p align="justify">
Cakupan materi mengikuti taksonomi umum machine learning: supervised learning, unsupervised learning, deep learning, reinforcement learning, dan deployment model.
</p>

---

## Daftar Isi

- [Struktur Repositori](#struktur-repositori)
- [Materi](#materi)
  - [Supervised Learning](#supervised-learning)
  - [Unsupervised Learning](#unsupervised-learning)
  - [Deep Learning](#deep-learning)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Deployment](#deployment)
- [Referensi Utama](#referensi-utama)

---

## Struktur Repositori

```
Modul-ML-RKA/
├── Supervised Learning/
│   ├── README.md
│   ├── LinearRegression.md
│   ├── PolynomialRegression.md
│   ├── LassoRidgeRegression.md
│   ├── LogisticRegression.md
│   ├── KNN.md
│   ├── SVM.md
│   ├── SVR.md
│   ├── DecisionTreeClassifier.md
│   ├── DecisionTreeRegressor.md
│   ├── ANN.md
│   └── NaiveBayes/
│       └── NaiveBayes.md
├── Unsupervised Learning/
│   ├── README.md
│   ├── K-Means.md
│   ├── Hierarchical.md
│   ├── DBSCAN.md
│   └── BIRCH.md
├── Deep Learning/
│   ├── README.md
│   ├── ANN.md
│   └── CNN.md
├── Reinforcement Learning/
│   └── RL.md
└── Deployment/
    └── deployment.md
```

---

## Materi

### Supervised Learning

<p align="justify">
Supervised learning adalah paradigma pembelajaran mesin yang menggunakan data berlabel untuk melatih model. Model belajar memetakan hubungan antara fitur input dan target output sehingga dapat membuat prediksi pada data baru yang belum pernah dilihat sebelumnya.
</p>

Materi dibagi menjadi dua kategori tugas:

**Klasifikasi**

| Algoritma | Deskripsi Singkat |
|---|---|
| [K-Nearest Neighbors](Supervised%20Learning/KNN.md) | Klasifikasi berdasarkan kedekatan jarak dengan tetangga terdekat |
| [Naive Bayes](Supervised%20Learning/NaiveBayes/NaiveBayes.md) | Klasifikasi probabilistik berbasis teorema Bayes |
| [Logistic Regression](Supervised%20Learning/LogisticRegression.md) | Model linear untuk prediksi probabilitas kelas |
| [Decision Tree](Supervised%20Learning/DecisionTreeClassifier.md) | Klasifikasi dengan struktur pohon keputusan |
| [Support Vector Machine](Supervised%20Learning/SVM.md) | Pemisahan kelas dengan hyperplane optimal |
| [Artificial Neural Network](Supervised%20Learning/ANN.md) | Klasifikasi dengan jaringan saraf tiruan berlapis |

**Regresi**

| Algoritma | Deskripsi Singkat |
|---|---|
| [Linear Regression](Supervised%20Learning/LinearRegression.md) | Prediksi nilai kontinu dengan fungsi linear |
| [Polynomial Regression](Supervised%20Learning/PolynomialRegression.md) | Perluasan regresi linear untuk hubungan non-linear |
| [Ridge dan Lasso Regression](Supervised%20Learning/LassoRidgeRegression.md) | Regresi dengan regularisasi untuk mengurangi overfitting |
| [Decision Tree Regressor](Supervised%20Learning/DecisionTreeRegressor.md) | Prediksi nilai kontinu dengan pohon keputusan |
| [Support Vector Regression](Supervised%20Learning/SVR.md) | Adaptasi SVM untuk tugas regresi |

Dokumentasi lengkap tersedia di [Supervised Learning/README.md](Supervised%20Learning/README.md).

---

### Unsupervised Learning

<p align="justify">
Unsupervised learning adalah paradigma pembelajaran mesin yang bekerja pada data tanpa label. Algoritma berusaha menemukan pola, struktur, atau pengelompokan yang tersembunyi di dalam data secara mandiri.
</p>

**Clustering**

| Algoritma | Deskripsi Singkat |
|---|---|
| [K-Means](Unsupervised%20Learning/K-Means.md) | Pengelompokan data ke dalam k klaster berdasarkan centroid |
| [Hierarchical Clustering](Unsupervised%20Learning/Hierarchical.md) | Pengelompokan dengan struktur hierarki dendogram |
| [DBSCAN](Unsupervised%20Learning/DBSCAN.md) | Pengelompokan berbasis densitas, tahan terhadap noise |
| [BIRCH](Unsupervised%20Learning/BIRCH.md) | Pengelompokan inkremental untuk dataset berskala besar |

Dokumentasi lengkap tersedia di [Unsupervised Learning/README.md](Unsupervised%20Learning/README.md).

---

### Deep Learning

<p align="justify">
Deep learning adalah subbidang machine learning yang menggunakan jaringan saraf tiruan berlapis dalam (deep neural networks). Pendekatan ini memungkinkan model untuk mempelajari representasi fitur bertingkat secara otomatis dari data mentah tanpa memerlukan rekayasa fitur manual.
</p>

| Arsitektur | Deskripsi Singkat |
|---|---|
| [Artificial Neural Network](Deep%20Learning/ANN.md) | Jaringan saraf tiruan dasar dengan lapisan tersembunyi |
| [Convolutional Neural Network](Deep%20Learning/CNN.md) | Arsitektur untuk pemrosesan data grid seperti citra |

Dokumentasi lengkap tersedia di [Deep Learning/README.md](Deep%20Learning/README.md).

---

### Reinforcement Learning

<p align="justify">
Reinforcement learning adalah paradigma di mana agen belajar mengambil keputusan melalui interaksi dengan lingkungan. Agen menerima sinyal imbalan atau penalti sebagai umpan balik dan secara iteratif mengoptimalkan kebijakan tindakannya untuk memaksimalkan akumulasi imbalan jangka panjang.
</p>

Dokumentasi tersedia di [Reinforcement Learning/RL.md](Reinforcement%20Learning/RL.md).

---

### Deployment

<p align="justify">
Tahap deployment mencakup proses mengemas, menyajikan, dan mengintegrasikan model machine learning yang telah dilatih ke dalam lingkungan produksi agar dapat digunakan oleh pengguna akhir atau sistem lain.
</p>

Dokumentasi tersedia di [Deployment/deployment.md](Deployment/deployment.md).

---

## Referensi Utama

- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Goodfellow, I., Bengio, Y., dan Courville, A. (2016). *Deep Learning*. MIT Press.
- Scikit-learn Documentation. https://scikit-learn.org/stable/
- TensorFlow Documentation. https://www.tensorflow.org/
