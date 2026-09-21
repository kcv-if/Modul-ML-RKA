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
Support Vector Machine (SVM) adalah salah satu algoritma supervised learning yang sangat kuat dan serbaguna. Bayangkan Anda memiliki data dari dua kelompok yang berbeda, misalnya titik biru dan titik merah. Tujuan SVM adalah menemukan sebuah garis (atau bidang jika datanya 3D / lebih) yang menjadi pemisah terbaik di antara kedua kelompok tersebut.

<img width="1443" height="811" alt="image" src="https://github.com/user-attachments/assets/68223b00-7301-4a81-9a67-d656b7737ddb" />

Pemisah ini disebut hyperplane. SVM tidak hanya sekadar mencari garis pemisah, tetapi mencari garis yang memiliki margin (jarak) paling lebar dari titik terdekat di setiap kelompok. Titik-titik terdekat inilah yang disebut support vectors, karena merekalah yang menentukan posisi garis pemisah tersebut.

## Cara Kerja
Cara kerja SVM dapat dipecah menjadi beberapa konsep inti:

### 1. Menemukan Hyperplane Optimal

Dari sekian banyak garis yang bisa memisahkan dua kelas, SVM akan mencari satu garis yang paling optimal. Optimal di sini artinya garis tersebut memiliki *margin* yang paling jauh ke titik data terdekat dari masing-masing kelas. Dengan memaksimalkan margin, model menjadi lebih tahan terhadap kesalahan klasifikasi pada data baru.

**Secara Matematis**

Sebuah *hyperplane* dapat didefinisikan oleh persamaan:

$$\mathbf{w} \cdot \mathbf{x} - b = 0$$

* `w` adalah vektor bobot (*weight vector*), yang menentukan orientasi atau kemiringan *hyperplane*.
* `x` adalah vektor fitur dari data yang akan diklasifikasi.
* `b` adalah bias, yang menggeser *hyperplane* dari titik asal.

Di kedua sisi hyperplane ini terdapat dua garis batas margin yang sejajar dengannya:

$$\mathbf{w} \cdot \mathbf{x} - b = 1 \quad \text{(batas kelas positif)}$$
$$\mathbf{w} \cdot \mathbf{x} - b = -1 \quad \text{(batas kelas negatif)}$$

Jarak antara kedua garis batas ini (lebar margin) dapat dihitung secara geometris, dan hasilnya adalah:

$$\text{Margin} = \frac{2}{\|\mathbf{w}\|}$$

**Mengapa Meminimalkan ‖w‖?**

Perhatikan rumus margin di atas. Margin **berbanding terbalik** dengan `‖w‖`:

| ‖w‖ | Margin (2/‖w‖) |
|---|---|
| 1 | 2 (lebar) |
| 2 | 1 |
| 4 | 0.5 (sempit) |

Semakin kecil `‖w‖`, semakin lebar marginnya. Karena tujuan SVM adalah **memaksimalkan margin**, hal ini secara matematis setara dengan **meminimalkan `‖w‖`**. Keduanya adalah persoalan optimasi yang sama, hanya dibalik arah pandangnya saja.

Tujuan SVM adalah menemukan `w` dan `b` yang optimal, yaitu dengan meminimalkan norma dari vektor w, yaitu `‖w‖`. Ini secara matematis setara dengan meminimalkan `½ ‖w‖²`. Bentuk ini digunakan karena dua alasan teknis:

1. **Kuadrat (‖w‖²):** norma `‖w‖` melibatkan akar kuadrat, yang menyulitkan proses penurunan (diferensiasi). Karena `‖w‖` dan `‖w‖²` sama-sama fungsi yang naik monoton, nilai `w` yang meminimalkan salah satunya juga meminimalkan yang lain, sehingga menggantinya tidak mengubah solusi, hanya mempermudah perhitungan.
2. **Faktor ½:** ketika `‖w‖²` diturunkan, akan muncul faktor 2 (aturan turunan pangkat). Faktor ½ di depan sengaja ditambahkan agar faktor 2 tersebut saling meniadakan, sehingga hasil turunannya lebih rapi.

Jadi, tujuan optimasinya adalah:

**Minimalkan** `½ ‖w‖²`

Dengan **syarat (constraint)** bahwa semua data diklasifikasikan dengan benar:

$$y_i(\mathbf{w} \cdot \mathbf{x}_i - b) \geq 1$$

untuk setiap titik data `xᵢ` dengan label kelas `yᵢ` (yang bernilai -1 atau 1).

**Memahami Constraint Ini**

Nilai `w⋅x - b` adalah "skor posisi" suatu titik relatif terhadap hyperplane: positif berarti di satu sisi, negatif berarti di sisi lainnya. Mengalikannya dengan label `yᵢ` (+1 atau -1) menghasilkan sebuah trik yang berguna:

| Kondisi | Hasil `yᵢ(w⋅xᵢ - b)` |
|---|---|
| Klasifikasi **benar** (titik ada di sisi yang sesuai labelnya) | selalu **positif** |
| Klasifikasi **salah** | selalu **negatif** |

Ini berlaku untuk kelas positif maupun negatif sekaligus, tanpa perlu menulis dua rumus terpisah. Misalnya, untuk titik kelas negatif (`yᵢ = -1`) yang tepat berada di tepi margin, skor mentahnya adalah -1, tetapi setelah dikalikan `yᵢ`, hasilnya `(-1) × (-1) = 1`, tetap memenuhi `≥ 1`.

Syaratnya dibuat `≥ 1`, bukan sekadar `≥ 0`, karena SVM tidak hanya menuntut klasifikasi yang benar, tetapi juga menuntut adanya **jarak aman (margin)** dari hyperplane. Titik yang hasilnya tepat `= 1` adalah support vectors (menempel di tepi margin); titik yang hasilnya `> 1` berada lebih jauh dan aman.

### 2. Support Vectors

SVM tidak peduli dengan semua titik data, melainkan hanya fokus pada titik-titik yang berada paling dekat dengan *hyperplane*. Titik-titik inilah yang paling sulit untuk diklasifikasikan dan menjadi penentu posisi *hyperplane*. Jika titik-titik ini digeser, maka *hyperplane* pun akan ikut bergeser; titik-titik lain yang jauh dari garis pemisah dapat digeser tanpa mengubah posisi hyperplane sama sekali.

**Secara Matematis**

*Support vectors* adalah titik-titik data yang membuat syarat (constraint) dari poin sebelumnya menjadi sebuah **persamaan yang pas (equality)**. Artinya, mereka adalah titik-titik yang terletak persis di tepi margin.

$$y_i(\mathbf{w} \cdot \mathbf{x}_i - b) = 1$$

Hanya titik-titik inilah yang "menopang" *hyperplane* dan margin. Dalam model SVM yang sudah terlatih, hanya *support vectors* ini yang digunakan untuk melakukan prediksi pada data baru, inilah yang membuat SVM sangat efisien dalam penggunaan memori.

### 3. Kernel Trick

Bagaimana jika data tidak bisa dipisahkan dengan satu garis lurus? Di sinilah keajaiban SVM muncul. Kita tidak perlu mencari garis pemisah yang melengkung dan rumit. Sebaliknya, SVM menggunakan sebuah teknik bernama "kernel trick" untuk memproyeksikan data ke dimensi yang lebih tinggi agar bisa dipisahkan secara linear.

**Contoh Sederhana**

<img width="838" height="334" alt="image" src="https://github.com/user-attachments/assets/88ec2b24-d90e-45e8-93c0-333ebdd5b877" />

Bayangkan data Anda adalah dua kelompok semut (merah dan hijau) di atas selembar kertas datar 2D, di mana satu kelompok mengelilingi kelompok lainnya. Mustahil memisahkan mereka dengan satu potongan lurus. Kernel trick ini ibarat kita melipat kertas tersebut. Tiba-tiba, jika dilihat dari samping, satu kelompok semut berada di ketinggian yang berbeda dari kelompok lainnya. Sekarang, kita bisa dengan mudah menyelipkan selembar karton lurus (sebuah bidang/hyperplane di 3D) untuk memisahkan mereka. Trik ini mengubah masalah yang mustahil di 2D menjadi masalah yang mudah di 3D.

**Mengapa Disebut "Trik"?**

Proses training SVM, jika ditelusuri lebih dalam, hanya membutuhkan satu jenis operasi: **dot product** antar titik data (`x ⋅ x'`). Jika ingin memisahkan data di dimensi yang lebih tinggi, cara "jujur" adalah: (1) transformasikan setiap titik ke dimensi tinggi menggunakan suatu fungsi `φ(x)`, lalu (2) hitung dot product dari titik-titik yang sudah ditransformasikan tersebut. Namun jika dimensi tujuannya tak terbatas, langkah pertama ini mustahil dihitung komputer.

Kernel trick adalah jalan pintas: untuk fungsi kernel tertentu (termasuk RBF), hasil dot product di dimensi tinggi (`φ(xᵢ) ⋅ φ(xⱼ)`) ternyata bisa dihitung langsung dari titik data asli, tanpa perlu benar-benar melakukan transformasi tersebut. Artinya:

$$K(x_i, x_j) = \varphi(x_i) \cdot \varphi(x_j)$$

Kita mendapatkan hasil yang seolah-olah dihitung di dimensi tak terbatas, tanpa harus menanggung biaya komputasi untuk benar-benar bekerja di sana.

### Kernel RBF

Kernel **RBF (Radial Basis Function)** adalah salah satu *kernel* paling kuat dan umum digunakan dalam SVM, terutama untuk menangani masalah klasifikasi yang sangat kompleks dan tidak dapat dipisahkan secara linear. Keajaiban sesungguhnya dari RBF terletak pada kemampuannya untuk melakukan transformasi data ke **ruang fitur berdimensi tak terbatas**.

Bayangkan data Anda begitu rumit dan tumpang tindih sehingga menambahkan satu atau dua dimensi baru pun tidak cukup untuk memisahkannya. Kernel RBF mengatasi ini dengan memproyeksikan setiap titik data ke sebuah ruang dengan jumlah dimensi yang tak terhingga. Di dalam ruang yang luas ini, secara teoretis dijamin selalu ada sebuah *hyperplane* yang dapat memisahkan kelas-kelas data secara sempurna.

Tentu saja, komputer tidak benar-benar menciptakan dan menghitung koordinat dalam dimensi tak terbatas. RBF menggunakan rumus sederhana untuk menghitung **skor kesamaan (similarity score)** antara dua titik, yang hasilnya setara dengan *dot product* di antara kedua titik tersebut seandainya mereka berada di ruang dimensi tak terbatas.

Rumus Kernel RBF adalah:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2)$$

Maksud dari rumus ini adalah:
* Skor kesamaan antara dua titik (`xᵢ` dan `xⱼ`) dihitung berdasarkan **jarak kuadrat** (`||xᵢ - xⱼ||²`) di antara keduanya.
* Fungsi **eksponensial (`exp`)** dengan tanda negatif memastikan bahwa semakin jauh jarak antara dua titik, skor kesamaannya akan semakin mendekati 0. Sebaliknya, jika jaraknya 0 (titik yang sama), skor kesamaannya adalah 1. Sebagai gambaran, dengan `γ = 1`: jarak² = 0 menghasilkan skor 1, jarak² = 1 menghasilkan skor ≈ 0.37, dan jarak² = 4 menghasilkan skor ≈ 0.02.
* Parameter **`γ` (gamma)** bertindak sebagai pengatur skala. Ia menentukan seberapa cepat pengaruh sebuah titik data "memudar" seiring dengan bertambahnya jarak. Nilai gamma yang besar berarti pengaruhnya sangat lokal (hanya titik terdekat yang dianggap mirip, berisiko overfitting), sedangkan nilai gamma yang kecil berarti pengaruhnya lebih luas (model lebih halus, berisiko underfitting jika terlalu kecil).

Dengan demikian, SVM menggunakan RBF untuk mengubah masalah dari mencari batas non-linear yang rumit menjadi masalah mengukur kesamaan berbasis jarak. Ia mendapatkan kekuatan luar biasa dari ruang dimensi tak terbatas tanpa harus menanggung biaya komputasi yang mustahil.

**Dari Skor Kesamaan ke Keputusan Klasifikasi**

Skor kesamaan ini bukan sekadar angka lepas, melainkan bahan baku langsung dari fungsi keputusan SVM. Ketika kernel digunakan, fungsi klasifikasi untuk titik data baru `x` menjadi:

$$f(x) = \sum_{i \in \text{support vectors}} \alpha_i \, y_i \, K(x_i, x) - b$$

di mana `αᵢ` adalah bobot kepentingan tiap support vector hasil training. Cara kerjanya: titik baru `x` dibandingkan kemiripannya (`K(xᵢ, x)`) dengan setiap support vector. Kemiripan tinggi terhadap support vector berlabel positif akan "mendorong" keputusan ke arah kelas positif, begitu pula sebaliknya. Seluruh dorongan ini dijumlahkan, dan hasilnya menentukan kelas akhir dari `x`. Dengan kata lain, menghitung kernel **adalah** proses klasifikasinya, bukan langkah terpisah sebelum klasifikasi.

## Kelebihan

* **Efektif di Ruang Dimensi Tinggi**

    SVM bekerja sangat baik pada dataset dengan jumlah fitur yang sangat banyak, bahkan jika jumlah fitur lebih banyak daripada jumlah sampel data. Ini membuatnya cocok untuk klasifikasi teks atau data genomik.
* **Hemat Memori**

    Model SVM hanya menggunakan sebagian kecil dari titik data training (yaitu *support vectors*) untuk membangun keputusan. Karena tidak bergantung pada semua data, ini membuatnya sangat efisien dalam penggunaan memori.
* **Sangat Fleksibel**

    Berkat adanya *kernel trick*, SVM dapat beradaptasi dengan berbagai jenis data. Ia dapat memodelkan batas keputusan yang sangat kompleks dan non-linear (misalnya dengan kernel RBF) atau batas linear yang sederhana.
* **Kuat dan Akurat**

    Konsep memaksimalkan margin membuat SVM menjadi model yang kuat dan cenderung tidak *overfitting*, terutama pada data yang terpisah dengan jelas. Ini seringkali menghasilkan akurasi yang tinggi.

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

Catatan: parameter `C` mengatur trade-off antara margin lebar dan toleransi kesalahan klasifikasi (`C` kecil → margin lebih lebar tapi lebih toleran terhadap sedikit kesalahan; `C` besar → margin lebih ketat, cenderung meminimalkan kesalahan pada data training). Parameter `gamma='scale'` adalah nilai default yang secara otomatis menyesuaikan skala gamma berdasarkan variansi fitur.

## Referensi
- [Support Vector Machines Part 1 (of 3): Main Ideas!!!](https://www.youtube.com/watch?v=efR1C6CvhmE)
- [Support Vector Machines Part 2: The Polynomial Kernel (Part 2 of 3)](https://www.youtube.com/watch?v=Toet3EiSFcM)
- [Support Vector Machines Part 3: The Radial (RBF) Kernel (Part 3 of 3)](https://www.youtube.com/watch?v=Qc5IyLW_hns)
