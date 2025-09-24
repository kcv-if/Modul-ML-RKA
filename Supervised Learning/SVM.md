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
Support Vector Machine (SVM) adalah salah satu algoritma supervised learning yang sangat kuat dan serbaguna. Bayangkan kamu memiliki data dari dua kelompok yang berbeda, misalnya titik biru dan titik merah. Tujuan SVM adalah menemukan sebuah garis (atau bidang jika datanya 3D / lebih) yang menjadi pemisah terbaik di antara kedua kelompok tersebut.

<img width="1443" height="811" alt="image" src="https://github.com/user-attachments/assets/68223b00-7301-4a81-9a67-d656b7737ddb" />

Pemisah ini disebut hyperplane. SVM tidak hanya sekadar mencari garis pemisah, tetapi mencari garis yang memiliki margin (jarak) paling lebar dari titik terdekat di setiap kelompok. Titik-titik terdekat inilah yang disebut support vectors, karena merekalah yang menentukan posisi si garis pemisah.

## Cara Kerja
Cara kerja SVM dapat dipecah menjadi beberapa konsep inti:

1.  **Menemukan Hyperplane Optimal**

    Dari sekian banyak garis yang bisa memisahkan dua kelas, SVM akan mencari satu garis yang paling optimal. Optimal di sini artinya garis tersebut punya *margin* yang paling jauh ke titik data terdekat dari masing-masing kelas. Dengan memaksimalkan margin, model menjadi lebih tahan terhadap kesalahan klasifikasi pada data baru.

    **Secara Matematis**

    Sebuah *hyperplane* dapat didefinisikan oleh persamaan:
    
    $$\mathbf{w} \cdot \mathbf{x} - b = 0$$
    
    * `w` adalah vektor bobot (*weight vector*), yang menentukan orientasi atau kemiringan *hyperplane*.
    * `x` adalah vektor fitur dari data yang akan diklasifikasi.
    * `b` adalah bias, yang menggeser *hyperplane* dari titik asal.
    
    Tujuan SVM adalah menemukan `w` dan `b` yang optimal. Untuk memaksimalkan margin, SVM bertujuan untuk **meminimalkan norma dari vektor w**, yaitu `||w||`. Ini secara matematis setara dengan meminimalkan `½ ||w||²` (digunakan untuk mempermudah perhitungan turunan).
    
    Jadi, tujuan optimasinya adalah:
    
    **Minimalkan** `½ ||w||²`
    
    Dengan **syarat (constraint)** bahwa semua data diklasifikasikan dengan benar:
    
    $$y_i(\mathbf{w} \cdot \mathbf{x}_i - b) \geq 1$$
    
    untuk setiap titik data `xᵢ` dengan label kelas `yᵢ` (yang bernilai -1 atau 1).


2.  **Support Vectors**

    SVM tidak peduli dengan semua titik data, melainkan hanya fokus pada titik-titik yang berada paling dekat dengan *hyperplane*. Titik-titik inilah yang paling sulit untuk diklasifikasikan dan menjadi penentu posisi *hyperplane*. Jika titik-titik ini digeser, maka *hyperplane* pun akan ikut bergeser.

    **Secara Matematis**

    *Support vectors* adalah titik-titik data yang membuat syarat (constraint) dari poin sebelumnya menjadi sebuah **persamaan yang pas (equality)**. Artinya, mereka adalah titik-titik yang terletak persis di tepi margin.
    
    $$y_i(\mathbf{w} \cdot \mathbf{x}_i - b) = 1$$
    
    Hanya titik-titik inilah yang "menopang" *hyperplane* dan margin. Dalam model SVM yang sudah terlatih, hanya *support vectors* ini yang digunakan untuk melakukan prediksi pada data baru, inilah yang membuat SVM sangat efisien dalam penggunaan memori.


3.  **Kernel Trick**

    Bagaimana jika data tidak bisa dipisahkan dengan satu garis lurus? Di sinilah keajaiban SVM muncul. Kita tidak perlu mencari garis pemisah yang melengkung dan rumit. Sebaliknya, SVM menggunakan sebuah teknik bernama "kernel trick" untuk memproyeksikan data ke dimensi yang lebih tinggi agar bisa dipisahkan secara linear.

    **Contoh Sederhana**
    
    <img width="838" height="334" alt="image" src="https://github.com/user-attachments/assets/88ec2b24-d90e-45e8-93c0-333ebdd5b877" />
    
    Bayangkan data Anda adalah dua kelompok semut (merah dan hijau) di atas selembar kertas datar 2D, di mana satu kelompok mengelilingi kelompok lainnya. Mustahil memisahkan mereka dengan satu potongan lurus. Kernel trick ini ibarat kita melipat kertas tersebut. Tiba-tiba, jika dilihat dari samping, satu kelompok semut berada di ketinggian yang berbeda dari kelompok lainnya. Sekarang, kita bisa dengan mudah menyelipkan selembar karton lurus (sebuah bidang/hyperplane di 3D) untuk memisahkan mereka. Trik ini mengubah masalah yang mustahil di 2D menjadi masalah yang mudah di 3D.

    ### Kernel RBF
    
    Kernel **RBF (Radial Basis Function)** adalah salah satu *kernel* paling kuat dan umum digunakan dalam SVM, terutama untuk menangani masalah klasifikasi yang sangat kompleks dan tidak dapat dipisahkan secara linear. Keajaiban sesungguhnya dari RBF terletak pada kemampuannya untuk melakukan transformasi data ke **ruang fitur berdimensi tak terbatas**.
    
    Bayangkan data Anda begitu rumit dan tumpang tindih sehingga menambahkan satu atau dua dimensi baru pun tidak cukup untuk memisahkannya. Kernel RBF mengatasi ini dengan memproyeksikan setiap titik data ke sebuah ruang dengan jumlah dimensi yang tak terhingga. Di dalam ruang yang luas ini, secara teoretis dijamin selalu ada sebuah *hyperplane* yang dapat memisahkan kelas-kelas data secara sempurna.

    Tentu saja, komputer tidak benar-benar menciptakan dan menghitung koordinat dalam dimensi tak terbatas. RBF menggunakan rumus sederhana untuk menghitung **skor kesamaan (similarity score)** antara dua titik, yang hasilnya setara dengan *dot product* di antara kedua titik tersebut seandainya mereka berada di ruang dimensi tak terbatas.

    Rumus Kernel RBF adalah:
    
    $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2)$$
    
    Maksud dari rumus ini adalah:
    * Skor kesamaan antara dua titik (`xᵢ` dan `xⱼ`) dihitung berdasarkan **jarak kuadrat** (`||xᵢ - xⱼ||²`) di antara keduanya.
    * Fungsi **eksponensial (`exp`)** dengan tanda negatif memastikan bahwa semakin jauh jarak antara dua titik, skor kesamaannya akan semakin mendekati 0. Sebaliknya, jika jaraknya 0 (titik yang sama), skor kesamaannya adalah 1.
    * Parameter **`γ` (gamma)** bertindak sebagai pengatur skala. Ia menentukan seberapa cepat pengaruh sebuah titik data "memudar" seiring dengan bertambahnya jarak. Nilai gamma yang besar berarti pengaruhnya sangat lokal (hanya titik terdekat yang dianggap mirip), sedangkan nilai gamma yang kecil berarti pengaruhnya lebih luas.
    
    Dengan demikian, SVM menggunakan RBF untuk mengubah masalah dari mencari batas non-linear yang rumit menjadi masalah mengukur kesamaan berbasis jarak. Ia mendapatkan kekuatan luar biasa dari ruang dimensi tak terbatas tanpa harus menanggung biaya komputasi yang mustahil.


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


## Referensi
- [Support Vector Machines Part 1 (of 3): Main Ideas!!!](https://www.youtube.com/watch?v=efR1C6CvhmE)
- [Support Vector Machines Part 2: The Polynomial Kernel (Part 2 of 3)](https://www.youtube.com/watch?v=Toet3EiSFcM)
- [Support Vector Machines Part 3: The Radial (RBF) Kernel (Part 3 of 3)](https://www.youtube.com/watch?v=Qc5IyLW_hns)
