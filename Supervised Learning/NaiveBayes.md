# Naive Bayes

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Definisi](#definisi)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Definisi

**Naive Bayes** adalah algoritma klasifikasi **supervised learning** yang didasarkan pada **teorema Bayes**, dengan asumsi **independensi kondisional** antar fitur (bersifat "naïf").  
Dengan kata lain, satu fitur dianggap tidak memengaruhi fitur lainnya, jika sudah diketahui kelas targetnya. Hal ini membuat perhitungan lebih sederhana dan efisien.

<img src="https://upload.wikimedia.org/wikipedia/commons/b/b4/Naive_Bayes_Classifier.gif">

## Cara Kerja
1. **Hitung prior**  
   Estimasi probabilitas awal dari setiap kelas berdasarkan frekuensi di data latih.  
   \[
   P(C_k) = \frac{\text{jumlah sampel kelas } C_k}{\text{total sampel}}
   \]

Contoh: jika terdapat 12 pesan, dengan 8 normal (N) dan 4 spam (S):

   \[
   P(N) = \frac{8}{8+4} = 0.67
   \]

   <img width="1082" height="520" alt="image" src="https://github.com/user-attachments/assets/a4f4f6b8-447d-4dc8-a86c-e76b083dcb63" />


3. **Hitung likelihood per fitur**  
   - **Data kontinu** → gunakan distribusi Gaussian.  
   - **Data diskrit** → gunakan model Multinomial.  
   - **Data biner** → gunakan model Bernoulli.

   <img width="1100" height="331" alt="image" src="https://github.com/user-attachments/assets/1f1da253-da9c-48a5-a024-1fc1bac7a8fd" />

4. **Hitung posterior**  
   \[
   P(C_k \mid x) \propto P(C_k) \times \prod_i P(x_i \mid C_k)
   \]  
   Pilih kelas dengan nilai posterior terbesar.

   <img width="730" height="372" alt="image" src="https://github.com/user-attachments/assets/0c59f10d-b849-44ba-907d-dd21a1e4e81e" />


6. **Laplace smoothing**  
   Menghindari probabilitas nol jika suatu fitur tidak muncul pada data training.
   <img width="851" height="488" alt="image" src="https://github.com/user-attachments/assets/a455bfb5-0b0f-4326-8be3-851c9587b0a1" />
   <img width="844" height="500" alt="image" src="https://github.com/user-attachments/assets/3a9b6a79-2714-471e-8b92-96ac72a45119" />


## Kelebihan

- **Cepat dan efisien**: training dan klasifikasi sangat cepat, bahkan untuk dataset besar.  
- **Kebutuhan memori rendah**: hanya perlu menyimpan statistik (prior & likelihood).  
- **Skalabilitas tinggi**: performa tetap baik meski jumlah fitur banyak.  
- **Mudah diimplementasikan**: tersedia di banyak toolkit (misalnya `scikit-learn`).  
- **Efektif dengan sedikit data latih**: masih bekerja baik meski data terbatas.  
- **Cocok untuk data berdimensi tinggi**: seperti klasifikasi teks atau analisis dokumen.  
- **Probabilistik**: memberikan nilai probabilitas untuk setiap kelas.

## Kekurangan


## Implementasi


## Referensi

- https://scikit-learn.org/stable/modules/naive_bayes.html
- https://youtu.be/O2L2Uv9pdDA?si=d0QOVtqqAe_toA0h
