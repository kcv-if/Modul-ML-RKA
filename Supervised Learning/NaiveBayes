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

<img src="https://machinelearningknowledge.ai/wp-content/uploads/2018/08/Value-of-K.gif">

## Cara Kerja
1. **Hitung prior**  
   Estimasi probabilitas awal dari setiap kelas berdasarkan frekuensi di data latih.  
   \[
   P(C_k) = \frac{\text{jumlah sampel kelas }C_k}{\text{total sampel}}
   \]

2. **Hitung likelihood per fitur**  
   - **Data kontinu** → gunakan distribusi Gaussian.  
   - **Data diskrit** → gunakan model Multinomial.  
   - **Data biner** → gunakan model Bernoulli.  

3. **Hitung posterior**  
   \[
   P(C_k \mid x) \propto P(C_k) \times \prod_i P(x_i \mid C_k)
   \]  
   Pilih kelas dengan nilai posterior terbesar.

4. **Laplace smoothing**  
   Menghindari probabilitas nol jika suatu fitur tidak muncul pada data training.

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
