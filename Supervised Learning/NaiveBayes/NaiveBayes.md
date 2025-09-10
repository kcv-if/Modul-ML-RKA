# Naive Bayes

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Definisi](#definisi)
- [Cara Kerja](#cara-kerja)
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
   
   $P(C_k) = \frac{\text{jumlah sampel kelas } C_k}{\text{total sampel}}$
   
   Contoh: jika terdapat 12 pesan, dengan 8 normal (N) dan 4 spam (S):
   
   $P(N) = \frac{8}{8+4} = 0.67$
   
   <br>
   <p>
      <img width="1082" height="520" alt="image" src="https://github.com/user-attachments/assets/a4f4f6b8-447d-4dc8-a86c-e76b083dcb63" />
   </p>
   


2. **Hitung likelihood per fitur**  
   - **Data kontinu** → gunakan distribusi Gaussian.
     #### Contoh sederhana:
      Misal klasifikasi **Sehat (N)** vs **Sakit (S)** dari **suhu tubuh** (°C).
      **Data latih**
      - N: 36.4, 36.7, 36.5 → rata-rata $\mu_N \approx 36.53$, simpangan baku $\sigma_N \approx 0.125$  
      - S: 38.0, 37.8, 38.2 → $\mu_S \approx 38.00$, $\sigma_S \approx 0.163$
      
      **Rumus likelihood Gaussian**  
      $p(x\mid C)=\dfrac{1}{\sqrt{2\pi}\,\sigma_C}\exp\!\left(-\dfrac{(x-\mu_C)^2}{2\sigma_C^2}\right)$
      
      **Contoh hitung**
      - Untuk $x = 37.0$:  
        $p(37.0\mid N)\approx \mathbf{0.0029}$,  
        $p(37.0\mid S)\approx \mathbf{1.76\times 10^{-8}}$
      - Untuk $x = 38.1$:  
        $p(38.1\mid N)\approx \mathbf{1.75\times 10^{-34}}$,  
        $p(38.1\mid S)\approx \mathbf{2.03}$

   - **Data diskrit** → gunakan model Multinomial.
     <img width="1393" height="697" alt="image" src="https://github.com/user-attachments/assets/ba58fb95-64d9-4bdb-9b96-18d3ef1ce848" />

   - **Data biner** → gunakan model Bernoulli.
     #### Contoh sederhana:
      Fitur hanya **ada (1)** atau **tidak (0)**.  
      **Fitur:** `has_link`, `has_dear`, `has_lunch`
      
      **Frekuensi hadir (1) di data latih** — per kelas ada 5 dokumen:  
      - Spam: link=4/5, dear=3/5, lunch=0/5  
      - Normal: link=1/5, dear=1/5, lunch=3/5
      
      **Smoothing Bernoulli ($\alpha=1$):**  
      $P(f{=}1\mid C)=\dfrac{n_1+1}{n_C+2}$, dan $P(f{=}0\mid C)=1-P(f{=}1\mid C)$
      
      **Prob. hadir setelah smoothing**
      - Spam: $P(\text{link}=1)=\mathbf{0.714}$, $P(\text{dear}=1)=\mathbf{0.571}$, $P(\text{lunch}=1)=\mathbf{0.143}$  
      - Normal: $P(\text{link}=1)=\mathbf{0.286}$, $P(\text{dear}=1)=\mathbf{0.286}$, $P(\text{lunch}=1)=\mathbf{0.571}$
      
      **Contoh hitung likelihood vektor fitur**  
      Untuk fitur (`link`=1, `dear`=1, `lunch`=0):  
      - $p(\mathbf{x}\mid \text{Spam})=0.714\times 0.571\times (1-0.143)=\mathbf{0.3499}$  
      - $p(\mathbf{x}\mid \text{Normal})=0.286\times 0.286\times (1-0.571)=\mathbf{0.0350}$

3. **Hitung posterior**  
   Kombinasikan prior dan likelihood untuk mendapatkan probabilitas akhir (posterior):
   
   $P(C_k \mid x) \propto P(C_k) \times \prod_i P(x_i \mid C_k)$
   
   Pilih kelas dengan nilai posterior terbesar.  
   
   Contoh perhitungan untuk pesan `"Dear Friend"`:
   
   $P(N) \times P(Dear \mid N) \times P(Friend \mid N) = 0.09$
   $P(S) \times P(Dear \mid S) \times P(Friend \mid S) = 0.01$
   
   Karena **0.09 > 0.01**, maka pesan diklasifikasikan sebagai **Normal Message (N)**.

   <img width="730" height="372" alt="image" src="https://github.com/user-attachments/assets/0c59f10d-b849-44ba-907d-dd21a1e4e81e" />

4. Laplace Smoothing

   Masalah utama pada Naive Bayes adalah ketika suatu **fitur tidak pernah muncul** dalam data latih untuk kelas tertentu.  
   
   Contoh: kata **"Lunch"** tidak pernah muncul pada pesan **Spam**.  
   Jika dihitung langsung:
   
   $P(Lunch \mid Spam) = \frac{0}{7} = 0$
   
   Hasil ini bermasalah karena:
   - Jika ada **satu fitur dengan probabilitas nol**, maka seluruh hasil perkalian posterior akan menjadi **nol**.
   - Akibatnya, pesan langsung dianggap **tidak mungkin** Spam, hanya karena satu kata tidak muncul di data latih.
   
   Untuk mengatasi hal ini digunakan **Laplace Smoothing** (atau *add-one smoothing*):
   - Tambahkan **+1** pada setiap hitungan kata.  
   - Tambahkan jumlah total kata unik pada penyebut.  
   
   Sehingga perhitungan berubah:
   
   $P(Lunch \mid Spam) = \frac{0 + 1}{7 + 4} = \frac{1}{11}$
   
   Dengan cara ini:
   - Probabilitas tidak pernah benar-benar **0**, hanya menjadi **sangat kecil**.  
   - Model jadi lebih **robust** terhadap kata-kata baru atau jarang muncul.
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

- **Asumsi independensi fitur**: Naive Bayes mengasumsikan bahwa fitur bersifat independen meskipun fitur bisa saja saling berkolerasi.
- **Sensitif terhadap class imbalance**: Karena Naive Bayes menggunakan probabilitas, label mayoritas dapat mendominasi prediksi dan membuat model bias.
- **Sensitif terhadap outlier**: Karena perhitungan probabilitas (terutama pada Gaussian Naive Bayes) sangat dipengaruhi oleh nilai ekstrem.

## Implementasi

Berikut adalah contoh implementasi dari salah satu varian Naive Bayes, yakni Gaussian Naive Bayes, menggunakan `scikit-learn`.

```python
from sklearn.naive_bayes import GaussianNB

# Matriks fitur
X_train = [
   [1.2, 0.7],
   [2.3, 1.9],
   [1.8, 2.2],
   [3.0, 3.1],
   [2.9, 0.2],
   [0.5, 1.7],
   [3.2, 2.8],
   [1.1, 0.9],
   # ...
]

# Vektor label
y_train = [
   0, 
   1, 
   1, 
   1, 
   0, 
   0, 
   1, 
   0, 
   # ...
]

# Data uji
X_test = [
   [1.5, 1.0],
   [2.8, 2.5],
   [0.7, 1.2],
   # ...
]

model = GaussianNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(pred)
```


To understand more , you guys can watch the youtube videos in the reference
## Referensi
- [Scikit-Learn - Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Youtube - Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=O2L2Uv9pdDA)
- [Youtube - Gaussian Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=H3EjCKtlVog)
