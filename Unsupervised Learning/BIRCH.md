# BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Definisi](#definisi)
- [Cara Kerja](#cara-kerja)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Definisi

BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) adalah algoritma clustering yang dirancang untuk menangani dataset berukuran besar secara efisien. Algoritma ini membangun struktur pohon hierarkis yang disebut CF Tree (Clustering Feature Tree) untuk meringkas data dan mempercepat proses pembentukan klaster.

## Cara Kerja

<img width="532" height="450" alt="image" src="https://github.com/user-attachments/assets/b4e5c137-f9b8-4211-8027-e1ea98e17f07" />

Secara garis besar, proses kerja BIRCH terdiri dari empat tahap utama:

1. Membangun CF Tree

   * Data dimasukkan satu per satu untuk membentuk Clustering Feature (CF) yang berisi ringkasan setiap sub-kluster:
     <p align="center">
      <h3><i>𝐶𝐹 = (𝑁, 𝐿𝑆, 𝑆𝑆)</i></h3>
     </p>
    
     di mana:
     - 𝑁 = jumlah titik data  
     - 𝐿𝑆 = penjumlahan linear semua titik (∑x)  
     - 𝑆𝑆 = penjumlahan kuadrat semua titik (∑x²)

   * Struktur pohon terbentuk dengan *branching factor* tertentu (jumlah maksimum anak per node).

2. Kompresi CF Tree (Optional)

   * Pohon dapat dipangkas untuk mengurangi ukuran memori jika terlalu besar, dengan cara menggabungkan node yang dekat satu sama lain.

3. Clustering Global

   * Setelah CF Tree terbentuk, centroid dari setiap leaf dapat dijadikan representasi data.
   * Kemudian, algoritma clustering lain seperti **k-means** atau **agglomerative clustering** dapat diterapkan pada centroid tersebut.

4. Refinement (Optional)

   * Hasil klaster dapat diperbaiki dengan melakukan *reclustering* terhadap data asli jika diperlukan.

## Kelebihan

- **Skalabilitas**: Dapat mengatasi dataset berukuran besar.
- **Jumlah kluster otomatis**: Tidak perlu mendefinisikan jumlah kluster.
- **Efisien**: Karena penggunaan CF, penggunaan memori menjadi lebih efisien.
- **Dapat mengatasi outlier**: CF dan threshold dapat mengisolasi outlier.

## Kekurangan

- **Sensitif terhadap parameter**: Hasil clustering bergantung pada parameter.
- **Asumsi data numerik**: Asumsi bahwa setiap vektor bersifat numerik dan berada di dalam geometri Euclidean, sehingga data kategorikal perlu preprocessing lebih lanjut.
- **Kinerja buruk untuk kluster berbentuk sembarangan**: Karena BIRCH menggunakan rangkuman radius/centroid dan jarak Euclidean, maka hanya berkinerja baik bagi cluster sferis saja.

## Implementasi

```python
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch

X, y = make_blobs()
birch = Birch(
    n_clusters=None, # Jumlah cluster (opsional)
    threshold=2, # Batas jarak antar nilai dalam cluster (opsional, digunakan jika n_clusters=None)
    branching_factor=50 # Branching factor
)
birch.fit(X)
y_pred = birch.predict(X)

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_pred).set_title('Hasil Clustering')
```

<img width="656" height="525" alt="Hasil Clustering Birch" src="https://github.com/user-attachments/assets/9f2bedd4-3800-43d8-99e9-e25971f510b2" />

## Reference
- [Birch](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html)
