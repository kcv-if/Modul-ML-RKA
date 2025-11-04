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
