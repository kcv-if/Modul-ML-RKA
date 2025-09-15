# Decision Tree

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Pendahuluan](#pendahuluan)
- [Definisi](#definisi)
- [Cara Kerja](#cara-kerja)
- [Kelebihan](#kelebihan)
- [Kekurangan](#kekurangan)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

## Pendahuluan

## Definisi

## Cara Kerja

## Kelebihan

## Kekurangan
1) Overfitting , Decision Tree bisa tumbuh terlalu dalam dan rumit, menyesuaikan setiap detail termasuk noise (data yang tidak mewakili pola sebenarnya).misal jika ada 1 data outlier, pohon bisa membuat cabang khusus hanya untuk data itu = memicu overfitting
2) Perubahan kecil pada data (menambah/menghapus 1 sampel) bisa menghasilkan struktur pohon yang sangat berbeda.
3) Pembelajaran serakah (greedy) ,Pohon dibentuk dengan memilih split terbaik di tiap langkah secara lokal (greedy) sehingga tidak menjamin struktur pohon terbaik secara keseluruhan (global optimal).
4) Bias kelas , kelas mayoritas dapat mendominasi pembelahan (split), menyebabkan bias.

## Implementasi
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Buat dataset sintetis (40 sampel, 2 fitur)
np.random.seed(42)
X_class0 = np.random.randn(20, 2) * 0.5 + np.array([-1, -1])  # kelas 0
X_class1 = np.random.randn(20, 2) * 0.5 + np.array([1, 1])    # kelas 1
X = np.vstack([X_class0, X_class1])
y = np.array([0]*20 + [1]*20)

plt.figure(figsize=(6, 6))
plt.scatter(X_class0[:, 0], X_class0[:, 1], color='blue', label='Class 0')
plt.scatter(X_class1[:, 0], X_class1[:, 1], color='red', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Dataset: Two Clusters')
plt.legend()
plt.grid(True)
plt.show()

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Buat model Decision Tree
dt = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_leaf=2,
    random_state=42
)

# Training dan Evaluasi
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualisasi pohon
plt.figure(figsize=(8,8))
plot_tree(dt, filled=True, rounded=True, feature_names=['f0','f1'], class_names=['0','1'])
plt.show()
```

## Referensi
https://scikit-learn.org/stable/modules/tree.html
