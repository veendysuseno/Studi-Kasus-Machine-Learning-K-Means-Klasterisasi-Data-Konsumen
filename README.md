# Penerapan Machine Learning dengan Pendekatan Unsupervised dengan Metode Algoritma K-Means

## Definisi K-Means

K-Means adalah algoritma klasterisasi yang digunakan untuk mengelompokkan data ke dalam klaster. Algoritma ini mempartisi data menjadi beberapa klaster berdasarkan kemiripan antar data, dengan tujuan agar setiap klaster memiliki karakteristik yang mirip satu sama lain.

## Langkah-langkah Algoritma K-Means

1. **Inisialisasi**: Tentukan jumlah klaster `k`, lalu pilih `k` pusat klaster secara acak atau menggunakan metode K-Means++.
2. **Penugasan Klaster**: Setiap titik data dihitung jaraknya dari pusat klaster dan ditugaskan ke klaster dengan pusat terdekat.
3. **Update Pusat Klaster**: Rata-rata posisi semua titik dalam klaster digunakan untuk memperbarui posisi pusat klaster.
4. **Iterasi**: Proses ini diulang sampai pusat klaster stabil atau jumlah iterasi maksimum tercapai.
5. **Hasil Akhir**: Data dikelompokkan ke dalam klaster yang telah terbentuk.

## Implementasi K-Means dengan Python dan Scikit-Learn

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data

# Inisialisasi dan latih model K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Prediksi klaster
y_kmeans = kmeans.predict(X)

# Visualisasi hasil (2 fitur pertama)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()
```

## Kelebihan dan Kekurangan K-Means

### Kelebihan K-Means :

- Sederhana dan mudah diterapkan.
- Efisien untuk dataset besar.
- Komputasi cepat.

### Kekurangan K-Means :

- Harus menentukan jumlah klaster k sebelumnya.
- Sensitif terhadap inisialisasi pusat klaster.
- Tidak cocok untuk klaster yang tidak berbentuk bulat atau ukuran yang bervariasi.

## Studi Kasus: Klasterisasi Data Konsumen

- Perusahaan melakukan pengelompokan data konsumen berdasarkan gaji dan pengeluaran bulanan. Data terdiri dari 20 baris dengan 3 kolom. Klasterisasi ini bertujuan untuk memetakan produk yang cocok dengan karakteristik setiap kelompok konsumen.

#### @Copyright 2024 | Veendy
