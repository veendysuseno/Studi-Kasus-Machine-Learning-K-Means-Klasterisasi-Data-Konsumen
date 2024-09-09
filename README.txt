Penerapan Machine Learning dengan Pendekatan Unsupervised dengan Metode Algoritma K-Means

Definisi K-Means
Algoritma K-Means adalah metode klasterisasi yang sering digunakan dalam machine learning dan analisis data untuk mengelompokkan data ke dalam sejumlah kelompok (klaster) yang telah ditentukan sebelumnya. Tujuan utama dari algoritma ini adalah untuk mempartisi data menjadi kk klaster yang berbeda, di mana kk adalah jumlah klaster yang diinginkan. Berikut adalah penjelasan mendalam tentang metode K-Means:
Langkah-langkah Algoritma K-Means:

    Inisialisasi:
        Tentukan jumlah klaster kk.
        Pilih kk titik awal secara acak dari data sebagai pusat klaster awal (centroids). Metode pemilihan awal ini bisa menggunakan berbagai teknik, seperti pemilihan acak atau metode K-Means++.

    Penugasan Klaster:
        Untuk setiap titik data, tentukan klaster yang paling dekat dengan pusat klaster (centroid) menggunakan metrik jarak, biasanya jarak Euclidean.
        Titik data akan ditugaskan ke klaster dengan pusat terdekat.

    Update Pusat Klaster:
        Hitung ulang pusat klaster (centroid) untuk setiap klaster dengan mengambil rata-rata dari semua titik data yang telah ditugaskan ke klaster tersebut.

    Iterasi:
        Ulangi langkah penugasan klaster dan update pusat klaster sampai posisi pusat klaster stabil atau tidak berubah secara signifikan (konvergensi) atau sampai jumlah iterasi maksimum tercapai.

    Hasil Akhir:
        Setelah konvergensi, klasterisasi dianggap selesai, dan data telah dikelompokkan ke dalam kk klaster berdasarkan pusat klaster terakhir.

Detail Implementasi

1. Inisialisasi:

    Jumlah Klaster kk: Tentukan berapa banyak klaster yang ingin Anda bagi data.
    Pemilihan Pusat Klaster Awal: Pusat klaster dapat dipilih secara acak atau menggunakan metode K-Means++ untuk meningkatkan kinerja konvergensi.

2. Penugasan Klaster:

    Hitung jarak setiap titik ke semua pusat klaster dan tetapkan titik tersebut ke klaster dengan jarak terkecil.
    Rumus jarak Euclidean antara titik (x1,x2,...,xn)(x1​,x2​,...,xn​) dan pusat klaster (c1,c2,...,cn)(c1​,c2​,...,cn​) adalah:

distance=(x1−c1)2+(x2−c2)2+...+(xn−cn)2
distance=(x1​−c1​)2+(x2​−c2​)2+...+(xn​−cn​)2
​

3. Update Pusat Klaster:

    Hitung pusat klaster baru dengan rata-rata posisi titik data yang termasuk dalam klaster:

cj=1Nj∑i∈Cjxi
cj​=Nj​1​i∈Cj​∑​xi​

Di mana cjcj​ adalah pusat klaster ke-jj, NjNj​ adalah jumlah titik dalam klaster ke-jj, dan CjCj​ adalah himpunan titik dalam klaster ke-jj.

4. Iterasi:

    Proses ini diulang hingga pusat klaster stabil atau perubahan minimal pada iterasi berikutnya.

5. Hasil Akhir:

    Setelah konvergensi, klaster yang terbentuk adalah hasil akhir yang menggambarkan struktur data berdasarkan kk klaster yang telah ditentukan.

Contoh Implementasi dengan Python dan Scikit-Learn

Berikut adalah contoh implementasi K-Means menggunakan scikit-learn:

python

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

# Visualisasi hasil (hanya untuk 2 fitur pertama)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()

Kelebihan dan Kekurangan K-Means

Kelebihan:

    Sederhana dan mudah dipahami.
    Efisien untuk dataset besar.
    Hasil yang cepat dengan waktu komputasi yang relatif rendah.

Kekurangan:

    Memerlukan jumlah klaster kk yang sudah ditentukan.
    Sensitif terhadap inisialisasi pusat klaster.
    Tidak cocok untuk klaster yang tidak berbentuk bulat atau memiliki ukuran yang sangat berbeda.
    Tidak bekerja dengan baik jika klaster tumpang tindih atau jika data tidak terdistribusi secara seragam.

Dengan pemahaman ini, Anda dapat menggunakan K-Means untuk berbagai aplikasi klasterisasi dan analisis data.


Studi Kasus:
Sebuah perushaan melakukan penelitiann terhadap data-data konsumen yang dimilikinya. Perusahaan tersebut akan melakukan pengelompokkan data ke dalam beberapa cluster berdasarkan kriteria besaran gaji yang diterima dan pengeluaran per bulannya. Data-data konsumen terdiri dari 20 baris dengan 3 kolom.
Studi kasus kali ini akan melakukan klustering terhadap data-data konsumen di atas ke dalam beberapa kelompok, dimana masing-masing kelompok memiliki tingkat kemiripan maksimum. Tujuan penelitian ini adalah agar perusahaan dapat memetakan jenis produk yang sesuai dengan karakteristik konsumen.


@Copyright 2024 Veendy
